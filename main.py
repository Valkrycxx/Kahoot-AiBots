#!/usr/bin/env python3

import asyncio
import base64
import os
import random
import re
import shutil
import signal
import subprocess
import time
from collections import deque
from io import BytesIO

import httpx
import ollama
from PIL import Image
from playwright.async_api import async_playwright
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.text import Text

console = Console()

AI_MODEL = "batiai/gemma4-e4b:q4"
AI_SYSTEM_PROMPT = (
    "You are a kahoot expert. You will receive a question and 2-4 numbered options. "
    "Answer with ONLY the number of the correct option — a single digit, nothing else. "
    "No period, no explanation, no word, no quotes. Just the digit.\n\n"
)
AI_TIMEOUT_SECONDS = 12

VISION_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
VISION_HEALTH_URL = "http://127.0.0.1:8080/health"
VISION_TIMEOUT = 25

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
START_SERVER_SCRIPT = os.path.join(PROJECT_DIR, "start-server.sh")
VISION_DEBUG_LOG = os.path.join(PROJECT_DIR, "vision_debug.log")

ai_client = ollama.AsyncClient()


def _vlog(message):
    try:
        with open(VISION_DEBUG_LOG, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    except Exception:
        pass


async def query_ai(question, options):
    options_str = " ".join(f"{idx}. {txt}" for idx, txt in options)
    prompt = f"{AI_SYSTEM_PROMPT}QUESTION: {question}\nOPTIONS: {options_str}\nANSWER:"

    async def _call():
        response = await ai_client.chat(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            keep_alive=-1,
            options={"temperature": 0, "num_ctx": 2048, "num_predict": 8},
        )
        return response["message"]["content"].strip()

    try:
        text = await asyncio.wait_for(_call(), timeout=AI_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return None, f"TIMEOUT after {AI_TIMEOUT_SECONDS}s"
    except Exception as exc:
        return None, f"ERR: {exc}"

    match = re.search(r"\d", text)
    if match:
        idx = int(match.group())
        if any(idx == i for i, _ in options):
            return idx, text

    normalized = text.lower().strip().strip(".").strip('"').strip("'")
    for i, opt_text in options:
        if opt_text.lower().strip() == normalized:
            return i, text
    for i, opt_text in options:
        if opt_text.lower().strip() in normalized and len(opt_text.strip()) > 1:
            return i, text

    return None, text


async def vision_server_alive():
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(VISION_HEALTH_URL)
            return response.status_code == 200
    except Exception:
        return False


async def ensure_vision_server():
    if await vision_server_alive():
        console.print("[green]✓ Vision server is running[/green]")
        return True

    console.print("[yellow]! Vision server not detected at 127.0.0.1:8080[/yellow]")
    answer = console.input("Start it now via start-server.sh? [y/N]: ").strip().lower()
    if answer != "y":
        console.print("[dim]Skipping. Image questions will fall back to text-only AI.[/dim]")
        return False

    if not os.path.exists(START_SERVER_SCRIPT):
        console.print(f"[red]Script not found: {START_SERVER_SCRIPT}[/red]")
        return False

    console.print("[cyan]Starting vision server (loading ~6GB into VRAM, this takes ~20-30s)...[/cyan]")
    try:
        subprocess.Popen(
            ["bash", START_SERVER_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        console.print(f"[red]Failed to launch script: {exc}[/red]")
        return False

    for i in range(120):
        await asyncio.sleep(1)
        if await vision_server_alive():
            console.print(f"[green]✓ Vision server ready ({i + 1}s)[/green]")
            return True
        if (i + 1) % 10 == 0:
            console.print(f"[dim]  ... still loading ({i + 1}s)[/dim]")

    console.print("[red]✗ Server did not become ready in 120s[/red]")
    return False


def _convert_to_png_b64(raw_bytes):
    image = Image.open(BytesIO(raw_bytes))
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def _fetch_image_as_png_b64(url):
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        raw = response.content
    _vlog(f"FETCH {url[:80]} -> {len(raw)} bytes")
    encoded = await asyncio.to_thread(_convert_to_png_b64, raw)
    return encoded, len(raw)


async def query_vision(image_url, question, options):
    raw_bytes = 0
    try:
        img_b64, raw_bytes = await _fetch_image_as_png_b64(image_url)
    except Exception as exc:
        _vlog(f"IMG ERR: {exc}")
        return None, f"IMG ERR: {exc}", raw_bytes

    options_str = " ".join(f"{idx}. {txt}" for idx, txt in options)
    prompt = (
        f"QUESTION: {question}\n"
        f"OPTIONS: {options_str}\n"
        f"Answer with ONLY the number of the correct option."
    )
    _vlog(f"VISION Q: {question!r} | options={[txt for _, txt in options]}")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
        "max_tokens": 5,
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=VISION_TIMEOUT) as client:
            response = await client.post(VISION_SERVER_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        _vlog(f"VIS ERR: {exc}")
        return None, f"VIS ERR: {exc}", raw_bytes

    _vlog(f"VISION R: {text!r}")

    match = re.search(r"\d", text)
    if match:
        idx = int(match.group())
        if any(idx == i for i, _ in options):
            return idx, text, raw_bytes

    normalized = text.lower().strip().strip(".").strip('"').strip("'")
    for i, opt_text in options:
        if opt_text.lower().strip() == normalized:
            return i, text, raw_bytes
    for i, opt_text in options:
        if opt_text.lower().strip() in normalized and len(opt_text.strip()) > 1:
            return i, text, raw_bytes

    return None, text, raw_bytes


class AIAnswerStore:
    def __init__(self):
        self.question = None
        self.answer = None
        self.proposer = None
        self.lock = asyncio.Lock()

    async def try_claim(self, question, bot_id):
        async with self.lock:
            if self.question != question:
                self.question = question
                self.answer = None
                self.proposer = bot_id
                return True
            return False

    async def set_answer(self, question, answer):
        async with self.lock:
            if self.question == question:
                self.answer = answer

    async def get_answer(self, question):
        async with self.lock:
            if self.question == question:
                return self.answer
            return None


class Dashboard:
    __slots__ = (
        "total",
        "joined",
        "active",
        "questions_detected",
        "answers_given",
        "questions_ended",
        "failed",
        "bot_states",
        "events",
        "lock",
        "start_time",
        "current_question",
        "current_options",
        "current_answer",
        "current_source",
    )

    def __init__(self, total_bots):
        self.total = total_bots
        self.joined = 0
        self.active = 0
        self.questions_detected = 0
        self.answers_given = 0
        self.questions_ended = 0
        self.failed = 0
        self.bot_states = {i: {"status": "waiting", "question": ""} for i in range(total_bots)}
        self.events = deque(maxlen=8)
        self.lock = asyncio.Lock()
        self.start_time = time.time()
        self.current_question = ""
        self.current_options = []
        self.current_answer = ""
        self.current_source = ""

    async def set_bot_status(self, bot_id, status, question=""):
        async with self.lock:
            self.bot_states[bot_id]["status"] = status
            if question:
                self.bot_states[bot_id]["question"] = question

    async def increment_joined(self, bot_id):
        async with self.lock:
            self.joined += 1
            self.active += 1
            self.bot_states[bot_id]["status"] = "joined"

    async def increment_failed(self, bot_id):
        async with self.lock:
            self.failed += 1
            self.bot_states[bot_id]["status"] = "failed"

    async def increment_detected(self, bot_id, question):
        async with self.lock:
            self.questions_detected += 1
            self.bot_states[bot_id]["status"] = "detected"
            self.bot_states[bot_id]["question"] = question

    async def increment_answered(self, bot_id):
        async with self.lock:
            self.answers_given += 1
            self.bot_states[bot_id]["status"] = "answered"

    async def increment_ended(self, bot_id):
        async with self.lock:
            self.questions_ended += 1
            self.bot_states[bot_id]["status"] = "ended"
            self.bot_states[bot_id]["question"] = ""

    async def decrement_active(self, bot_id):
        async with self.lock:
            self.active -= 1
            self.bot_states[bot_id]["status"] = "left"

    async def add_event(self, message):
        async with self.lock:
            self.events.append(message)

    async def set_current_question(self, question, options, source=""):
        async with self.lock:
            if self.current_question != question or self.current_options != list(options):
                self.current_question = question
                self.current_options = list(options)
                self.current_answer = ""
                self.current_source = source

    async def set_current_answer(self, idx, text, source=""):
        async with self.lock:
            self.current_answer = f"#{idx}  {text}"
            if source:
                self.current_source = source

    async def get_stats(self):
        async with self.lock:
            return {
                "total": self.total,
                "joined": self.joined,
                "active": self.active,
                "detected": self.questions_detected,
                "answers_given": self.answers_given,
                "ended": self.questions_ended,
                "failed": self.failed,
                "events": list(self.events),
                "elapsed": int(time.time() - self.start_time),
                "bot_states": dict(self.bot_states),
                "current_question": self.current_question,
                "current_options": list(self.current_options),
                "current_answer": self.current_answer,
                "current_source": self.current_source,
            }


_terminal_cache = {"width": 80, "height": 24, "last_check": 0}


def get_terminal_size():
    now = time.time()
    if now - _terminal_cache["last_check"] > 2.0:
        try:
            width, height = shutil.get_terminal_size()
            _terminal_cache["width"] = width
            _terminal_cache["height"] = height
        except Exception:
            pass
        _terminal_cache["last_check"] = now
    return _terminal_cache["width"], _terminal_cache["height"]


def render_dashboard(stats, pin, mode_label):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="progress", size=3),
        Layout(name="stats", size=3),
        Layout(name="qa", size=9),
        Layout(name="grid", ratio=1),
        Layout(name="events", size=5),
    )

    header = Text()
    header.append("KAHOOT BOT DASHBOARD", style="bold blue")
    header.append(f"  |  PIN: {pin}", style="white")
    header.append(f"  |  Mode: {mode_label}", style="bold magenta")
    header.append(f"  |  Total: {stats['total']}", style="cyan")
    header.append(f"  |  Uptime: {stats['elapsed']}s", style="green")
    layout["header"].update(Panel(header, border_style="blue"))

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    task_id = progress.add_task("Joining", total=stats["total"])
    progress.update(task_id, completed=stats["joined"])
    layout["progress"].update(Panel(progress, title="Progress", border_style="green"))

    stats_text = Text()
    stats_text.append(f"Active: {stats['active']}  ", style="bold yellow")
    stats_text.append(f"Detected: {stats['detected']}  ", style="bold magenta")
    stats_text.append(f"Answered: {stats['answers_given']}  ", style="bold green")
    stats_text.append(f"Ended: {stats['ended']}  ", style="bold cyan")
    stats_text.append(f"Failed: {stats['failed']}", style="bold red")
    layout["stats"].update(Panel(stats_text, border_style="white"))

    qa_text = Text()
    question = stats.get("current_question") or "(waiting for question...)"
    source = stats.get("current_source", "")
    source_marker = f"  [{source}]" if source else ""
    qa_text.append("Q: ", style="bold yellow")
    qa_text.append(question + source_marker + "\n", style="white")

    answer = stats.get("current_answer", "")
    match = re.match(r"#(\d+)", answer)
    chosen_idx = int(match.group(1)) if match else None

    for idx, text in stats.get("current_options", []):
        marker = "▶ " if idx == chosen_idx else "  "
        style = "bold cyan" if idx == chosen_idx else "dim white"
        qa_text.append(f"{marker}{idx}. ", style=style)
        qa_text.append(f"{text}\n", style=style)

    if not stats.get("current_options"):
        qa_text.append("   (options not yet visible)\n", style="dim")

    qa_text.append("A: ", style="bold green")
    qa_text.append(answer or "(pending)", style="bold cyan")
    layout["qa"].update(Panel(qa_text, title="Current Question", border_style="magenta"))

    width, height = get_terminal_size()
    columns = max(1, (width - 4) // 2)
    if columns > 60:
        columns = 60

    items = list(stats["bot_states"].items())
    icon_map = {
        "waiting": "○",
        "joining": "◐",
        "joined": "●",
        "detected": "◉",
        "answered": "✓",
        "ended": "◼",
        "left": "↗",
        "failed": "✗",
        "lobby": "⟳",
    }

    rows = []
    for i in range(0, len(items), columns):
        row_items = items[i : i + columns]
        row_str = " ".join(icon_map.get(state["status"], "?") for _, state in row_items)
        rows.append(row_str)

    max_rows = max(1, height - 23)
    grid_text = Text()
    grid_text.append("\n".join(rows[:max_rows]))
    if len(rows) > max_rows:
        grid_text.append(f"\n... and {len(rows) - max_rows} more rows")

    legend = "Legend: ○waiting ◐joining ●joined ◉detected ✓answered ◼ended ↗left ✗failed ⟳lobby"
    layout["grid"].update(Panel(grid_text, title=legend, border_style="yellow"))

    events_text = Text()
    if stats["events"]:
        events_text.append("\n".join(stats["events"][-7:]), style="italic dim")
    else:
        events_text.append("No recent events")
    layout["events"].update(Panel(events_text, title="Events", border_style="white"))

    return layout


def get_user_input():
    console.print("\n[bold blue]KAHOOT BOT CONTROLLER[/bold blue]")
    console.print("=" * 50)

    while True:
        pin = console.input("Enter Kahoot game PIN: ").strip()
        if pin.isdigit() and len(pin) >= 4:
            break
        console.print("[red]Invalid PIN.[/red]")

    while True:
        bot_count = console.input("Number of bots to join (1-500): ").strip()
        if bot_count.isdigit():
            bot_count = int(bot_count)
            if 1 <= bot_count <= 500:
                break
        console.print("[red]Enter 1-500.[/red]")

    console.print("\n[bold]Answer mode:[/bold]")
    console.print("  [cyan]1[/cyan]. Random         - each bot picks randomly")
    console.print("  [cyan]2[/cyan]. Specific       - all bots pick the same fixed option")
    console.print("  [cyan]3[/cyan]. AI (text)      - one Ollama call per question")
    console.print("  [cyan]4[/cyan]. AI + Vision    - text AI + local vision server for image questions")

    while True:
        mode_choice = console.input("Choose [1/2/3/4]: ").strip()
        if mode_choice in ("1", "2", "3", "4"):
            break
        console.print("[red]Enter 1, 2, 3, or 4.[/red]")

    specific_answer = None
    if mode_choice == "1":
        mode, mode_label = "random", "random"
    elif mode_choice == "2":
        mode = "specific"
        while True:
            answer = console.input("Which option index? (0-3): ").strip()
            if answer.isdigit() and 0 <= int(answer) <= 3:
                specific_answer = int(answer)
                break
            console.print("[red]Enter 0-3.[/red]")
        mode_label = f"specific ({specific_answer})"
    elif mode_choice == "3":
        mode, mode_label = "ai", "AI text"
    else:
        mode, mode_label = "ai_vision", "AI + Vision"

    while True:
        concurrency = console.input("Join concurrency (recommended 10-20, default 15): ").strip()
        if not concurrency:
            concurrency = 15
            break
        if concurrency.isdigit():
            concurrency = int(concurrency)
            if 1 <= concurrency <= bot_count:
                break
        console.print(f"[red]Enter 1-{bot_count}.[/red]")

    console.print("\n[green]Starting bots... (Ctrl+C to stop)[/green]\n")
    return int(pin), bot_count, concurrency, mode, specific_answer, mode_label


READ_STATE_JS = """() => {
    const titleEl =
        document.querySelector('[data-functional-selector="question-title"]') ||
        document.querySelector('[data-functional-selector="block-title"]') ||
        document.querySelector('span[role="heading"]') ||
        document.querySelector('[class*="question-title"]');
    const title = titleEl ? titleEl.innerText.trim() : null;

    const options = [];
    for (let i = 0; i < 10; i++) {
        const btn = document.querySelector(`[data-functional-selector="answer-${i}"]`);
        if (!btn) continue;
        const r = btn.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        const txtEl = btn.querySelector(`[data-functional-selector="question-choice-text-${i}"]`)
                   || btn.querySelector('p');
        const text = txtEl ? txtEl.innerText.trim() : '';
        options.push({ index: i, text });
    }

    const imgEl = document.querySelector('[data-functional-selector="media-container__media-image"]');
    let imageUrl = null;
    if (imgEl) {
        imageUrl = imgEl.getAttribute('src') || imgEl.getAttribute('data-src') || null;
    }

    const gameOver = !!(document.querySelector('[data-functional-selector="game-over"]') ||
                        document.querySelector('.game-over-screen'));

    let lobby = false;
    for (const b of document.querySelectorAll('button')) {
        if ((b.innerText || '').toLowerCase().includes('start')) {
            lobby = true;
            break;
        }
    }
    if (!lobby) {
        lobby = !!(document.querySelector('.lobby-screen') ||
                   document.querySelector('[data-functional-selector="game-lobby"]'));
    }

    return { title, options, imageUrl, gameOver, lobby };
}"""


async def click_answer(page, idx):
    selector = f'[data-functional-selector="answer-{idx}"]'
    try:
        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=2000)
        box = await locator.bounding_box()
        if not box or box["width"] <= 0 or box["height"] <= 0:
            return False
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2
        await page.mouse.move(center_x, center_y)
        await asyncio.sleep(0.05)
        await page.mouse.down()
        await asyncio.sleep(0.05)
        await page.mouse.up()
        return True
    except Exception as exc:
        console.print(f"[red]click_answer({idx}) failed: {exc}[/red]")
        return False


async def bot_worker(
    browser,
    bot_id,
    pin,
    stop_event,
    dashboard,
    join_semaphore,
    mode,
    specific_answer,
    ai_store,
    vision_ok,
):
    joined_success = False
    context = await browser.new_context(
        viewport={"width": 400, "height": 300},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        ),
    )
    page = await context.new_page()

    await page.route(
        "**/*",
        lambda route: route.abort()
        if route.request.resource_type in {"image", "media", "font", "stylesheet", "other"}
        else route.continue_(),
    )

    try:
        async with join_semaphore:
            await dashboard.set_bot_status(bot_id, "joining")
            await page.goto("https://kahoot.it/", wait_until="commit", timeout=None)

            pin_input = await page.wait_for_selector('input[placeholder="Enter PIN"]', timeout=None)
            await pin_input.fill(str(pin))

            join_button = page.get_by_role("button", name="Join")
            await join_button.wait_for(state="visible", timeout=None)
            await join_button.click()

            nickname_input = await page.wait_for_selector(
                'input[placeholder="Enter your nickname"]', timeout=None
            )
            await nickname_input.fill(f"scraper{bot_id}")

            ok_button = page.get_by_role("button", name="OK, go!")
            await ok_button.wait_for(state="visible", timeout=None)
            await ok_button.click()

            await asyncio.sleep(0.5)
            joined_success = True
            await dashboard.increment_joined(bot_id)
            await dashboard.add_event(f"Bot {bot_id} joined")

        last_q_key = None
        answered_this_round = False
        ai_wait_started = None
        pending_q_key = None
        pending_count = 0
        poll_interval = 1.0

        while not stop_event.is_set():
            state = await page.evaluate(READ_STATE_JS)

            if state["gameOver"]:
                last_q_key = None
                answered_this_round = False
                ai_wait_started = None
                pending_q_key = None
                pending_count = 0
                await dashboard.set_bot_status(bot_id, "waiting")
                await asyncio.sleep(poll_interval)
                continue

            if state["lobby"]:
                last_q_key = None
                answered_this_round = False
                ai_wait_started = None
                pending_q_key = None
                pending_count = 0
                await dashboard.set_bot_status(bot_id, "lobby")
                await asyncio.sleep(poll_interval)
                continue

            if not state["title"]:
                if last_q_key is not None:
                    await dashboard.increment_ended(bot_id)
                    await dashboard.add_event(f"Bot {bot_id} question ended")
                last_q_key = None
                answered_this_round = False
                ai_wait_started = None
                pending_q_key = None
                pending_count = 0
                await asyncio.sleep(poll_interval)
                continue

            current_question = state["title"]
            options = [(option["index"], option["text"]) for option in state["options"]]
            valid_indices = [idx for idx, _ in options]
            image_url = state.get("imageUrl")

            option_texts_key = tuple(sorted(text for _, text in options))
            q_key = (current_question, option_texts_key, image_url)

            if q_key != pending_q_key:
                pending_q_key = q_key
                pending_count = 1
                await asyncio.sleep(poll_interval)
                continue

            pending_count += 1
            if pending_count < 2:
                await asyncio.sleep(poll_interval)
                continue

            if last_q_key is None or q_key != last_q_key:
                last_q_key = q_key
                answered_this_round = False
                ai_wait_started = None
                await dashboard.increment_detected(bot_id, current_question)
                await dashboard.set_current_question(
                    current_question,
                    options,
                    source=("IMG" if image_url else ""),
                )
                img_note = f" [IMG {image_url[-30:]}]" if image_url else ""
                await dashboard.add_event(
                    f"Bot {bot_id} Q: {current_question[:25]}...{img_note}"
                )

            if not options:
                await asyncio.sleep(poll_interval)
                continue

            if answered_this_round:
                await asyncio.sleep(poll_interval)
                continue

            chosen_idx = None

            if mode == "random":
                chosen_idx = random.choice(valid_indices)

            elif mode == "specific":
                chosen_idx = (
                    specific_answer
                    if specific_answer in valid_indices
                    else random.choice(valid_indices)
                )

            elif mode in ("ai", "ai_vision"):
                store_key = f"{current_question}|{hash(option_texts_key)}|{image_url or ''}"
                claimed = await ai_store.try_claim(store_key, bot_id)

                if claimed:
                    use_vision = (mode == "ai_vision") and vision_ok and image_url

                    if use_vision:
                        await dashboard.add_event(f"Bot {bot_id} -> Vision query")
                        ai_idx, raw, bytes_n = await query_vision(
                            image_url, current_question, options
                        )
                        await dashboard.add_event(
                            f"Vision fetch {bytes_n}B | raw: {raw[:40]!r}"
                        )
                    else:
                        if mode == "ai_vision" and not image_url:
                            await dashboard.add_event(f"Bot {bot_id} -> AI (no img)")
                        else:
                            await dashboard.add_event(f"Bot {bot_id} -> AI query")
                        ai_idx, raw = await query_ai(current_question, options)
                        await dashboard.add_event(f"AI raw: {raw[:40]!r}")

                    if ai_idx is None:
                        ai_idx = random.choice(valid_indices)
                        await dashboard.add_event(f"Fallback -> #{ai_idx}")

                    await ai_store.set_answer(store_key, ai_idx)
                    text_map = dict(options)
                    source_tag = "VISION" if use_vision else "TEXT"
                    await dashboard.set_current_answer(
                        ai_idx, text_map.get(ai_idx, "?"), source=source_tag
                    )
                    await dashboard.add_event(
                        f"Picked #{ai_idx}: {text_map.get(ai_idx, '')[:20]}"
                    )
                    chosen_idx = ai_idx

                else:
                    ai_idx = await ai_store.get_answer(store_key)

                    if ai_idx is None:
                        if ai_wait_started is None:
                            ai_wait_started = time.time()
                        elif time.time() - ai_wait_started > AI_TIMEOUT_SECONDS + 5:
                            chosen_idx = random.choice(valid_indices)
                            await dashboard.add_event(
                                f"Bot {bot_id} AI timeout -> random"
                            )
                            ai_idx = None

                        if chosen_idx is None:
                            await asyncio.sleep(poll_interval)
                            continue
                    else:
                        chosen_idx = ai_idx
                        text_map = dict(options)
                        await dashboard.set_current_answer(ai_idx, text_map.get(ai_idx, "?"))

            if chosen_idx is not None:
                clicked = await click_answer(page, chosen_idx)
                if clicked:
                    answered_this_round = True
                    await dashboard.increment_answered(bot_id)
                    await dashboard.add_event(f"Bot {bot_id} answered #{chosen_idx}")

            await asyncio.sleep(poll_interval)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        console.print(f"[red]Bot {bot_id} ERROR: {exc}[/red]")
        await dashboard.add_event(f"Bot {bot_id} error: {str(exc)[:30]}")
        await dashboard.increment_failed(bot_id)
    finally:
        if joined_success:
            await dashboard.decrement_active(bot_id)
        await page.close()
        await context.close()
        if joined_success:
            await dashboard.add_event(f"Bot {bot_id} left")


async def main():
    pin, bot_count, join_concurrency, mode, specific_answer, mode_label = get_user_input()

    vision_ok = False
    if mode in ("ai", "ai_vision"):
        vision_ok = await ensure_vision_server()
        if not vision_ok and mode == "ai_vision":
            console.print("[yellow]Continuing in text-only AI mode.[/yellow]")
            mode = "ai"
            mode_label = "AI text (vision unavailable)"

    stop_event = asyncio.Event()
    dashboard = Dashboard(bot_count)
    ai_store = AIAnswerStore()

    def signal_handler():
        console.print("\n[yellow]Shutting down...[/yellow]")
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images",
                "--blink-settings=imagesEnabled=false",
                "--disable-software-rasterizer",
            ],
        )
        console.print("[green]Chromium launched.[/green]")

        join_semaphore = asyncio.Semaphore(join_concurrency)
        tasks = []

        with Live(refresh_per_second=4, screen=True) as live:
            for i in range(bot_count):
                tasks.append(
                    asyncio.create_task(
                        bot_worker(
                            browser,
                            i,
                            pin,
                            stop_event,
                            dashboard,
                            join_semaphore,
                            mode,
                            specific_answer,
                            ai_store,
                            vision_ok,
                        )
                    )
                )
                await asyncio.sleep(0.08)

            while not stop_event.is_set():
                stats = await dashboard.get_stats()
                layout = render_dashboard(stats, pin, mode_label)
                live.update(layout)
                await asyncio.sleep(0.3)

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await browser.close()
        console.print("\n[bold green]All bots stopped.[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())