# Kahoot Bot Joiner + Local AI (with Vision)

A blazing-fast, fully local Kahoot botting tool with a real-time terminal dashboard, AI-powered answering, and GPU-accelerated vision support for image questions.

**Recommended for mid-end computers** (RTX 3060/4060-class GPU + 16GB RAM) if running vision mode, or if running more than ~35 bots. On 8GB RAM systems, keep bots under 50 and skip vision mode.

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Prerequisites](#software-prerequisites)
5. [Models](#models)
6. [Building llama.cpp](#building-llamacpp)
7. [Running the Vision Server](#running-the-vision-server)
8. [Testing the Vision Server](#testing-the-vision-server)
9. [Usage](#usage)
10. [Architecture](#architecture)
11. [Performance Reference](#performance-reference)
12. [Troubleshooting](#troubleshooting)
13. [Legal](#legal)
14. [Credits](#credits)

---

## Features

- **Real-time TUI dashboard** — live bot grid, question panel, event log
- **4 answer modes** — Random, Specific, AI (text-only), AI + Vision
- **Fully local inference** — no API keys, no cloud, no rate limits
- **Vision support** — reads clocks, maps, logos, fractions, and other image-based questions
- **Master-bot architecture** — one AI call per question regardless of bot count
- **GPU-accelerated** — RTX 4060 does vision prefill in ~360ms (33× faster than CPU)
- **Auto-detects question changes** — uses title + options + image URL fingerprint
- **Real mouse events** — works around Kahoot's React synthetic-click filter
- **Auto-starts vision server** — prompts to launch `start-server.sh` if not running

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Valkrycxx/Kahoot-AiBots.git
cd KahootBots

# 2. Install Python dependencies
python -m pip install -r requirements.txt
python -m playwright install chromium

# 3. Pull the text model into Ollama
ollama pull batiai/gemma4-e4b:q4

# 4. Download the vision GGUF files into ./models (Of course you can use your own models of choice, 
This a good light model, if you want better results seek looking into "Qwen_Qwen3.5-4B-Q4_K_M-vendor-sampling"
mkdir -p models && cd models
wget https://huggingface.co/batiai/Gemma-4-E4B-it-GGUF/resolve/main/google-gemma-4-E4B-it-Q4_K_M.gguf
wget https://huggingface.co/batiai/Gemma-4-E4B-it-GGUF/resolve/main/mmproj-BF16.gguf
cd ..

# 5. Build llama.cpp with GPU support (see Building llama.cpp for details)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build --config Release -j$(nproc)
cd ..

# 6. Start the vision server
./start-server.sh

# 7. Run the bot in a new terminal
python main.py
```

---

## Hardware Requirements

### Minimum (text-only AI)
| Component | Requirement |
| :--- | :--- |
| RAM | 8 GB |
| CPU | Any modern x86_64 |
| Disk | ~8 GB free |
| GPU | Not required |

### Recommended (vision + 100+ bots)
| Component | Requirement |
| :--- | :--- |
| RAM | 16 GB+ |
| CPU | 6+ cores |
| GPU | NVIDIA RTX 3060/4060 or better (8GB VRAM) |
| Disk | ~20 GB free |

**Tested configuration:** RTX 4060 (8GB VRAM), Arch Linux, Python 3.13, CUDA 12.x. Handles 150 bots stably; 200 max.

---

## Software Prerequisites

- **Python 3.10+**
- **Ollama** — local text inference ([install](https://ollama.com/download))
- **llama.cpp** — vision server (build it yourself, see below)
- **CUDA toolkit 12.x+** — for NVIDIA GPU acceleration
- **Playwright Chromium** — installed by `playwright install chromium`

---

## Models

### Text Model (Ollama)

Default: **Gemma 4 E4B** Q4 quantized (~5.3 GB).

```bash
ollama pull batiai/gemma4-e4b:q4
```

Set in `main.py`:
```python
AI_MODEL = 'batiai/gemma4-e4b:q4'
```

**Model comparison:**

| Model | Size | Strength | Weakness |
| :--- | :--- | :--- | :--- |
| `batiai/gemma4-e4b:q4` | 5.3 GB | Good vision + general knowledge | Older training cutoff (~2024) |
| `qwen3.5:4b` | 2.9 GB | Newer pop culture knowledge | Weak vision |
| `qwen3:4b-instruct` | 2.6 GB | Fast, no thinking mode | Older knowledge |
| `llama3.2:3b` | 2.0 GB | Fastest | Least accurate |

**Honest assessment:** all local models under 7B top out at roughly **70% accuracy** on Kahoot trivia. Recent-events questions (last 12 months) will fail often.

### Vision Model (llama.cpp)

Two files are required. The main GGUF handles text + reasoning; the projector (`mmproj`) handles the image encoder.

| File | Purpose | Size |
| :--- | :--- | :--- |
| `google-gemma-4-E4B-it-Q4_K_M.gguf` | Main vision model | ~5.0 GB |
| `mmproj-BF16.gguf` | Vision projector | ~950 MB |

Download into `./models/`:

```bash
mkdir -p models && cd models
wget https://huggingface.co/batiai/Gemma-4-E4B-it-GGUF/resolve/main/google-gemma-4-E4B-it-Q4_K_M.gguf
wget https://huggingface.co/batiai/Gemma-4-E4B-it-GGUF/resolve/main/mmproj-BF16.gguf
```

If either URL 404s, open the [repo Files tab](https://huggingface.co/batiai/Gemma-4-E4B-it-GGUF/tree/main) and copy the current filenames.

**Why two files?** The base Gemma 4 model is multimodal, but in the GGUF ecosystem the vision encoder is packaged separately. `llama-server` loads both and combines them at inference time. Without `--mmproj`, the server is text-only.

**Alternative vision models** (same two-file pattern):

| Model | Repo |
| :--- | :--- |
| Qwen3.5-4B | `NobodyWho/Qwen_Qwen3.5-4B-GGUF` |
| Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-GGUF` |
| MiniCPM-V-4.6 | `openbmb/MiniCPM-V-4_6-gguf` |

---

## Building llama.cpp

`llama.cpp` isn't pre-installed — you compile it yourself for your specific GPU. This is what makes vision fast. You only need to do this once.

### Clone

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

### Build — NVIDIA (CUDA)

```bash
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build --config Release -j$(nproc)
```

**`CMAKE_CUDA_ARCHITECTURES` must match your GPU's compute capability:**

| GPU | Architecture code |
| :--- | :--- |
| RTX 20-series (Turing) | `75` |
| RTX 30-series (Ampere) | `86` |
| **RTX 40-series (Ada)** | **`89`** |
| RTX 50-series (Blackwell) | `120` |
| A100 / H100 (data center) | `80` / `90` |

Check yours:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Get it wrong and you'll get mysterious runtime errors.

### Build — AMD / Intel (Vulkan)

```bash
rm -rf build
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
```

Vulkan works on nearly every modern GPU. Performance is roughly 70–80% of CUDA.

### Build — CPU Only

```bash
rm -rf build
cmake -B build
cmake --build build --config Release -j$(nproc)
```

Expect ~7 tokens/sec generation and ~12 seconds of image prefill. Vision is technically usable but too slow for Kahoot timers.

### Verify the Build

```bash
ls build/bin/ | grep -E "llama-server|llama-mtmd-cli|llama-cli"
ldd build/bin/llama-server | grep -i cuda    # should print CUDA libs
grep -i cuda build/CMakeCache.txt             # should show GGML_CUDA:BOOL=ON
```

If `ldd` prints nothing, the binary is CPU-only despite the flag. Most common cause: `nvcc` wasn't on `PATH` when CMake ran.

```bash
export PATH=/opt/cuda/bin:$PATH
export CUDACXX=/opt/cuda/bin/nvcc
```

### Rebuilding After Updates

```bash
cd llama.cpp
git pull
cmake --build build --config Release -j$(nproc)
```

You **don't** need to `rm -rf build` unless you're switching backends (CUDA ↔ Vulkan ↔ CPU).

---

## Running the Vision Server

Run `./start-server.sh` from the project root. The script:

1. Kills any existing `llama-server` on port 8080
2. Launches a fresh server with the CUDA-accelerated binary
3. Loads the main GGUF + `mmproj` + chat template

**Expected startup output:**

```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4060, compute capability 8.9
load_tensors: offloading 29 repeating layers to GPU
load_tensors: offloaded 29/29 layers to GPU
model loaded
listening on http://127.0.0.1:8080
```

**If you don't see the `offloaded X/Y layers to GPU` line, CUDA isn't active.** Fall back to the `ldd` check above.

### Verify GPU Usage

In a separate terminal, while the server runs:

```bash
nvidia-smi
```

`llama-server` should show ~6 GB VRAM and non-zero GPU utilization.

---

## Testing the Vision Server

### Method 1 — curl (quick check)

```bash
cd ~/projects/KahootBots

python3 -c "
import json, base64
img = base64.b64encode(open('test.png','rb').read()).decode()
json.dump({'messages':[{'role':'user','content':[
  {'type':'text','text':'Answer with ONLY the correct option text. Q: What fraction is shown?'},
  {'type':'image_url','image_url':{'url':f'data:image/png;base64,{img}'}}
]}], 'max_tokens': 5}, open('/tmp/payload.json','w'))
" && curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -d @/tmp/payload.json | python3 -m json.tool
```

Expected response:

```json
{
  "choices": [{"message": {"content": "1/3"}}],
  "timings": {
    "prompt_ms": 361,
    "predicted_per_second": 45.9
  }
}
```

**Interpret the timings:**
- `prompt_ms` < 500 → GPU active. If > 5000 → CPU.
- `predicted_per_second` > 30 → GPU generation. If < 10 → CPU.

### Method 2 — `llama-mtmd-cli` (isolated, ignores server)

Useful for isolating server issues from model issues:

```bash
cd llama.cpp
./build/bin/llama-mtmd-cli \
  -m ../models/google-gemma-4-E4B-it-Q4_K_M.gguf \
  --mmproj ../models/mmproj-BF16.gguf \
  --image ../test.png \
  --jinja \
  -p "Describe this image in one sentence."
```

**Caveat:** `llama-mtmd-cli` doesn't support `--reasoning off` in most builds. The model will think out loud and be slow. Use it only to verify that vision works, then use the server for real inference.

### Canary Questions

Test a new model with these before trusting it in a game:

| Question | Correct answer |
| :--- | :--- |
| "What fraction is shown?" — image with 2 of 6 segments filled | `1/3` |
| "What time is shown on the clock?" — analog clock | varies |
| "Which country is highlighted?" — world map | varies |

If a model fails the fraction question, **don't use it** — it's a canary for weak vision.

---

## Usage

Run `main.py` and answer the prompts:

```
KAHOOT BOT CONTROLLER
==================================================
Enter Kahoot game PIN: 123456
Number of bots to join (1-500): 50

Answer mode:
  1. Random         - each bot picks randomly
  2. Specific       - all bots pick the same fixed option
  3. AI (text)      - one Ollama call per question
  4. AI + Vision    - text AI + local vision server for image questions
Choose [1/2/3/4]: 4

Join concurrency (recommended 10-20, default 15): 15
```

### Mode Selection Guide

| Mode | Best for |
| :--- | :--- |
| **Random** | Filling lobbies, testing connectivity |
| **Specific** | Answer-distribution analysis, controlled tests |
| **AI (text)** | Text-only games with fast timers |
| **AI + Vision** | Mixed games with image questions |

### Status Icons

| Icon | Meaning |
| :--- | :--- |
| ○ | Waiting |
| ◐ | Joining |
| ● | Joined |
| ◉ | Question detected |
| ✓ | Answered |
| ◼ | Question ended |
| ↗ | Left |
| ✗ | Failed |
| ⟳ | In lobby |

### Debugging

Every vision request/response is logged to `vision_debug.log`:

```bash
tail -f vision_debug.log
```

Log format:
```
[HH:MM:SS] FETCH <url> -> <bytes>
[HH:MM:SS] VISION Q: '<question>' | options=[...]
[HH:MM:SS] VISION R: '<model output>'
[HH:MM:SS] IMG ERR: <error>
```

If `FETCH` shows bytes but the answer is wrong, the model is the problem, not the pipeline.

---

## Architecture

### Master-Bot Pattern

Only **one bot** queries the AI per question. The rest read the cached answer from an in-memory store. Result:

- 1 AI call per question, regardless of running 5 or 500 bots
- Consistent answers across all bots
- Minimal CPU/GPU/RAM pressure

The first bot to detect a new question "claims" it and becomes the proposer. Every other bot polls the shared store on its next cycle (≤1s) and clicks the same option index.

### Question Fingerprint

A new question is detected when **any** of these change:

- Question title text
- Sorted list of option texts
- Image URL

This solves the "same title, different options" bug: two consecutive Kahoot rounds can both ask "name the sports team" with completely different options. Title-only matching breaks here.

### Settle Logic

A new fingerprint must be observed **twice in a row** (1s apart) before the bot acts. This filters the transient DOM state where the title has updated but options haven't rendered yet.

### Real Mouse Events

Kahoot's buttons are React components listening for `pointerdown` / `pointerup`. Synthetic `element.click()` gets ignored. The bot dispatches a real `page.mouse.move → down → up` sequence at the button's coordinates.

### Vision Pipeline

1. Detect image via `[data-functional-selector="media-container__media-image"]`
2. Download the WebP image with `httpx`
3. Convert to PNG via Pillow (llama.cpp can't decode WebP)
4. Base64-encode and POST to `http://127.0.0.1:8080/v1/chat/completions`
5. Parse the response for a digit `0-3` or matching option text
6. Fall back to random only if parsing fails

### Memory Layout

The two AI paths use **separate memory pools** and don't compete:

| Path | Model | Memory pool | Size |
| :--- | :--- | :--- | :--- |
| Text (Ollama) | Gemma 4 E4B | System RAM | 5.3 GB |
| Vision (llama-server) | Gemma 4 E4B | GPU VRAM | 6.25 GB |

This is why both can run simultaneously. To also move Ollama's model to GPU, set `OLLAMA_NUM_GPU=999`.

---

## Performance Reference

Measured on RTX 4060 (8GB) + Ryzen + 16GB RAM, Arch Linux:

### Vision Server

| Metric | CPU only | RTX 4060 (CUDA) | Speedup |
| :--- | :--- | :--- | :--- |
| Image prefill (`prompt_ms`) | 12,058 ms | **361 ms** | **33×** |
| Token generation | 6.5 t/s | **45.9 t/s** | **7×** |
| Total per image question | ~34 s | **~0.43 s** | **79×** |

### Text (Ollama)

| Operation | Time |
| :--- | :--- |
| Question → answer (warm) | ~0.22 s |
| Question → answer (cold start) | ~1.5 s |

### Bot Scaling

| RAM | Max bots (AI mode) | Max bots (Vision mode) |
| :--- | :--- | :--- |
| 8 GB | 50 | 30 |
| 16 GB | 150 | 150 |
| 32 GB | 400 | 400 |

**Formula:** each Chromium context uses ~40–60 MB. Budget ~100 MB per bot for safety.

---

## Troubleshooting

### "Vision server not detected"

```bash
lsof -i :8080                       # check for stale process
pkill -f llama-server               # kill it
./start-server.sh                   # restart
```

### Bots joined but host shows 0 answers

You're hitting Kahoot's synthetic-click filter. Confirm `main.py` uses `click_answer()` with `page.mouse.down/up`, not `element.click()`.

### AI answers wrong / hallucinates

**4B models are ~70% accurate on Kahoot trivia.** Recent events (2025+) will fail often. Options:

1. **Add Groq for text** — free tier, `llama-3.3-70b-versatile`, 1,000 req/day.
2. **Use Gemma 4 for vision only** — best 4B vision model you can run locally.
3. **Accept the ceiling** — you cannot run 70B on 8GB RAM.

### Model returns `1/2` for a `1/3` fraction

Vision model limitation, not a config issue. Gemma 4 E4B handles fractions correctly; Qwen3.5-4B does not. Use Gemma 4 for vision.

### `invalid argument: --reasoning`

Your `llama.cpp` build predates the flag. Either:

- Update: `cd llama.cpp && git pull && cmake --build build --config Release -j$(nproc)`
- Or drop the flag — the server still works, but the model will think out loud and be slow.

### `Argument list too long` in curl

Base64 exceeds the shell argument limit. Write the payload to a file with Python first (see Testing section).

### `unknown projector type` from llama-server

The `mmproj` file doesn't match the main model's architecture. Re-download both from the same repo — mixing a Qwen projector with a Gemma model won't work.

### Flash Attention produces garbage

Some builds are unstable with `--flash-attn on`. Switch to `off`. Performance drops ~15% but output quality recovers.

### CUDA build succeeds but `-ngl 999` doesn't offload

```bash
ldd build/bin/llama-server | grep cuda
```

If nothing prints, the binary is CPU-only. `nvcc` wasn't on `PATH` when CMake ran:

```bash
export PATH=/opt/cuda/bin:$PATH
rm -rf build && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build --config Release -j$(nproc)
```

### Out of memory / system freezes

Exceeded the RAM formula. Lower bot count, or drop to text-only mode.

---

## Legal

This tool is for **educational purposes** and personal experimentation with LLM inference pipelines. Using it to flood, disrupt, or manipulate public Kahoot games violates Kahoot's Terms of Service and may constitute abuse. Don't do that.

Only run this on private games you control.

---

## Credits

- [Ollama](https://ollama.com) — local text inference
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — vision server
- [BatiAI](https://huggingface.co/batiai) — quantized Gemma 4 GGUF files
- [Playwright](https://playwright.dev) — headless browser automation
- [Rich](https://github.com/Textualize/rich) — TUI framework
