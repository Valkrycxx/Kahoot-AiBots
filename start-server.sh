#!/bin/bash
# Start the Qwen3.5-4B vision server with CUDA GPU acceleration.

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/llama.cpp"

if pgrep -f "llama-server.*--port 8080" > /dev/null; then
    echo "Stopping existing server..."
    pkill -f "llama-server.*--port 8080" || true
    sleep 1
fi

echo "Starting Qwen3.5-4B vision server on http://127.0.0.1:8080 ..."

exec ./build/bin/llama-server \
  -m "$ROOT/models/Qwen_Qwen3.5-4B-Q4_K_M-vendor-sampling.gguf" \
  --mmproj "$ROOT/models/mmproj-BF16.gguf" \
  --jinja \
  --reasoning off \
  --flash-attn on \
  --image-min-tokens 512 \
  --image-max-tokens 512\
  --batch-size 2048 \
  --ubatch-size 2048 \
  -ngl 999 \
  -c 4096 \
  -np 1 \
  --host 127.0.0.1 \
  --port 8080