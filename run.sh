#!/bin/bash
# Voice Clone Studio: starts the local server on 127.0.0.1 and opens the browser.
set -e
cd "$(dirname "$0")"
PORT="${PORT:-7861}"
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Create it first (see README.md, 'Install')."; exit 1
fi
# Local only: no telemetry; use model weights from the local cache (the Setup screen's
# 'Install weights' button lifts HF_HUB_OFFLINE only for that one approved download).
export HF_HUB_DISABLE_TELEMETRY=1 GRADIO_ANALYTICS_ENABLED=False DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
URL="http://127.0.0.1:$PORT"
if [ -z "$NO_BROWSER" ]; then
  ( for i in $(seq 1 60); do curl -s -o /dev/null "$URL/api/steps" && { open "$URL"; break; }; sleep 0.5; done ) &
fi
echo "Voice Clone Studio on $URL  (Ctrl+C to stop)"
exec .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT" --log-level warning
