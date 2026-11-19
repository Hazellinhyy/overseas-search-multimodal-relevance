#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
export MODEL_NAME="${MODEL_NAME:-local-vlm}"
if [[ -x "$CODE_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$CODE_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "BLOCKED: 未找到项目 .venv/bin/python 或系统 python3" >&2
  exit 2
fi
exec "$PYTHON_BIN" -m overseas_search_relevance.web --host 127.0.0.1 --port "${WEB_PORT:-7860}"
