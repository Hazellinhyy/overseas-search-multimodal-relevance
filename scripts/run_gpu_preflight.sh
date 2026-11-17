#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/code}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(--code-root "$CODE_ROOT")
[[ -n "${GPU_COUNT:-}" ]] && ARGS+=(--gpu-count "$GPU_COUNT")
exec python scripts/gpu_preflight.py "${ARGS[@]}" "$@"
