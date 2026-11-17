#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/code}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/gpu_preflight.py --code-root "$CODE_ROOT" "$@"
