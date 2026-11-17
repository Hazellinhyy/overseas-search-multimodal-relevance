#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/code}"
MODEL_PATH="${MODEL_PATH:-/models/qwen3.5-35b-a3b-instruct}"
cd "$CODE_ROOT"
test -d "$MODEL_PATH" || { echo "BLOCKED: MODEL_PATH 不存在: $MODEL_PATH" >&2; exit 2; }
command -v vllm >/dev/null 2>&1 || { echo "BLOCKED: vLLM 未安装；禁止自动下载" >&2; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec vllm serve "$MODEL_PATH" --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-8}" --dtype bfloat16 --download-dir "${MODEL_CACHE:-/models/cache}" "$@"
