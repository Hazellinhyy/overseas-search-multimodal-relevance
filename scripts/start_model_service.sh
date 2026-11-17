#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/code}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MODEL_DOWNLOAD_ENABLED="${MODEL_DOWNLOAD_ENABLED:-false}"
REMOTE_API_ENABLED="${REMOTE_API_ENABLED:-false}"
cd "$CODE_ROOT"
[[ -n "$MODEL_PATH" ]] || { echo "BLOCKED: 必须显式设置 MODEL_PATH" >&2; exit 2; }
[[ "$MODEL_DOWNLOAD_ENABLED" == "false" && "$REMOTE_API_ENABLED" == "false" ]] || { echo "BLOCKED: 模型下载和远程 API 必须保持关闭" >&2; exit 2; }
[[ "$MODEL_DTYPE" =~ ^(auto|bfloat16|float16|float32)$ ]] || { echo "BLOCKED: MODEL_DTYPE 非法" >&2; exit 2; }
python - <<'PY'
import os
from pathlib import Path
import json
path = Path(os.environ["MODEL_PATH"])
if not (path / "config.json").is_file(): raise SystemExit("BLOCKED: 模型 config.json 不存在")
config = json.loads((path / "config.json").read_text())
if not config.get("architectures") and not config.get("model_type"): raise SystemExit("BLOCKED: 模型架构元数据缺失")
PY
test -d "$MODEL_PATH" || { echo "BLOCKED: MODEL_PATH 不存在: $MODEL_PATH" >&2; exit 2; }
command -v vllm >/dev/null 2>&1 || { echo "BLOCKED: vLLM 未安装；禁止自动下载" >&2; exit 2; }
VISIBLE_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[[ "$VISIBLE_GPUS" =~ ^[0-9]+$ && "$VISIBLE_GPUS" -gt 0 ]] || { echo "BLOCKED: 未检测到可用 GPU" >&2; exit 2; }
TP="${TENSOR_PARALLEL_SIZE:-1}"
[[ "$TP" =~ ^[1-9][0-9]*$ && "$PIPELINE_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BLOCKED: TP/PP 必须为正整数" >&2; exit 2; }
(( TP * PIPELINE_PARALLEL_SIZE <= VISIBLE_GPUS && VISIBLE_GPUS % (TP * PIPELINE_PARALLEL_SIZE) == 0 )) || { echo "BLOCKED: TP×PP 与可见 GPU 数量不兼容" >&2; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD=(vllm serve "$MODEL_PATH" --tensor-parallel-size "$TP" --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" --dtype "$MODEL_DTYPE" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --download-dir "${MODEL_CACHE:-/models/cache}")
[[ -n "${MAX_MODEL_LEN:-}" ]] && CMD+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "${MAX_NUM_SEQS:-}" ]] && CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
exec "${CMD[@]}" "$@"
