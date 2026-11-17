# 动态 GPU 配置

所有 GPU 和模型参数均可由 KML 启动命令通过环境变量传入，不假设固定卡数。命令行参数优先于环境变量，未指定时使用自动发现值。

| 参数 | 默认值 | 类型/范围 |
|---|---|---|
| `GPU_COUNT` | 自动发现 | 1–1024 的整数；仅在显式设置时校验可见卡数 |
| `GPU_MEMORY_MIN_GIB` | 不限制 | 0.1–1024 的数字 |
| `TENSOR_PARALLEL_SIZE` | 自动/1（服务启动） | 正整数 |
| `PIPELINE_PARALLEL_SIZE` | 1 | 正整数 |
| `MODEL_PATH` | 无 | 本地模型目录，必须有 `config.json` |
| `MODEL_DTYPE` | `bfloat16` | `auto`/`bfloat16`/`float16`/`float32` |
| `MAX_MODEL_LEN` | 框架默认 | 1–2,000,000 的整数 |
| `MAX_NUM_SEQS` | 框架默认 | 1–100,000 的整数 |
| `GPU_MEMORY_UTILIZATION` | 0.9 | 0.1–0.99 |
| `MODEL_DOWNLOAD_ENABLED` | `false` | `true`/`false`，生产默认关闭 |
| `REMOTE_API_ENABLED` | `false` | `true`/`false`，生产默认关闭 |
| `ENABLE_TRAINING` | `false` | `true`/`false`，生产默认关闭 |

`TENSOR_PARALLEL_SIZE × PIPELINE_PARALLEL_SIZE` 必须不超过可见 GPU 数量，并能整除可见 GPU 数量。`gpu_diagnose.py` 会按实际可见卡数执行 CUDA、BF16、显存释放检查；使用 `torchrun --nproc_per_node=<可见卡数> scripts/gpu_diagnose.py` 才会执行 NCCL all-reduce。

示例：

```bash
GPU_COUNT=2 MODEL_PATH=/models/qwen \
TENSOR_PARALLEL_SIZE=2 PIPELINE_PARALLEL_SIZE=1 \
MODEL_DTYPE=bfloat16 MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.9 \
scripts/start_model_service.sh
```

预检与诊断：

```bash
GPU_COUNT=2 scripts/run_gpu_preflight.sh
torchrun --nproc_per_node=2 scripts/gpu_diagnose.py
```

脚本始终启用离线环境变量，不自动下载模型、不调用第三方 API，也不将 Gold Label 注入 Prompt。
