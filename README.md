# 海外搜索多模态相关性标注与评测系统

`overseas-search-relevance` 是 Phase 1 的可执行工程，覆盖规则、数据审计、媒体检查、
模型适配、批量恢复、置信度分层和四档有序标签评测。原始企业数据仅保存在 Git 忽略目录。

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
osr --help
osr data audit --input data/raw/model_input.json --video-dir data/raw/videos \
  --output reports/phase1/data_profile.json
osr media inspect --video data/raw/videos/150101226827701.mp4
osr evaluate --gold examples/evaluation.jsonl --pred examples/evaluation.jsonl
```

真实模型推理需要企业批准的服务端点和密钥。未配置时，CPU 数据审计、媒体处理、规则与
评测功能仍可运行；Mock 适配器只用于测试，禁止用于真实效果报告。

## GPU 平台

代码默认从 `/code` 运行，模型通过 `configs/gpu.json` 的 `model_path` 注入，不提交权重。
启动前执行 `bash scripts/run_gpu_preflight.sh`。脚本只读取 `nvidia-smi`、CPU 和磁盘，
不会下载模型或调用远程 API；未检测到 NVIDIA GPU 或显存不足时以 `BLOCKED` 退出。

KML 多卡部署入口：

```bash
bash scripts/run_gpu_preflight.sh
torchrun --nproc_per_node=8 scripts/gpu_diagnose.py --expected-gpus 8
bash scripts/start_model_service.sh
bash scripts/run_batch.sh
```

`gpu_diagnose.py` 才会验证 CUDA 张量、BF16、显存释放和 NCCL；仅 `nvidia-smi` 通过不代表多卡通信通过。
模型服务入口要求模型目录和 `vllm` 已由平台预装，离线模式运行，不会自动下载模型。

## 数据安全

`data/raw/`、`data/external/`、视频、工作簿和推理缓存默认不入库。安全解压器会拒绝路径
穿越、符号链接、文件数或解压体积超限。详见 `SECURITY.md`。

