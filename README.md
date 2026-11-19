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
torchrun --nproc_per_node="$GPU_COUNT" scripts/gpu_diagnose.py --gpu-count "$GPU_COUNT"
bash scripts/start_model_service.sh
bash scripts/run_batch.sh
```

`gpu_diagnose.py` 才会验证 CUDA 张量、BF16、显存释放和 NCCL；仅 `nvidia-smi` 通过不代表多卡通信通过。
模型服务入口要求模型目录和 `vllm` 已由平台预装，离线模式运行，不会自动下载模型。

## 本地 Web Demo + KML 模型

KML 上启动 vLLM 后，通过 SSH 将模型端口转发到 Mac 的 `127.0.0.1:8000`。Web Demo 仅绑定
本机地址，并且模型适配器拒绝非本机端点，避免竞赛数据流向第三方服务。

```bash
MODEL_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions \
MODEL_NAME=/approved/model/path \
bash scripts/start_web_demo.sh
```

浏览器打开 `http://127.0.0.1:7860`。输入 query、标题、ASR/OCR 及 Mac 上的视频路径；本地
FFmpeg 抽取关键帧，经 SSH 隧道发送给 KML 模型，返回四档标签、置信度、理由和 A/B/C 层级。
SSH/KML Token 只在终端临时使用，不得写入 `.env`、脚本或 Git。

正式评审协议最小字段为 `item_id + query`，并兼容公开集完整 metadata Schema。视频由评审方
挂载后按 `item_id` 匹配；交互式 Demo 也可接受受信任的 `video_url` 并自动抽帧。企业内部域名
必须加入 `VIDEO_URL_ALLOWLIST`。标题、ASR、OCR和本地路径属于可选增强输入。

批量页支持 JSON、JSONL、CSV、XLSX，以及“一个数据文件 + videos/目录”的 ZIP：

```text
batch.zip
├── samples.csv
└── videos/
    ├── 10001.mp4
    └── 10002.mp4
```

数据文件至少包含 `query` 和 `video_url`；若 ZIP 内含视频，可用 `sample_id/item_id` 与视频文件名
匹配。完成后可下载 JSONL 和 CSV，并查看失败率、吞吐、平均延迟和 A/B/C 分层。Gold 字段仅在
推理完成后用于评测，不进入 Prompt。

从真实结果生成评估报告：

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_evaluation_report.py \
  --input artifacts/jobs/<job_id>/results.jsonl \
  --output reports/phase1/evaluation_report.md
```

KML 最终批量打标入口、完整输入 Schema、视频目录约定和实测 QPS 见
[`docs/operations/labeling_interface.md`](docs/operations/labeling_interface.md)。

## 数据安全

`data/raw/`、`data/external/`、视频、工作簿和推理缓存默认不入库。安全解压器会拒绝路径
穿越、符号链接、文件数或解压体积超限。详见 `SECURITY.md`。

