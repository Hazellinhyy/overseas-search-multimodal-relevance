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

## 数据安全

`data/raw/`、`data/external/`、视频、工作簿和推理缓存默认不入库。安全解压器会拒绝路径
穿越、符号链接、文件数或解压体积超限。详见 `SECURITY.md`。

