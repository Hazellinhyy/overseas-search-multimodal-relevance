# Phase 2F 正式 Git 发布与 KML 交付报告

执行日期：2026-11-17

## 发布信息

- 目标仓库：`https://github.com/Hazellinhyy/overseas-search-multimodal-relevance`
- 实际远程名称：`github-release`
- 发布分支：`feature/phase2f-kml-release`
- Commit SHA：`4c764267c8362301585131d99ace3f1ecd29ca7b`（工程提交）
- 推送结果：已成功推送到目标 Private 仓库；本报告更新提交随后同步推送。

## 工程变更

- Phase 2D：GPU 预检、GPU 配置、Dockerfile、`/code` 路径说明。
- Phase 2E：CUDA/BF16/NCCL 诊断入口、多卡配置、模型服务阻断入口、批处理入口。
- Phase 1 规则、看后搜评分公式、原有测试和数据审计功能未修改。

## 质量验证

- Ruff：PASS
- Mypy：PASS
- Pytest：98 passed
- 配置 JSON：PASS
- Shell 语法：PASS
- Python 编译：PASS
- Dockerfile 静态检查：PASS

## 数据安全

- 原始视频、Excel、Gold Label、模型权重和缓存未被 Git 跟踪。
- 未发现 Token、密码、API Key 或个人绝对路径。
- Dockerfile 仅复制源码、配置和脚本，不复制 `data/`、模型或缓存。
- 本阶段未下载模型、未启动训练、未运行完整 300 条评测。

## KML 配置

- 工作目录：`/code`
- Git 引用：分支
- Git 分支：`feature/phase2f-kml-release`
- 模型目录：通过 `configs/gpu.json` 的 `/models/qwen3.5-35b-a3b-instruct` 注入
- GPU 诊断：`torchrun --nproc_per_node=8 scripts/gpu_diagnose.py --expected-gpus 8`

## 尚未完成

- 需要 GitHub Private 仓库写入权限后完成真实推送。
- 需要在 KML A100×8 环境验证实际 CUDA、BF16 和 NCCL。
- 需要在模型已获批并挂载后验证 Qwen3.5-35B-A3B-Instruct 与 vLLM 兼容性。
