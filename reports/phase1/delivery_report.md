# Phase 1 交付验收报告

执行日期：2026-11-16

## 仓库

- 名称：`overseas-search-relevance`
- 分支：`main`
- 远程仓库：BLOCKED。GitHub CLI 现有凭据无效，且没有创建远程仓库的明确授权。
- 原始数据：位于 Git 忽略目录。

## 模型与评测

- 统一模型协议、OpenAI-compatible 适配器和测试 Mock：PASS
- 企业批准的真实模型服务：BLOCKED（未提供端点、密钥和数据外发批准）
- 文本 Baseline 真实效果：NOT_EVALUATED
- 多模态 Baseline 真实效果：NOT_EVALUATED
- QWK / Exact / Adjacent / 混淆矩阵 / 分标签指标 / 延迟分位：PASS（工具实现）
- QWK 与 scikit-learn 二次加权 Kappa 交叉校验：PASS
- 真实 QWK、视觉增益、成本、吞吐和模型延迟：NOT_EVALUATED
- Badcase 候选生成器：PASS；真实案例分析：NOT_EVALUATED（缺少真实预测）

## 工程质量

- 测试：98 项通过，0 失败
- 覆盖率：80.84%；评分与指标核心模块 93%～100%
- Ruff：PASS
- Mypy strict：PASS
- Python sdist/wheel：PASS
- CLI 帮助与媒体烟测：PASS
- CI 配置：已建立；远程 CI 实际运行：NOT_EVALUATED

## 验收清单

- PASS：正式名称、本地 Git、工程结构、可安装包、CLI、四档标签、看后搜公式
- PASS：不少于 50 个规则案例、数据审计、视频处理、批处理恢复、JSONL 输出协议
- PASS：QWK 交叉验证、分层策略、中文注释、Ruff、Mypy、Pytest、构建、安全忽略
- BLOCKED：远程仓库、真实模型 API、真实文本/多模态 Baseline
- NOT_EVALUATED：真实 QWK、视觉增益、人工替代覆盖率、成本、吞吐、模型延迟、远程 CI
- NOT_DONE：20 条真实模型 Badcase 根因分析；缺少真实模型预测，不能伪造

## 风险与下一阶段

主标签中 2 分样本仅 11 条，类别分布明显不均衡；还有 5 个视频缺失。Phase 2 应先获得批准的
模型服务，使用分组切分后的开发/校准/验证集运行双 Baseline，再根据真实 Badcase 决定是否进行
路由或模型融合。当前没有训练、LoRA、隐藏集调参或 GPU 长期占用。

