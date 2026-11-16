# 运行手册

1. 原始数据放入 `data/raw/`，确认 Git 状态不包含数据文件。
2. 运行 `osr data audit` 并检查 JSON 报告。
3. 仅在获批端点与数据传输范围内配置真实模型。
4. 失败样本保留 `failed` 或 `review`，不得改写为 0 分。
5. 发布前运行完整质量门禁并记录 commit hash。

