#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/code}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "批处理入口已预留；必须显式传入已验证模型服务和输入路径。"
echo "不会自动下载模型、调用第三方 API 或将 Gold Label 注入 Prompt。"
exec python -m overseas_search_relevance.cli.main infer "$@"
