#!/usr/bin/env python3
"""GPU 平台启动前检查；不下载模型，不调用远程服务。"""

import argparse
from pathlib import Path

from overseas_search_relevance.infrastructure.gpu import (
    audit_environment,
    choose_safe_profile,
    write_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path("/code"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase1/gpu_preflight.json"))
    args = parser.parse_args()
    audit = audit_environment(args.code_root)
    profile = choose_safe_profile(audit)
    write_audit(args.output, audit, profile)
    print(profile)
    return 0 if profile["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
