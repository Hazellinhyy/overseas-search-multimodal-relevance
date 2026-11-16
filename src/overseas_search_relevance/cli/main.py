"""`osr` 命令行。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from overseas_search_relevance.data.audit import audit_dataset, load_model_input, write_profile
from overseas_search_relevance.evaluation.metrics import evaluate
from overseas_search_relevance.media.video import inspect_video


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _data_audit(args: argparse.Namespace) -> int:
    profile = audit_dataset(load_model_input(args.input), args.video_dir)
    write_profile(profile, args.output)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


def _media_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_video(args.video), ensure_ascii=False, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    gold_rows = _jsonl(args.gold)
    pred_rows = _jsonl(args.pred)
    predictions = {str(row["sample_id"]): row for row in pred_rows}
    gold: list[int] = []
    predicted: list[int] = []
    for row in gold_rows:
        sample_id = str(row["sample_id"])
        if sample_id in predictions:
            gold.append(int(str(row["relevance_label"])))
            predicted.append(int(str(predictions[sample_id]["relevance_label"])))
    print(json.dumps(asdict(evaluate(gold, predicted)), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osr", description="海外搜索多模态相关性标注与评测系统")
    groups = parser.add_subparsers(dest="group", required=True)
    data = groups.add_parser("data", help="数据工具")
    data_commands = data.add_subparsers(dest="command", required=True)
    audit = data_commands.add_parser("audit", help="审计模型输入与视频")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--video-dir", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(handler=_data_audit)
    media = groups.add_parser("media", help="媒体工具")
    media_commands = media.add_subparsers(dest="command", required=True)
    inspect = media_commands.add_parser("inspect", help="检查视频")
    inspect.add_argument("--video", type=Path, required=True)
    inspect.set_defaults(handler=_media_inspect)
    infer = groups.add_parser("infer", help="运行基线推理（需批准的模型配置）")
    infer.set_defaults(handler=lambda _args: print("未配置真实模型；请提供批准的服务端点") or 2)
    evaluation = groups.add_parser("evaluate", help="计算评测指标")
    evaluation.add_argument("--gold", type=Path, required=True)
    evaluation.add_argument("--pred", type=Path, required=True)
    evaluation.set_defaults(handler=_evaluate)
    report = groups.add_parser("report", help="报告入口")
    report.set_defaults(
        handler=lambda _args: print("报告由 data audit 与 evaluate 的真实输出生成") or 0
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
