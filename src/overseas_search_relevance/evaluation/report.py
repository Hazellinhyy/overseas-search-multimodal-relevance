"""由真实 JSONL 推理结果生成评估报告，不补造缺失指标。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from overseas_search_relevance.evaluation.badcases import candidate_badcases
from overseas_search_relevance.evaluation.metrics import evaluate


def build_report(rows: list[dict[str, Any]]) -> str:
    successful = [row for row in rows if row.get("processing_status") == "success"]
    evaluated = [
        row
        for row in successful
        if str(row.get("gold_label", "")).strip() in {"0", "1", "2", "3"}
    ]
    discarded = [row for row in successful if row.get("gold_status") == "discarded"]
    media_missing = [row for row in successful if row.get("media_status") == "missing"]
    lines = ["# 多模态相关性效果评估报告", "", "## 运行概况", ""]
    lines.extend(
        [
            f"- 总样本：{len(rows)}",
            f"- 成功：{len(successful)}",
            f"- 有效 Gold：{len(evaluated)}",
            f"- 官方丢弃：{len(discarded)}",
            f"- 视频缺失/文本降级：{len(media_missing)}",
            f"- 失败率：{(1 - len(successful) / len(rows)):.2%}" if rows else "- 失败率：n.a.",
            f"- A/B/C：{dict(Counter(str(row.get('tier')) for row in successful))}",
        ]
    )
    latencies = [float(row.get("latency_ms", 0)) for row in successful]
    if latencies:
        ordered = sorted(latencies)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
        lines.append(f"- 平均延迟：{sum(latencies) / len(latencies):.2f} ms")
        lines.append(f"- P50/P95：{p50:.2f} / {p95:.2f} ms")
    lines.extend(["", "## Gold 指标", ""])
    if evaluated:
        metrics = evaluate(
            [int(row["gold_label"]) for row in evaluated],
            [int(row["relevance_label"]) for row in evaluated],
        )
        values = asdict(metrics)
        lines.extend(
            [
                f"- 评测条数：{metrics.count}",
                f"- QWK：{metrics.qwk}",
                f"- Exact Accuracy：{metrics.exact_accuracy}",
                f"- ±1 Adjacent Accuracy：{metrics.adjacent_accuracy}",
                f"- 混淆矩阵：`{values['confusion_matrix']}`",
            ]
        )
    else:
        lines.append("未提供 Gold Label，指标保持 n.a.，不伪造结果。")
    visual = [row for row in evaluated if row.get("visual_strong")]
    comparable = [row for row in visual if row.get("text_baseline_label") not in (None, "")]
    lines.extend(["", "## 视觉强依赖子集", ""])
    if comparable:
        multi = evaluate(
            [int(row["gold_label"]) for row in comparable],
            [int(row["relevance_label"]) for row in comparable],
        )
        text = evaluate(
            [int(row["gold_label"]) for row in comparable],
            [int(row["text_baseline_label"]) for row in comparable],
        )
        gain = multi.qwk - text.qwk if multi.qwk is not None and text.qwk is not None else None
        lines.extend(
            [f"- 多模态 QWK：{multi.qwk}", f"- 纯文本 QWK：{text.qwk}", f"- QWK 增益：{gain}"]
        )
    else:
        lines.append("缺少 visual_strong 与 text_baseline_label 配对结果，增益保持 n.a.。")
    lines.extend(["", "## 典型 Case（最多20条）", ""])
    cases = candidate_badcases(evaluated)[:20]
    if not cases:
        lines.append("当前没有可生成的真实错误 Case。")
    for index, row in enumerate(cases, 1):
        lines.extend(
            [
                f"### Case {index}: {row.get('sample_id')}",
                "",
                f"- Gold / Prediction：{row.get('gold_label')} / {row.get('relevance_label')}",
                f"- Tier / Confidence：{row.get('tier')} / {row.get('confidence')}",
                f"- 模型理由：{row.get('reason', '')}",
                "- 人工归因：待复核（不得由模型伪造根因）",
                "",
            ]
        )
    return "\n".join(lines)


def generate_report(input_path: Path, output_path: Path) -> None:
    rows = [
        json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(rows), encoding="utf-8")
