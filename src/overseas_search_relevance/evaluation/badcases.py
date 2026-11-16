"""从真实预测差异生成候选 Badcase。"""

from __future__ import annotations

from typing import Any


def candidate_badcases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """筛选错误或失败样本；错误类型保留为可人工修订的候选，不伪造根因。"""

    candidates: list[dict[str, Any]] = []
    for row in rows:
        gold = row.get("gold_label")
        prediction = row.get("relevance_label")
        status = row.get("processing_status", "success")
        if status != "success" or gold != prediction:
            copy = dict(row)
            copy["badcase_type"] = (
                "processing_failure" if status != "success" else "unclassified_error"
            )
            candidates.append(copy)
    return candidates
