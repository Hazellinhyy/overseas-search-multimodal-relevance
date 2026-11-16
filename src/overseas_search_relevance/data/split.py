"""按 Query、视频 ID 和近重复组进行数据隔离。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


def grouped_split(
    rows: list[dict[str, Any]], ratios: tuple[int, int, int] = (60, 20, 20)
) -> dict[str, list[dict[str, Any]]]:
    """确定性分组切分，保证同 Query 或同视频不会跨开发/校准/验证集泄漏。"""

    if any(value <= 0 for value in ratios) or sum(ratios) != 100:
        raise ValueError("切分比例必须为正整数且总和为 100")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query = " ".join(str(row.get("query", "")).casefold().split())
        item_id = str(row.get("item_id", row.get("video_id", "")))
        duplicate_group = str(row.get("near_duplicate_group", ""))
        key = duplicate_group or f"{query}|{item_id}"
        groups[key].append(row)
    result: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "calibration": [],
        "validation": [],
    }
    dev_boundary = ratios[0]
    calibration_boundary = ratios[0] + ratios[1]
    for key, group in groups.items():
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
        name = (
            "development"
            if bucket < dev_boundary
            else "calibration"
            if bucket < calibration_boundary
            else "validation"
        )
        result[name].extend(group)
    return result
