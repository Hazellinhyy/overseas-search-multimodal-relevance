"""有界重试、缓存、原子写入和断点恢复批处理。"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def cache_key(
    payload: dict[str, Any],
    *,
    model_version: str,
    prompt_version: str,
    rule_version: str,
    media_config: str,
) -> str:
    """缓存键包含所有影响结果的版本，避免错误复用旧推理结果。"""

    material = {
        "payload": payload,
        "model_version": model_version,
        "prompt_version": prompt_version,
        "rule_version": rule_version,
        "media_config": media_config,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("processing_status") == "success":
            completed.add(str(row.get("sample_id")))
    return completed


def run_batch(
    rows: Iterable[dict[str, Any]],
    processor: Callable[[dict[str, Any]], dict[str, Any]],
    output: Path,
    *,
    retries: int = 2,
    budget: int | None = None,
) -> int:
    """逐条原子追加并支持恢复；失败样本保留失败状态，绝不自动记为 0 分。"""

    if retries < 0 or retries > 10:
        raise ValueError("retries 必须在 0～10")
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output)
    written = 0
    with output.open("a", encoding="utf-8") as stream:
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            if sample_id in completed:
                continue
            if budget is not None and written >= budget:
                break
            for attempt in range(retries + 1):
                try:
                    result = processor(row)
                    break
                except Exception as error:  # 边界层将异常编码为结构化失败记录
                    if attempt == retries:
                        result = {
                            "sample_id": sample_id,
                            "relevance_label": None,
                            "processing_status": "failed",
                            "error_code": type(error).__name__,
                        }
                    else:
                        time.sleep(min(0.1 * 2**attempt, 1.0))
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            written += 1
    return written
