"""真实数据集字段、标签、模态与视频覆盖审计。"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

TEXT_FIELDS = ("doc_title", "doc_caption", "doc_asr", "doc_ocr", "doc_ocr_all")


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _language(query: str) -> str:
    lowered = query.casefold()
    portuguese = set("áàâãéêíóôõúç")
    if any(char in lowered for char in portuguese):
        return "pt"
    ascii_letters = sum(char.isascii() and char.isalpha() for char in query)
    if ascii_letters >= max(1, len(query) // 3):
        return "pt_or_latin_unknown"
    return "unknown"


def load_model_input(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("model_input 必须是对象数组")
    return data


def _probe_video(path: Path, timeout: float = 15.0) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name:stream=codec_type,codec_name,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": type(error).__name__}
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip()[:300]}
    payload: dict[str, Any] = json.loads(result.stdout)
    payload["ok"] = True
    return payload


def audit_dataset(rows: list[dict[str, Any]], video_dir: Path | None = None) -> dict[str, Any]:
    """按真实样本生成可复现画像；解码失败与缺失视频分开统计。"""

    item_ids = [str(row.get("item_id", "")) for row in rows]
    duplicates = sum(count - 1 for count in Counter(item_ids).values() if count > 1)
    profile: dict[str, Any] = {
        "sample_count": len(rows),
        "language_distribution": dict(
            Counter(_language(str(row.get("query", ""))) for row in rows)
        ),
        "scene_distribution": {"general_search": len(rows), "post_watch_search": 0},
        "duplicate_item_ids": duplicates,
        "missing_fields": {
            field: sum(not _present(row.get(field)) for row in rows)
            for field in ("query", "item_id", *TEXT_FIELDS)
        },
        "modality_coverage": {
            field: sum(_present(row.get(field)) for row in rows) / len(rows) if rows else 0.0
            for field in TEXT_FIELDS
        },
        "refer_video_coverage": 0.0,
    }
    if video_dir is None:
        profile["video"] = {"status": "NOT_EVALUATED"}
        return profile
    probes: list[dict[str, Any]] = []
    missing = 0
    for item_id in item_ids:
        path = video_dir / f"{item_id}.mp4"
        if not path.exists():
            missing += 1
            continue
        probes.append(_probe_video(path))
    successful = [probe for probe in probes if probe.get("ok")]
    durations = [
        float(probe["format"]["duration"])
        for probe in successful
        if probe.get("format", {}).get("duration")
    ]
    formats = Counter(
        str(probe.get("format", {}).get("format_name", "unknown")) for probe in successful
    )
    profile["video"] = {
        "expected": len(item_ids),
        "found": len(probes),
        "missing": missing,
        "decode_success": len(successful),
        "decode_success_rate": len(successful) / len(probes) if probes else 0.0,
        "format_distribution": dict(formats),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "average": sum(durations) / len(durations) if durations else None,
        },
        "errors": [probe.get("error") for probe in probes if not probe.get("ok")][:20],
    }
    return profile


def write_profile(profile: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
