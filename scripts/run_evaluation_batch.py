#!/usr/bin/env python3
"""Run a resumable local/KML evaluation batch without exposing Gold to the prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from overseas_search_relevance.inference.batch import run_batch
from overseas_search_relevance.inference.jobs import load_rows, normalize_row
from overseas_search_relevance.inference.service import infer_sample
from overseas_search_relevance.models.adapters import LocalVLLMVisionAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Optional directory containing <item_id>.mp4/.mov/.mkv/.webm files",
    )
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8001/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen3-VL-32B-Instruct")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--allow-video-download",
        action="store_true",
        help="Allow downloading video_url; disabled by default for private evaluation data",
    )
    parser.add_argument(
        "--require-video",
        action="store_true",
        help="Fail before inference when an item_id has no local video",
    )
    args = parser.parse_args()

    adapter = LocalVLLMVisionAdapter(args.endpoint, args.model, timeout=args.timeout)
    video_index: dict[str, Path] = {}
    if args.video_dir:
        for suffix in ("*.mp4", "*.mov", "*.mkv", "*.webm"):
            video_index.update({path.stem: path for path in args.video_dir.glob(suffix)})
    rows = [normalize_row(row, video_index) for row in load_rows(args.input)]
    for row in rows:
        row["video_fetch_enabled"] = args.allow_video_download
        if args.require_video and not row.get("video_path"):
            raise SystemExit(f"missing local video for item_id={row['sample_id']}")

    def processor(row: dict[str, Any]) -> dict[str, Any]:
        result = infer_sample(
            row,
            adapter,
            frame_dir=args.frame_root / str(row["sample_id"]),
        )
        result.update(
            query=row.get("query", ""),
            title=row.get("title", ""),
            video_url=row.get("video_url", ""),
            media_status=row.get("media_status", ""),
            gold_status=row.get("gold_status", ""),
        )
        if row.get("gold_label") not in (None, ""):
            result["gold_label"] = int(row["gold_label"])
        result["visual_strong"] = bool(row.get("visual_strong"))
        return result

    written = run_batch(rows, processor, args.output, retries=2)
    print(json.dumps({"written": written, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
