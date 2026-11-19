"""批量文件解析、异步任务状态与结果持久化。"""

from __future__ import annotations

import csv
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from overseas_search_relevance.data.safe_zip import safe_extract
from overseas_search_relevance.data.xlsx import iter_xlsx_sheets
from overseas_search_relevance.evaluation.metrics import evaluate
from overseas_search_relevance.inference.batch import run_batch
from overseas_search_relevance.inference.service import infer_sample
from overseas_search_relevance.models.adapters import ModelAdapter

DATA_SUFFIXES = {".json", ".jsonl", ".csv", ".xlsx"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError("JSON 必须是对象数组")
        return value
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if suffix == ".xlsx":
        sheets = list(iter_xlsx_sheets(path))
        if not sheets or len(sheets[0][1]) < 2:
            raise ValueError("XLSX 没有可用数据行")
        headers, *values = sheets[0][1]
        return [dict(zip(headers, row, strict=False)) for row in values]
    raise ValueError("仅支持 JSON、JSONL、CSV、XLSX 或 ZIP")


def normalize_row(row: dict[str, Any], video_index: dict[str, Path]) -> dict[str, Any]:
    def first(*names: str) -> Any:
        return next((row[name] for name in names if row.get(name) not in (None, "")), "")

    sample_id = str(first("sample_id", "item_id", "video_id", "id"))
    video = video_index.get(sample_id)
    raw_gold = first("gold_label", "label", "relevance_gold")
    gold_label = raw_gold if str(raw_gold).strip() in {"0", "1", "2", "3"} else ""
    normalized = {
        "sample_id": sample_id or str(uuid.uuid4()),
        "query": str(first("query", "search_query")),
        "title": str(first("title", "doc_title", "caption", "doc_caption")),
        "asr": str(first("asr", "doc_asr")),
        "ocr": str(first("ocr", "doc_ocr", "doc_ocr_all")),
        "video_url": str(first("video_url", "url")),
        "refer_video_info": str(first("refer_video_info", "refer_info")),
        "video_path": str(video) if video else str(first("video_path")),
        "gold_label": gold_label,
        "gold_status": "valid" if gold_label != "" else str(first("gold_status")),
        "media_status": str(first("media_status")),
        "visual_strong": str(first("visual_strong", "is_visual_query")).lower()
        in {"1", "true", "yes"},
    }
    passthrough = {
        key: value
        for key, value in row.items()
        if key.startswith("doc_")
        or key in {"blobstore_key", "url_path", "create_time"}
    }
    normalized.update(passthrough)
    return normalized


class BatchJobManager:
    def __init__(self, root: Path, adapter: ModelAdapter) -> None:
        self.root = root
        self.adapter = adapter
        self._lock = threading.Lock()
        self._status: dict[str, dict[str, Any]] = {}

    def submit(self, upload: Path) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._status[job_id] = {"job_id": job_id, "status": "queued", "processed": 0}
        threading.Thread(target=self._run, args=(job_id, upload), daemon=True).start()
        return job_id

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._status.get(job_id)
            return dict(value) if value else None

    def result_path(self, job_id: str) -> Path:
        return self.root / job_id / "results.jsonl"

    def _run(self, job_id: str, upload: Path) -> None:
        started = time.perf_counter()
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            files = (
                safe_extract(upload, job_dir / "input")
                if upload.suffix.lower() == ".zip"
                else [upload]
            )
            data_files = [path for path in files if path.suffix.lower() in DATA_SUFFIXES]
            if len(data_files) != 1:
                raise ValueError("批次必须包含且仅包含一个 JSON/JSONL/CSV/XLSX 数据文件")
            videos = {path.stem: path for path in files if path.suffix.lower() in VIDEO_SUFFIXES}
            rows = [normalize_row(row, videos) for row in load_rows(data_files[0])]
            with self._lock:
                self._status[job_id].update(status="running", total=len(rows))

            def processor(row: dict[str, Any]) -> dict[str, Any]:
                result = infer_sample(
                    row, self.adapter, frame_dir=job_dir / "frames" / row["sample_id"]
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
                with self._lock:
                    self._status[job_id]["processed"] += 1
                return result

            run_batch(rows, processor, self.result_path(job_id), retries=2)
            results = [
                json.loads(line)
                for line in self.result_path(job_id).read_text(encoding="utf-8").splitlines()
                if line
            ]
            self._write_csv(job_dir / "results.csv", results)
            summary = self._summary(results, time.perf_counter() - started)
            (job_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            with self._lock:
                self._status[job_id]["status"] = "completed"
                self._status[job_id]["result"] = f"/api/jobs/{job_id}/result"
                self._status[job_id]["csv_result"] = f"/api/jobs/{job_id}/result.csv"
                self._status[job_id]["summary"] = summary
        except Exception as error:
            with self._lock:
                self._status[job_id].update(
                    status="failed", error=type(error).__name__, message=str(error)
                )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in row.items()
                    }
                )

    @staticmethod
    def _summary(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        successful = [row for row in rows if row.get("processing_status") == "success"]
        summary: dict[str, Any] = {
            "total": len(rows),
            "success": len(successful),
            "failure_rate": 1 - len(successful) / len(rows) if rows else 0,
            "elapsed_seconds": round(elapsed, 3),
            "throughput_items_per_second": round(len(rows) / elapsed, 3) if elapsed else None,
            "average_latency_ms": round(
                sum(float(row.get("latency_ms", 0)) for row in successful) / len(successful), 2
            )
            if successful
            else None,
            "tier_counts": {
                tier: sum(row.get("tier") == tier for row in successful) for tier in "ABC"
            },
        }
        evaluated = [row for row in successful if row.get("gold_label") not in (None, "")]
        if evaluated:
            metrics = evaluate(
                [int(row["gold_label"]) for row in evaluated],
                [int(row["relevance_label"]) for row in evaluated],
            )
            summary["evaluation"] = {
                "count": metrics.count,
                "qwk": metrics.qwk,
                "exact_accuracy": metrics.exact_accuracy,
                "adjacent_accuracy": metrics.adjacent_accuracy,
            }
        return summary
