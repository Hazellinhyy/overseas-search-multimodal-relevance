import json
import time
from zipfile import ZipFile

from overseas_search_relevance.inference.jobs import BatchJobManager, load_rows, normalize_row
from overseas_search_relevance.models.adapters import ModelResponse


class FakeAdapter:
    def infer(self, request):
        return ModelResponse(
            '{"relevance_label":1,"confidence":0.8,"reason":"ok","evidence":["text"]}',
            "fake",
            {},
        )


def test_load_csv_json_and_normalize(tmp_path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("item_id,query,doc_title,gold_label\n7,praia,video,3\n")
    rows = load_rows(csv_path)
    normalized = normalize_row(rows[0], {})
    assert normalized["sample_id"] == "7"
    assert normalized["title"] == "video"
    assert normalized["gold_label"] == "3"
    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps([{"query": "x"}]))
    assert load_rows(json_path)[0]["query"] == "x"


def test_normalize_excludes_non_numeric_gold():
    normalized = normalize_row(
        {"item_id": "8", "query": "x", "gold_label": "丢弃", "gold_status": "discarded"},
        {},
    )
    assert normalized["gold_label"] == ""
    assert normalized["gold_status"] == "discarded"


def test_batch_job_zip(tmp_path):
    bundle = tmp_path / "batch.zip"
    with ZipFile(bundle, "w") as archive:
        archive.writestr("input.jsonl", '{"sample_id":"1","query":"praia"}\n')
    manager = BatchJobManager(tmp_path / "jobs", FakeAdapter())
    job_id = manager.submit(bundle)
    deadline = time.monotonic() + 3
    while manager.status(job_id)["status"] not in {"completed", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert manager.status(job_id)["status"] == "completed"
    assert manager.result_path(job_id).is_file()
