import json
from pathlib import Path

from overseas_search_relevance.inference.batch import cache_key, run_batch


def test_cache_key_changes_with_version() -> None:
    base = dict(payload={"x": 1}, prompt_version="p", rule_version="r", media_config="m")
    assert cache_key(model_version="1", **base) != cache_key(model_version="2", **base)


def test_batch_failure_is_not_zero(tmp_path: Path) -> None:
    output = tmp_path / "result.jsonl"
    run_batch(
        [{"sample_id": "1"}], lambda _row: (_ for _ in ()).throw(RuntimeError()), output, retries=0
    )
    result = json.loads(output.read_text())
    assert result["processing_status"] == "failed"
    assert result["relevance_label"] is None


def test_batch_resume(tmp_path: Path) -> None:
    output = tmp_path / "result.jsonl"

    def processor(row: dict[str, str]) -> dict[str, str]:
        return {"sample_id": row["sample_id"], "processing_status": "success"}

    assert run_batch([{"sample_id": "1"}], processor, output) == 1
    assert run_batch([{"sample_id": "1"}], processor, output) == 0
