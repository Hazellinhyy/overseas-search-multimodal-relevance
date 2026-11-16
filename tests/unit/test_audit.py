import json
from pathlib import Path

import pytest

from overseas_search_relevance.data import audit


def test_load_and_audit_dataset(tmp_path: Path) -> None:
    source = tmp_path / "model.json"
    rows = [
        {"query": "ação", "item_id": 1, "doc_title": "t", "doc_asr": "fala"},
        {"query": "ascii query", "item_id": 1, "doc_title": "", "doc_ocr_all": "texto"},
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    loaded = audit.load_model_input(source)
    profile = audit.audit_dataset(loaded)
    assert profile["sample_count"] == 2
    assert profile["duplicate_item_ids"] == 1
    assert profile["language_distribution"]["pt"] == 1
    assert profile["video"]["status"] == "NOT_EVALUATED"


def test_invalid_model_input(tmp_path: Path) -> None:
    source = tmp_path / "model.json"
    source.write_text('{"not": "a list"}')
    with pytest.raises(ValueError):
        audit.load_model_input(source)


def test_video_audit_with_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "1.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        audit,
        "_probe_video",
        lambda _path: {"ok": True, "format": {"duration": "2.5", "format_name": "mp4"}},
    )
    profile = audit.audit_dataset([{"query": "q", "item_id": 1}], tmp_path)
    assert profile["video"]["decode_success_rate"] == 1.0
    assert profile["video"]["duration_seconds"]["average"] == 2.5


def test_write_profile_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "profile.json"
    audit.write_profile({"ok": True}, output)
    assert json.loads(output.read_text()) == {"ok": True}
