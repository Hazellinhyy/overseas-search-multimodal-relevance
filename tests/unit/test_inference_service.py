import json

import pytest

from overseas_search_relevance.inference.service import (
    _confidence_value,
    _json_object,
    infer_sample,
)
from overseas_search_relevance.models.adapters import ModelResponse


class FakeAdapter:
    def infer(self, request):
        return ModelResponse(
            json.dumps(
                {
                    "relevance_label": 3,
                    "confidence": 0.91,
                    "reason": "画面与查询一致",
                    "evidence": ["视觉证据"],
                }
            ),
            "fake-vlm",
            {},
        )


def test_infer_sample_without_gold(tmp_path) -> None:
    result = infer_sample({"query": "praia", "title": "praia"}, FakeAdapter(), frame_dir=tmp_path)
    assert result["relevance_label"] == 3
    assert result["tier"] == "B"
    assert "gold" not in json.dumps(result).lower()


def test_json_fence_and_invalid_label(tmp_path) -> None:
    assert _json_object('```json\n{"relevance_label": 1}\n```')["relevance_label"] == 1
    assert _json_object('分析如下： {"relevance_label": 2} 完成。')["relevance_label"] == 2

    class InvalidAdapter(FakeAdapter):
        def infer(self, request):
            return ModelResponse('{"relevance_label":9,"confidence":0.8}', "fake", {})

    with pytest.raises(ValueError, match="0～3"):
        infer_sample({"query": "x"}, InvalidAdapter(), frame_dir=tmp_path)


def test_infer_sample_normalizes_string_evidence(tmp_path) -> None:
    class StringEvidenceAdapter(FakeAdapter):
        def infer(self, request):
            return ModelResponse(
                '{"relevance_label":1,"confidence":0.8,"evidence":"title match"}',
                "fake",
                {},
            )

    result = infer_sample({"query": "x"}, StringEvidenceAdapter(), frame_dir=tmp_path)
    assert result["evidence"] == ["title match"]


def test_confidence_value_maps_qualitative_levels() -> None:
    assert _confidence_value("high") == 0.85
    assert _confidence_value("medium") == 0.65
    assert _confidence_value("low") == 0.4
    assert _confidence_value("0.75") == 0.75


def test_infer_sample_repairs_non_json_and_passes_full_metadata(tmp_path) -> None:
    class RepairAdapter(FakeAdapter):
        def __init__(self):
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse("label is two", "fake", {})
            return ModelResponse(
                '{"relevance_label":2,"confidence":0.7,"reason":"ok","evidence":[]}',
                "fake",
                {},
            )

    adapter = RepairAdapter()
    result = infer_sample(
        {
            "item_id": "42",
            "query": "praia",
            "doc_author": "author",
            "blobstore_key": "blob-key",
        },
        adapter,
        frame_dir=tmp_path,
    )
    assert result["sample_id"] == "42"
    assert result["relevance_label"] == 2
    assert "doc_author: author" in adapter.requests[0].prompt
    assert "blobstore_key: blob-key" in adapter.requests[0].prompt
    assert len(adapter.requests) == 2


def test_infer_sample_allows_second_json_repair(tmp_path) -> None:
    class SecondRepairAdapter(FakeAdapter):
        def __init__(self):
            self.calls = 0

        def infer(self, request):
            self.calls += 1
            if self.calls < 3:
                return ModelResponse("not json", "fake", {})
            return ModelResponse(
                '{"relevance_label":1,"confidence":0.6,"reason":"ok","evidence":[]}',
                "fake",
                {},
            )

    adapter = SecondRepairAdapter()
    result = infer_sample({"query": "x"}, adapter, frame_dir=tmp_path)
    assert result["relevance_label"] == 1
    assert adapter.calls == 3


def test_infer_sample_can_disable_video_download(tmp_path) -> None:
    result = infer_sample(
        {
            "item_id": "42",
            "query": "praia",
            "video_url": "https://internal.invalid/video",
            "video_fetch_enabled": False,
        },
        FakeAdapter(),
        frame_dir=tmp_path,
    )
    assert result["frame_count"] == 0
