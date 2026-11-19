import json

import pytest

from overseas_search_relevance.models.adapters import (
    EnsembleAdapter,
    LocalVLLMVisionAdapter,
    MockAdapter,
    ModelRequest,
)


def test_mock_adapter_contract() -> None:
    response = MockAdapter().infer(ModelRequest("prompt"))
    assert response.model == "mock-test-only"
    assert "relevance_label" in response.content


def test_local_vllm_rejects_non_tunnel_endpoint() -> None:
    with pytest.raises(ValueError, match="SSH 隧道"):
        LocalVLLMVisionAdapter("https://example.com/v1/chat/completions", "model")


def test_local_vllm_encodes_image(tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    encoded = LocalVLLMVisionAdapter._image_url(str(image))
    assert encoded.startswith("data:image/jpeg;base64,")


def test_local_vllm_inference_protocol(tmp_path, monkeypatch) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "model": "vlm",
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 2},
                }
            ).encode()

    def urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    response = LocalVLLMVisionAdapter("http://127.0.0.1:8000/v1/chat/completions", "vlm").infer(
        ModelRequest("prompt", (str(image),))
    )
    assert response.model == "vlm"
    assert captured["payload"]["messages"][0]["content"][1]["type"] == "image_url"


def test_ensemble_exposes_agreement_and_entropy() -> None:
    first = MockAdapter('{"relevance_label":2,"confidence":0.9,"reason":"a","evidence":[]}')
    second = MockAdapter('{"relevance_label":2,"confidence":0.8,"reason":"b","evidence":[]}')
    third = MockAdapter('{"relevance_label":1,"confidence":0.7,"reason":"c","evidence":[]}')
    result = json.loads(EnsembleAdapter([first, second, third]).infer(ModelRequest("x")).content)
    assert result["relevance_label"] == 2
    assert result["agreement"] == pytest.approx(2 / 3)
    assert result["entropy"] > 0
