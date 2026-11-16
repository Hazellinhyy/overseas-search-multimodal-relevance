from overseas_search_relevance.models.adapters import MockAdapter, ModelRequest


def test_mock_adapter_contract() -> None:
    response = MockAdapter().infer(ModelRequest("prompt"))
    assert response.model == "mock-test-only"
    assert "relevance_label" in response.content
