import pytest

from overseas_search_relevance.domain.schemas import InferenceInput, InferenceOutput
from overseas_search_relevance.prompts.baseline import build_prompt


def test_prompt_preserves_query_and_mode() -> None:
    prompt = build_prompt("ação original", {"title": "conteúdo"}, multimodal=False)
    assert "ação original" in prompt
    assert "纯文本" in prompt
    assert "不得编造" in prompt
    assert "3分：核心实体正确" in prompt
    assert "0分：核心实体、主题或主要意图不一致" in prompt
    assert "只输出一个 JSON 对象" in prompt


def test_input_validation() -> None:
    InferenceInput("1", "query").validate()
    with pytest.raises(ValueError):
        InferenceInput("", "query").validate()
    with pytest.raises(ValueError):
        InferenceInput("1", " ").validate()


def test_output_to_dict() -> None:
    output = InferenceOutput("1", 2, 0.8, None, "B", "reason", ["evidence"])
    assert output.to_dict()["relevance_label"] == 2
