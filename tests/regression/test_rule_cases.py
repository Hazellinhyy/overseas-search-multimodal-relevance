"""官方规则与四档边界的回归案例。"""

from pathlib import Path

import pytest

from overseas_search_relevance.domain.schemas import QueryReferRelation
from overseas_search_relevance.domain.scoring import compose_post_watch_score


@pytest.mark.parametrize(
    ("relation", "query_doc", "refer_doc", "expected"),
    [
        *[(QueryReferRelation.STRONG, q, None, q) for q in range(4)],
        *[(QueryReferRelation.UNRELATED, q, None, q) for q in range(4)],
        *[(QueryReferRelation.WEAK, q, r, max(q, r - 1)) for q in range(4) for r in range(4)],
    ],
)
def test_post_watch_formula_cases(
    relation: QueryReferRelation, query_doc: int, refer_doc: int | None, expected: int
) -> None:
    assert compose_post_watch_score(relation, query_doc, refer_doc) == expected


@pytest.mark.parametrize("score", [-10, -1, 4, 5, 100])
def test_invalid_query_doc_score(score: int) -> None:
    with pytest.raises(ValueError):
        compose_post_watch_score(QueryReferRelation.STRONG, score)


@pytest.mark.parametrize("score", [-10, -1, 4, 5, 100])
def test_invalid_refer_doc_score(score: int) -> None:
    with pytest.raises(ValueError):
        compose_post_watch_score(QueryReferRelation.WEAK, 1, score)


@pytest.mark.parametrize(
    ("rule_id", "business_boundary"),
    [
        ("R001", "核心实体匹配"),
        ("R002", "核心实体不匹配"),
        ("R003", "同名不同实体"),
        ("R004", "多义项"),
        ("R005", "主要义项与小众义项"),
        ("R006", "不可省略限定词缺失"),
        ("R007", "不可省略限定词冲突"),
        ("R008", "可省略限定词"),
        ("R009", "影视客观集数冲突"),
        ("R010", "非客观Part差异"),
        ("R011", "原唱与翻唱"),
        ("R012", "泛时效Query"),
        ("R013", "明确日期Query"),
        ("R014", "问答明确回答"),
        ("R015", "仅提供推导方法"),
        ("R016", "有问无答"),
        ("R017", "仅关键词命中"),
        ("R018", "内容主体不匹配"),
        ("R019", "视频相关内容占比不足"),
        ("R020", "音画信息冲突"),
        ("R021", "外露信息不充分"),
        ("R022", "Query拼写纠错"),
        ("R023", "错误拼写导致实体转义"),
        ("R024", "无意义Query"),
        ("R025", "视频不可播放"),
        ("R026", "看后搜强相关"),
        ("R027", "看后搜弱相关"),
        ("R028", "看后搜不相关"),
        ("R029", "前序视频缺失"),
        ("R030", "前序与候选证据冲突"),
    ],
)
def test_required_rule_is_versioned(rule_id: str, business_boundary: str) -> None:
    catalog = Path("docs/rules/rule_catalog.yaml").read_text(encoding="utf-8")
    assert f"id: {rule_id}" in catalog
    assert business_boundary in catalog
    assert "version: 1.0.0" in catalog


def test_review_relation_cannot_score() -> None:
    with pytest.raises(ValueError, match="待复核"):
        compose_post_watch_score(QueryReferRelation.REVIEW, 2)


def test_weak_relation_requires_refer_score() -> None:
    with pytest.raises(ValueError, match="必须提供"):
        compose_post_watch_score(QueryReferRelation.WEAK, 2)
