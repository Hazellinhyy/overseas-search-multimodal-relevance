import pytest

from overseas_search_relevance.data.split import grouped_split
from overseas_search_relevance.evaluation.badcases import candidate_badcases


def test_grouped_split_prevents_duplicate_leakage() -> None:
    rows = [
        {"query": "Same Query", "item_id": "1", "row": 1},
        {"query": " same  query ", "item_id": "1", "row": 2},
        {"query": "other", "item_id": "2", "row": 3},
    ]
    split = grouped_split(rows)
    locations = [
        name for name, values in split.items() if sum(row["row"] in {1, 2} for row in values)
    ]
    assert len(locations) == 1
    assert sum(len(values) for values in split.values()) == 3


def test_invalid_split_ratio() -> None:
    with pytest.raises(ValueError):
        grouped_split([], (50, 20, 20))


def test_badcase_candidates() -> None:
    rows = [
        {"sample_id": "1", "gold_label": 2, "relevance_label": 1},
        {"sample_id": "2", "gold_label": 3, "relevance_label": 3},
        {"sample_id": "3", "processing_status": "failed"},
    ]
    result = candidate_badcases(rows)
    assert [row["sample_id"] for row in result] == ["1", "3"]
