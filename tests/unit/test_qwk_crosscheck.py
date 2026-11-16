from sklearn.metrics import cohen_kappa_score

from overseas_search_relevance.evaluation.metrics import quadratic_weighted_kappa


def test_qwk_matches_sklearn() -> None:
    gold = [0, 0, 1, 1, 2, 2, 3, 3, 3]
    predicted = [0, 1, 1, 2, 2, 3, 3, 0, 3]
    expected = cohen_kappa_score(gold, predicted, weights="quadratic", labels=[0, 1, 2, 3])
    actual = quadratic_weighted_kappa(gold, predicted)
    assert actual is not None
    assert abs(actual - expected) < 1e-12
