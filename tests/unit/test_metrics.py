from overseas_search_relevance.evaluation.metrics import evaluate, quadratic_weighted_kappa


def test_perfect_qwk() -> None:
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_empty_qwk_is_none() -> None:
    assert quadratic_weighted_kappa([], []) is None


def test_single_class_qwk_is_none() -> None:
    assert quadratic_weighted_kappa([1, 1], [1, 1]) is None


def test_accuracy_and_latency() -> None:
    result = evaluate([0, 1, 2, 3], [0, 2, 0, 3], latencies_ms=[10, 20, 30, 40])
    assert result.exact_accuracy == 0.5
    assert result.adjacent_accuracy == 0.75
    assert result.latency_ms == {"p50": 20, "p95": 40, "p99": 40}
