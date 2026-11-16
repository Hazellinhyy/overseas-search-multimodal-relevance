"""0～3 有序标签评测；空集和退化样本不会伪造有效 QWK。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    count: int
    qwk: float | None
    exact_accuracy: float | None
    adjacent_accuracy: float | None
    confusion_matrix: list[list[int]]
    per_label: dict[int, dict[str, float | None]]
    failure_rate: float | None = None
    retry_rate: float | None = None
    latency_ms: dict[str, float | None] | None = None


def _validate(gold: list[int], predictions: list[int]) -> None:
    if len(gold) != len(predictions):
        raise ValueError("gold 与 predictions 长度必须一致")
    if any(label not in range(4) for label in [*gold, *predictions]):
        raise ValueError("标签必须为 0～3")


def quadratic_weighted_kappa(gold: list[int], predictions: list[int]) -> float | None:
    """计算二次加权 Kappa；期望分歧为零时返回 None。"""

    _validate(gold, predictions)
    n = len(gold)
    if n == 0:
        return None
    observed = [[0] * 4 for _ in range(4)]
    for actual, predicted in zip(gold, predictions, strict=True):
        observed[actual][predicted] += 1
    actual_hist = Counter(gold)
    predicted_hist = Counter(predictions)
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(4):
        for j in range(4):
            weight = ((i - j) / 3) ** 2
            observed_disagreement += weight * observed[i][j] / n
            expected_disagreement += weight * actual_hist[i] * predicted_hist[j] / (n * n)
    if expected_disagreement == 0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(percentile * len(ordered)) - 1)]


def evaluate(
    gold: list[int],
    predictions: list[int],
    *,
    statuses: list[str] | None = None,
    retries: list[int] | None = None,
    latencies_ms: list[float] | None = None,
) -> EvaluationMetrics:
    _validate(gold, predictions)
    count = len(gold)
    matrix = [[0] * 4 for _ in range(4)]
    for actual, predicted in zip(gold, predictions, strict=True):
        matrix[actual][predicted] += 1
    per_label: dict[int, dict[str, float | None]] = {}
    for label in range(4):
        tp = matrix[label][label]
        predicted_total = sum(row[label] for row in matrix)
        actual_total = sum(matrix[label])
        per_label[label] = {
            "precision": tp / predicted_total if predicted_total else None,
            "recall": tp / actual_total if actual_total else None,
        }
    failures = sum(status != "success" for status in statuses or [])
    retry_count = sum(value > 0 for value in retries or [])
    latency = list(latencies_ms or [])
    return EvaluationMetrics(
        count=count,
        qwk=quadratic_weighted_kappa(gold, predictions),
        exact_accuracy=sum(a == b for a, b in zip(gold, predictions, strict=True)) / count
        if count
        else None,
        adjacent_accuracy=sum(abs(a - b) <= 1 for a, b in zip(gold, predictions, strict=True))
        / count
        if count
        else None,
        confusion_matrix=matrix,
        per_label=per_label,
        failure_rate=failures / len(statuses) if statuses else None,
        retry_rate=retry_count / len(retries) if retries else None,
        latency_ms={
            "p50": _percentile(latency, 0.50),
            "p95": _percentile(latency, 0.95),
            "p99": _percentile(latency, 0.99),
        },
    )
