import pytest

from overseas_search_relevance.calibration.tiers import ConfidenceSignals, assign_tier
from overseas_search_relevance.domain.schemas import ConfidenceTier


def test_high_confidence_tier() -> None:
    assert assign_tier(ConfidenceSignals(0.9, 0.9, 0.9)) is ConfidenceTier.A


def test_conflict_forces_review() -> None:
    assert (
        assign_tier(ConfidenceSignals(0.99, 0.99, 0.99, has_modal_conflict=True))
        is ConfidenceTier.C
    )


def test_invalid_signal() -> None:
    with pytest.raises(ValueError):
        assign_tier(ConfidenceSignals(1.1, 0.9, 0.9))
