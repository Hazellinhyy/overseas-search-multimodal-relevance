"""基于多信号的初始 A/B/C 分层策略。"""

from dataclasses import dataclass

from overseas_search_relevance.domain.schemas import ConfidenceTier


@dataclass(frozen=True, slots=True)
class ConfidenceSignals:
    model_confidence: float
    agreement: float
    evidence_completeness: float
    has_rule_conflict: bool = False
    has_modal_conflict: bool = False


def assign_tier(signals: ConfidenceSignals) -> ConfidenceTier:
    """分配初始层级；模型自报值只是信号，不能解释为真实正确率。"""

    for value in (signals.model_confidence, signals.agreement, signals.evidence_completeness):
        if not 0.0 <= value <= 1.0:
            raise ValueError("置信度信号必须在 0～1")
    if signals.has_rule_conflict or signals.has_modal_conflict:
        return ConfidenceTier.C
    conservative = min(signals.model_confidence, signals.agreement, signals.evidence_completeness)
    if conservative >= 0.85:
        return ConfidenceTier.A
    if conservative >= 0.60:
        return ConfidenceTier.B
    return ConfidenceTier.C
