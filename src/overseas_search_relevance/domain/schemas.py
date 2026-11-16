"""四档相关性、输入输出协议与业务枚举。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class RelevanceLabel(IntEnum):
    """官方四档标签：0 完全不匹配，3 完全匹配。"""

    IRRELEVANT = 0
    WEAK = 1
    PARTIAL = 2
    FULL = 3


class Scene(StrEnum):
    GENERAL_SEARCH = "general_search"
    POST_WATCH_SEARCH = "post_watch_search"
    LIVE_SEARCH = "live_search"


class QueryReferRelation(StrEnum):
    STRONG = "strong"
    WEAK = "weak"
    UNRELATED = "unrelated"
    REVIEW = "review"


class ProcessingStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    INVALID = "invalid"
    REVIEW = "review"


class ConfidenceTier(StrEnum):
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True, slots=True)
class InferenceInput:
    sample_id: str
    query: str
    video_url: str | None = None
    scene: Scene = Scene.GENERAL_SEARCH
    refer_video_info: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id 不能为空")
        if not self.query.strip():
            raise ValueError("query 不能为空")


@dataclass(slots=True)
class InferenceOutput:
    sample_id: str
    relevance_label: int | None
    confidence: float | None
    confidence_calibrated: float | None
    tier: str | None
    reason: str
    evidence: list[str]
    query_refer_relation: str | None = None
    query_doc_score: int | None = None
    refer_doc_score: int | None = None
    rule_applied: list[str] = field(default_factory=list)
    rule_version: str = "1.0.0"
    model_name: str = ""
    model_version: str = ""
    prompt_version: str = "1.0.0"
    latency_ms: float = 0.0
    estimated_cost: float = 0.0
    processing_status: str = ProcessingStatus.SUCCESS.value
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
