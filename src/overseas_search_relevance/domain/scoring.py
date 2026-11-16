"""确定性相关性合成规则。"""

from .schemas import QueryReferRelation, RelevanceLabel


def _validate_score(score: int, name: str) -> None:
    if score not in {label.value for label in RelevanceLabel}:
        raise ValueError(f"{name} 必须是 0～3 的整数")


def compose_post_watch_score(
    relation: QueryReferRelation,
    query_doc_score: int,
    refer_doc_score: int | None = None,
) -> int:
    """执行看后搜公式。

    强相关和不相关分支只使用 Query-Doc；弱相关分支允许 Refer-Doc 提供有限补充，
    但最多贡献 `refer_doc_score - 1`，防止前序视频覆盖用户的显式 Query。
    证据缺失或冲突时必须进入人工复核，不能默认按不相关处理。
    """

    _validate_score(query_doc_score, "query_doc_score")
    if relation is QueryReferRelation.REVIEW:
        raise ValueError("关系待复核时不能生成最终分数")
    if relation is not QueryReferRelation.WEAK:
        return query_doc_score
    if refer_doc_score is None:
        raise ValueError("弱相关分支必须提供 refer_doc_score")
    _validate_score(refer_doc_score, "refer_doc_score")
    return max(query_doc_score, refer_doc_score - 1)
