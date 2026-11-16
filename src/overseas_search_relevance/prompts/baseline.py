"""文本与多模态基线的共享规则骨架。"""

PROMPT_VERSION = "1.0.0"


def build_prompt(query: str, metadata: dict[str, str], *, multimodal: bool) -> str:
    """构建可比 Prompt；多模态证据必须由调用方显式提供，不能凭标题臆测画面。"""

    fields = "\n".join(f"{key}: {value}" for key, value in sorted(metadata.items()) if value)
    mode = "多模态（另附已抽取关键帧）" if multimodal else "纯文本"
    return f"""你是海外搜索相关性评审。模式：{mode}
按 0/1/2/3 四档输出 JSON。核心实体和不可省略限定词优先，不得只按词汇重合评分。
Query 原文：{query}
候选文本：
{fields}
输出字段：relevance_label、confidence、reason、evidence。不得编造未提供的视频内容。
"""
