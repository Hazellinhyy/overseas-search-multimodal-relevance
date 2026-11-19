"""文本与多模态基线的共享规则骨架。"""

PROMPT_VERSION = "1.1.0"


def build_prompt(query: str, metadata: dict[str, str], *, multimodal: bool) -> str:
    """构建可比 Prompt；多模态证据必须由调用方显式提供，不能凭标题臆测画面。"""

    fields = "\n".join(f"{key}: {value}" for key, value in sorted(metadata.items()) if value)
    mode = "多模态（另附已抽取关键帧）" if multimodal else "纯文本"
    return f"""你是海外搜索相关性评审。模式：{mode}
严格使用以下官方四档定义：
- 3分：核心实体正确，主要需求和关键/不可省略限定词全部满足，候选主体高度一致。
- 2分：核心实体正确，主要需求基本满足，但只有非关键缺失或合理不确定性。
- 1分：内容存在关联，但关键需求未满足，或只覆盖相邻事件、版本、集数或次要内容。
- 0分：核心实体、主题或主要意图不一致。
判断顺序：核心实体 → 主要意图 → 不可省略限定词 → 内容主体 → 信息覆盖 → 模态冲突。
词汇重合只能作为证据，不能单独决定分数；不要把“证据不足但可能相关”自动升为2分。
Query 原文：{query}
候选文本：
{fields}
只输出一个 JSON 对象，禁止 Markdown 或额外解释。字段：relevance_label(0到3整数)、
confidence(0到1数字)、reason(字符串)、evidence(字符串数组)。不得编造未提供的视频内容。
"""
