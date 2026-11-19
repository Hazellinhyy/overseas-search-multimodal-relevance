"""本地 Demo 与批处理共享的多模态推理服务。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from overseas_search_relevance.calibration.tiers import ConfidenceSignals, assign_tier
from overseas_search_relevance.media.fetch import fetch_video
from overseas_search_relevance.media.video import FrameConfig, extract_uniform_frames
from overseas_search_relevance.models.adapters import ModelAdapter, ModelRequest
from overseas_search_relevance.prompts.baseline import PROMPT_VERSION, build_prompt

MODEL_METADATA_FIELDS = (
    "title",
    "asr",
    "ocr",
    "video_url",
    "refer_video_info",
    "doc_title",
    "doc_caption",
    "doc_author",
    "doc_kwaiid",
    "doc_asr",
    "doc_ocr",
    "doc_ocr_all",
    "doc_hashtag",
    "doc_mllm_queries",
    "doc_mllm_query1",
    "doc_mllm_query2",
    "doc_mllm_query3",
    "doc_mllm_query4",
    "doc_mllm_query5",
    "doc_hetu_tag_id1",
    "doc_text_source",
    "create_time",
    "blobstore_key",
    "url_path",
)


def _json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        if text.startswith("json"):
            text = text[4:].lstrip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("模型输出中没有可解析的 JSON 对象")


def _confidence_value(value: Any) -> float:
    if isinstance(value, str):
        qualitative = {"high": 0.85, "medium": 0.65, "low": 0.4}
        normalized = value.strip().lower()
        if normalized in qualitative:
            return qualitative[normalized]
    return float(value)


def infer_sample(
    payload: dict[str, Any], adapter: ModelAdapter, *, frame_dir: Path
) -> dict[str, Any]:
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("query 不能为空")
    metadata = {
        key: str(payload[key]) for key in MODEL_METADATA_FIELDS if payload.get(key)
    }
    images = [str(Path(item)) for item in payload.get("images", [])]
    video_path = str(payload.get("video_path", "")).strip()
    video_url = str(payload.get("video_url", "")).strip()
    if not video_path and video_url and payload.get("video_fetch_enabled", True):
        video_path = str(fetch_video(video_url, frame_dir.parent / "downloads"))
    if video_path:
        video = Path(video_path).resolve()
        if not video.is_file():
            raise ValueError("本地视频文件不存在")
        images.extend(
            str(path)
            for path in extract_uniform_frames(video, frame_dir, FrameConfig(max_frames=8))
        )
    started = time.perf_counter()
    request = ModelRequest(build_prompt(query, metadata, multimodal=bool(images)), tuple(images))
    response = adapter.infer(request)
    try:
        parsed = _json_object(response.content)
    except ValueError:
        raw_output = response.content
        parsed = None
        for _ in range(2):
            repair_prompt = f"""你是 JSON 格式转换器。把原输出改写为一个 JSON 对象。
只允许输出从 {{ 开始、以 }} 结束的 JSON，禁止 markdown 和解释。
字段必须是 relevance_label(0到3整数)、confidence(0到1数字)、reason(字符串)、evidence(字符串数组)。
保持原判断，不添加新事实。原输出：
{raw_output}
"""
            response = adapter.infer(ModelRequest(repair_prompt))
            raw_output = response.content
            try:
                parsed = _json_object(raw_output)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError("模型输出经两次格式修复后仍无可解析 JSON") from None
    assert parsed is not None
    label = int(parsed["relevance_label"])
    confidence = _confidence_value(parsed["confidence"])
    if label not in range(4):
        raise ValueError("relevance_label 必须是 0～3")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence 必须在 0～1")
    evidence = parsed.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        raise ValueError("evidence 必须是数组")
    completeness = 1.0 if images and evidence else 0.7 if evidence else 0.4
    agreement = float(parsed.get("agreement", min(confidence, 0.7)))
    entropy = float(parsed.get("entropy", 0.0))
    tier = assign_tier(ConfidenceSignals(confidence, agreement, completeness)).value
    return {
        "sample_id": str(payload.get("sample_id") or payload.get("item_id") or "demo"),
        "relevance_label": label,
        "confidence": confidence,
        "confidence_calibrated": confidence,
        "tier": tier,
        "agreement": agreement,
        "entropy": entropy,
        "teacher_count": int(parsed.get("teacher_count", 1)),
        "label_distribution": parsed.get("label_distribution", {str(label): 1}),
        "reason": str(parsed.get("reason", "")),
        "evidence": [str(item) for item in evidence],
        "model_name": response.model,
        "prompt_version": PROMPT_VERSION,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "processing_status": "success",
        "frame_count": len(images),
    }
