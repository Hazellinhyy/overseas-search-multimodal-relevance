"""统一模型协议、测试适配器和受控 OpenAI-compatible 适配器。"""

from __future__ import annotations

import base64
import ipaddress
import json
import math
import mimetypes
import socket
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class ModelRequest:
    prompt: str
    images: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    model: str
    usage: dict[str, int]


class ModelAdapter(Protocol):
    def infer(self, request: ModelRequest) -> ModelResponse: ...


class MockAdapter:
    """固定返回的测试替身，禁止写入真实效果报告。"""

    def __init__(self, content: str = '{"relevance_label": 0, "confidence": 0.5}') -> None:
        self.content = content

    def infer(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            self.content, "mock-test-only", {"input_tokens": 0, "output_tokens": 0}
        )


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("模型端点必须是有效 HTTPS URL")
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        address = ipaddress.ip_address(result[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
        ):
            raise ValueError("模型端点不能解析到本地或私网地址")


class OpenAICompatibleAdapter:
    def __init__(self, endpoint: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        _validate_endpoint(endpoint)
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def infer(self, request: ModelRequest) -> ModelResponse:
        if request.images:
            raise ValueError("此基础适配器尚未启用图像传输，避免误传内部数据")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": 0,
        }
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:  # noqa: S310
            result: dict[str, Any] = json.load(response)
        content = str(result["choices"][0]["message"]["content"])
        usage = {
            key: int(value)
            for key, value in result.get("usage", {}).items()
            if isinstance(value, int)
        }
        return ModelResponse(content, str(result.get("model", self.model)), usage)


class LocalVLLMVisionAdapter:
    """通过本机 SSH 隧道访问 KML vLLM；拒绝直接向公网发送图片。"""

    def __init__(self, endpoint: str, model: str, timeout: float = 180.0) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("KML 模型端点必须是本机 SSH 隧道地址")
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _image_url(path: str) -> str:
        image = Path(path).resolve()
        if not image.is_file():
            raise ValueError(f"图片不存在：{image}")
        if image.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("单张图片不得超过 20 MiB")
        mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
        if not mime.startswith("image/"):
            raise ValueError("只允许图片文件")
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def infer(self, request: ModelRequest) -> ModelResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": self._image_url(path)}}
            for path in request.images
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
        }
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:  # noqa: S310
            result: dict[str, Any] = json.load(response)
        text = str(result["choices"][0]["message"]["content"])
        usage = {
            key: int(value)
            for key, value in result.get("usage", {}).items()
            if isinstance(value, int)
        }
        return ModelResponse(text, str(result.get("model", self.model)), usage)


class EnsembleAdapter:
    """融合多个本地 teacher 服务，输出投票、一致性与归一化熵。"""

    def __init__(self, adapters: Sequence[ModelAdapter]) -> None:
        if not adapters:
            raise ValueError("至少需要一个 teacher adapter")
        self.adapters = list(adapters)

    def infer(self, request: ModelRequest) -> ModelResponse:
        responses = [adapter.infer(request) for adapter in self.adapters]
        parsed = [
            json.loads(response.content.strip().strip("`\n").removeprefix("json"))
            for response in responses
        ]
        labels = [int(item["relevance_label"]) for item in parsed]
        counts = {label: labels.count(label) for label in range(4)}
        winner = max(
            counts,
            key=lambda label: (
                counts[label],
                sum(
                    float(item.get("confidence", 0))
                    for item in parsed
                    if int(item["relevance_label"]) == label
                ),
            ),
        )
        selected = [item for item in parsed if int(item["relevance_label"]) == winner]
        probabilities = [count / len(labels) for count in counts.values() if count]
        entropy = -sum(value * math.log(value) for value in probabilities) / math.log(4)
        result = {
            "relevance_label": winner,
            "confidence": sum(float(item.get("confidence", 0)) for item in selected)
            / len(selected),
            "reason": str(selected[0].get("reason", "")),
            "evidence": list(
                dict.fromkeys(str(value) for item in selected for value in item.get("evidence", []))
            ),
            "agreement": counts[winner] / len(labels),
            "entropy": entropy,
            "teacher_count": len(labels),
            "label_distribution": counts,
        }
        usage: dict[str, int] = {}
        for response in responses:
            for key, value in response.usage.items():
                usage[key] = usage.get(key, 0) + value
        return ModelResponse(
            json.dumps(result, ensure_ascii=False),
            "+".join(response.model for response in responses),
            usage,
        )
