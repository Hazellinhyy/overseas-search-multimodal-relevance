"""统一模型协议、测试适配器和受控 OpenAI-compatible 适配器。"""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.request
from dataclasses import dataclass
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
