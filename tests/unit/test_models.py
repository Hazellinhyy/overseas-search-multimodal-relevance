import socket

import pytest

from overseas_search_relevance.models import adapters


def test_endpoint_requires_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        adapters.OpenAICompatibleAdapter("http://example.com", "key", "model")


def test_private_endpoint_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="私网"):
        adapters.OpenAICompatibleAdapter("https://example.com", "key", "model")
