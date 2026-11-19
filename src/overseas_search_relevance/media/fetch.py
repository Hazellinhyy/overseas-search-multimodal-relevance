"""受控视频 URL 获取：仅下载显式允许的 HTTP(S) 视频。"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def _allowed_host(hostname: str) -> bool:
    allowlist = {
        value.strip().lower()
        for value in os.environ.get("VIDEO_URL_ALLOWLIST", "").split(",")
        if value.strip()
    }
    return hostname.lower() in allowlist or any(
        hostname.lower().endswith(f".{suffix}") for suffix in allowlist
    )


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("video_url 必须是有效 HTTP(S) 地址")
    allowlisted = _allowed_host(parsed.hostname)
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        address = ipaddress.ip_address(result[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
        ) and not allowlisted:
            raise ValueError("内部视频域名必须通过 VIDEO_URL_ALLOWLIST 显式授权")


def fetch_video(url: str, cache_dir: Path, *, max_bytes: int = 2 * 1024**3) -> Path:
    _validate_url(url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.mp4"
    if target.is_file() and target.stat().st_size:
        return target
    request = urllib.request.Request(url, headers={"User-Agent": "overseas-relevance/1.0"})
    temporary = target.with_suffix(".tmp")
    total = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as stream,
        ):  # noqa: S310
            content_type = response.headers.get_content_type()
            if not content_type.startswith("video/") and content_type != "application/octet-stream":
                raise ValueError(f"视频链接返回了非视频内容：{content_type}")
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("视频超过 2 GiB 限制")
                stream.write(chunk)
        temporary.replace(target)
        return target
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
