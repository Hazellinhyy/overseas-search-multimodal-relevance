from email.message import Message

import pytest

from overseas_search_relevance.media.fetch import fetch_video


class Response:
    def __init__(self, content_type: str, data: bytes):
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.data = data
        self.sent = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, size):
        if self.sent:
            return b""
        self.sent = True
        return self.data


def test_fetch_video_allowlisted(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_URL_ALLOWLIST", "video.internal")
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *args: [(None, None, None, None, ("10.0.0.1", 0))]
    )
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: Response("video/mp4", b"video")
    )
    path = fetch_video("https://video.internal/1.mp4", tmp_path)
    assert path.read_bytes() == b"video"


def test_fetch_video_rejects_private_unlisted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *args: [(None, None, None, None, ("10.0.0.1", 0))]
    )
    with pytest.raises(ValueError, match="ALLOWLIST"):
        fetch_video("https://private.invalid/1.mp4", tmp_path)
