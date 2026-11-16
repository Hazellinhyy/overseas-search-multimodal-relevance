import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from overseas_search_relevance.media import video


def test_inspect_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {"format": {"duration": "3"}, "streams": []}
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args, 0, json.dumps(payload), ""),
    )
    assert video.inspect_video(tmp_path / "x.mp4") == payload


def test_inspect_video_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        video.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args, 1, "", "broken"),
    )
    with pytest.raises(ValueError, match="不可解析"):
        video.inspect_video(tmp_path / "x.mp4")


def test_frame_limit_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        video.extract_uniform_frames(tmp_path / "x.mp4", tmp_path, video.FrameConfig(max_frames=0))
