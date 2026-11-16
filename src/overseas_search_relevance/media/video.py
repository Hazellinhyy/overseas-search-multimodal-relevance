"""FFmpeg 视频元信息、封面与均匀关键帧处理。"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FrameConfig:
    max_frames: int = 8
    max_width: int = 1280
    timeout_seconds: float = 60.0
    strategy: str = "uniform"


def inspect_video(path: Path, timeout: float = 15.0) -> dict[str, Any]:
    command = ["ffprobe", "-v", "error", "-show_entries", "format:stream", "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise ValueError(f"视频不可解析：{result.stderr.strip()[:300]}")
    payload: dict[str, Any] = json.loads(result.stdout)
    return payload


def extract_uniform_frames(
    path: Path, output_dir: Path, config: FrameConfig = FrameConfig()
) -> list[Path]:
    """按时长均匀抽帧并限制尺寸；不把只读第一帧当作视频理解。"""

    if config.max_frames < 1 or config.max_frames > 64:
        raise ValueError("max_frames 必须在 1～64")
    if config.strategy not in {"uniform", "scene"}:
        raise ValueError("strategy 必须是 uniform 或 scene")
    metadata = inspect_video(path)
    duration = float(metadata.get("format", {}).get("duration", 0.0))
    if duration <= 0:
        raise ValueError("视频时长无效")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{path.stem}_*.jpg"))
    if existing:
        return existing[: config.max_frames]
    interval = max(duration / config.max_frames, 0.1)
    pattern = output_dir / f"{path.stem}_%03d.jpg"
    selection = (
        f"fps=1/{interval},mpdecimate"
        if config.strategy == "uniform"
        else "select='gt(scene,0.30)',mpdecimate"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"{selection},scale='min({config.max_width},iw)':-2",
        "-frames:v",
        str(config.max_frames),
        "-y",
        str(pattern),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=config.timeout_seconds, check=False
    )
    if result.returncode != 0:
        raise ValueError(f"关键帧提取失败：{result.stderr.strip()[:300]}")
    return sorted(output_dir.glob(f"{path.stem}_*.jpg"))
