"""防路径穿越、符号链接和 ZIP Bomb 的安全解压。"""

from __future__ import annotations

import hashlib
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile, ZipInfo


@dataclass(frozen=True, slots=True)
class ZipLimits:
    max_files: int = 10_000
    max_total_bytes: int = 20 * 1024**3
    max_single_bytes: int = 2 * 1024**3
    max_compression_ratio: float = 2_000.0


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_member(info: ZipInfo, limits: ZipLimits) -> None:
    target = Path(info.filename)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"ZIP 包含不安全路径：{info.filename}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP 包含符号链接：{info.filename}")
    if info.file_size > limits.max_single_bytes:
        raise ValueError(f"单文件超过限制：{info.filename}")
    ratio = info.file_size / max(info.compress_size, 1)
    if ratio > limits.max_compression_ratio:
        raise ValueError(f"压缩比异常：{info.filename}")


def safe_extract(zip_path: Path, destination: Path, limits: ZipLimits = ZipLimits()) -> list[Path]:
    """验证全部成员后再逐文件写入，异常时不会越出目标目录。"""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted: list[Path] = []
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        if len(members) > limits.max_files:
            raise ValueError("ZIP 文件数量超过限制")
        total = 0
        for info in members:
            _validate_member(info, limits)
            total += info.file_size
            if total > limits.max_total_bytes:
                raise ValueError("ZIP 总解压体积超过限制")
        for info in members:
            output = (destination / info.filename).resolve()
            if output != root and root not in output.parents:
                raise ValueError(f"目标路径越界：{info.filename}")
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            extracted.append(output)
    return extracted
