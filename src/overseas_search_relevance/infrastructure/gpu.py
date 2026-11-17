"""Linux/NVIDIA GPU 平台启动审计，不执行模型下载或远程调用。"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def audit_environment(code_root: Path = Path("/code")) -> dict[str, Any]:
    """收集 GPU、CUDA、CPU、RAM、磁盘和运行路径。"""
    nvidia = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,index,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if nvidia:
        for line in nvidia.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 4:
                gpus.append(
                    {
                        "name": fields[0],
                        "index": fields[1],
                        "memory_mb": fields[2],
                        "driver": fields[3],
                    }
                )
    disk = shutil.disk_usage(code_root if code_root.exists() else Path.cwd())
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "code_root": str(code_root),
        "cwd": str(Path.cwd()),
        "gpu": {
            "count": len(gpus),
            "devices": gpus,
            "cuda": _run(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"])
            or "unknown",
        },
        "cpu": {"model": platform.processor() or "unknown", "logical_count": os.cpu_count()},
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free, "used_bytes": disk.used},
        "policy": {"network_model_download": False, "remote_api": False, "training": False},
    }


def choose_safe_profile(audit: dict[str, Any]) -> dict[str, Any]:
    """按显存保守选择资源档位；无法识别 GPU 时拒绝启动大型模型。"""
    devices = audit.get("gpu", {}).get("devices", [])
    if not devices:
        return {"status": "BLOCKED", "reason": "未检测到 NVIDIA GPU"}
    try:
        minimum = min(float(device["memory_mb"]) for device in devices)
    except (KeyError, TypeError, ValueError):
        return {"status": "BLOCKED", "reason": "GPU 显存信息不可解析"}
    if minimum < 8000:
        return {
            "status": "BLOCKED",
            "reason": "单卡显存低于保守 VLM 最低要求",
            "memory_mb": minimum,
        }
    return {
        "status": "READY",
        "tensor_parallel": len(devices),
        "memory_mb": minimum,
        "max_image_count": 8,
    }


def write_audit(output: Path, audit: dict[str, Any], profile: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"audit": audit, "profile": profile}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
