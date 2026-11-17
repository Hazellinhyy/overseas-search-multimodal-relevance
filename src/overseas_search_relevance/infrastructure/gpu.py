"""Linux/NVIDIA GPU 平台启动审计，不执行模型下载或远程调用。"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def parse_positive_int(value: str | None, name: str, *, maximum: int = 2_000_000) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数") from error
    if parsed < 1 or parsed > maximum:
        raise ValueError(f"{name} 必须在 1～{maximum} 范围内")
    return parsed


def parse_float(value: str | None, name: str, *, minimum: float, maximum: float) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} 必须是数字") from error
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} 必须在 {minimum}～{maximum} 范围内")
    return parsed


def parse_bool(value: str | None, name: str, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"{name} 必须是 true 或 false")
    return lowered == "true"


def resolve_runtime_config(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = environ or os.environ
    gpu_count = parse_positive_int(env.get("GPU_COUNT"), "GPU_COUNT", maximum=1024)
    tp = parse_positive_int(env.get("TENSOR_PARALLEL_SIZE"), "TENSOR_PARALLEL_SIZE", maximum=1024)
    pp = parse_positive_int(
        env.get("PIPELINE_PARALLEL_SIZE"), "PIPELINE_PARALLEL_SIZE", maximum=1024
    ) or 1
    if tp and gpu_count and (tp * pp > gpu_count or gpu_count % (tp * pp) != 0):
        raise ValueError(
            "TENSOR_PARALLEL_SIZE × PIPELINE_PARALLEL_SIZE 必须不超过 GPU_COUNT 且整除 GPU_COUNT"
        )
    dtype = env.get("MODEL_DTYPE", "bfloat16")
    if dtype not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError("MODEL_DTYPE 必须是 auto、bfloat16、float16 或 float32")
    return {
        "gpu_count": gpu_count,
        "gpu_memory_min_gib": parse_float(
            env.get("GPU_MEMORY_MIN_GIB"), "GPU_MEMORY_MIN_GIB", minimum=0.1, maximum=1024
        ),
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "model_path": env.get("MODEL_PATH"),
        "model_dtype": dtype,
        "max_model_len": parse_positive_int(env.get("MAX_MODEL_LEN"), "MAX_MODEL_LEN"),
        "max_num_seqs": parse_positive_int(env.get("MAX_NUM_SEQS"), "MAX_NUM_SEQS", maximum=100000),
        "gpu_memory_utilization": parse_float(
            env.get("GPU_MEMORY_UTILIZATION", "0.9"),
            "GPU_MEMORY_UTILIZATION",
            minimum=0.1,
            maximum=0.99,
        ),
        "model_download_enabled": parse_bool(
            env.get("MODEL_DOWNLOAD_ENABLED"), "MODEL_DOWNLOAD_ENABLED"
        ),
        "remote_api_enabled": parse_bool(
            env.get("REMOTE_API_ENABLED"), "REMOTE_API_ENABLED"
        ),
        "enable_training": parse_bool(env.get("ENABLE_TRAINING"), "ENABLE_TRAINING"),
    }


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
