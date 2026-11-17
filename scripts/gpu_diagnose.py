#!/usr/bin/env python3
"""CUDA、BF16、显存和 NCCL 多卡诊断；不下载模型、不训练、不调用 API。"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--output", type=Path, default=Path("reports/phase1/gpu_diagnose.json"))
    args = parser.parse_args()
    result: dict[str, Any] = {"host": socket.gethostname(), "requested_gpu_count": args.gpu_count}
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        result["status"] = "BLOCKED"
        result["error"] = f"PyTorch unavailable: {error}"
        return _write(args.output, result)
    result["torch"] = torch.__version__
    result["cuda"] = {
        "available": torch.cuda.is_available(),
        "version": torch.version.cuda,
        "count": torch.cuda.device_count(),
    }
    visible = torch.cuda.device_count()
    if not torch.cuda.is_available() or (args.gpu_count is not None and visible < args.gpu_count):
        result["status"] = "BLOCKED"
        result["error"] = "GPU 数量或 CUDA 不满足预期"
        return _write(args.output, result)
    result["devices"] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        result["devices"].append(
            {
                "index": index,
                "name": props.name,
                "memory_bytes": props.total_memory,
                "bf16": getattr(props, "major", 0) >= 8,
            }
        )
    result["tests"] = {"cuda_tensor": False, "bf16": False, "memory_release": False, "nccl": False}
    device = torch.device("cuda:0")
    first = torch.randn((1024, 1024), device=device)
    second = torch.randn((1024, 1024), device=device)
    _ = first @ second
    torch.cuda.synchronize()
    result["tests"]["cuda_tensor"] = True
    _ = (first.to(torch.bfloat16) @ second.to(torch.bfloat16)).float()
    torch.cuda.synchronize()
    result["tests"]["bf16"] = True
    del first, second
    torch.cuda.empty_cache()
    result["tests"]["memory_release"] = torch.cuda.memory_allocated(0) == 0
    if os.environ.get("WORLD_SIZE") and int(os.environ["WORLD_SIZE"]) > 1:
        result["distributed_env"] = {
            key: os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        }
        try:
            import torch.distributed as dist  # type: ignore[import-not-found]

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
            value = torch.ones(
                1, device=torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            )
            dist.all_reduce(value)
            torch.cuda.synchronize()
            result["tests"]["nccl"] = bool(value.item() == int(os.environ["WORLD_SIZE"]))
            dist.destroy_process_group()
        except Exception as error:  # diagnostics must preserve the concrete failure
            result["nccl_error"] = f"{type(error).__name__}: {error}"
    else:
        result["nccl_note"] = (
            f"未在分布式启动环境中执行 NCCL；请用 torchrun --nproc_per_node={visible} 运行本脚本。"
        )
    result["status"] = "READY" if all(result["tests"].values()) else "PARTIAL"
    return _write(args.output, result)


def _write(output: Path, result: dict[str, Any]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
