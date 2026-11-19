#!/usr/bin/env python3
"""OpenAI-compatible local server for an approved Qwen3-VL Transformers checkpoint."""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any


def materialize_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode local data URIs with Pillow, avoiding optional torchvision image codecs."""
    from PIL import Image  # type: ignore[import-not-found]

    output: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content: list[dict[str, Any]] = []
        for item in message.get("content", []):
            if item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("only embedded data-URI images are allowed")
                raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                if len(raw) > 20 * 1024 * 1024:
                    raise ValueError("image exceeds 20 MiB")
                image = Image.open(BytesIO(raw)).convert("RGB")
                content.append({"type": "image", "image": image})
            else:
                content.append(item)
        copied["content"] = content
        output.append(copied)
    return output


class VisionServer(BaseHTTPRequestHandler):
    model: Any
    processor: Any
    model_id: str
    lock = threading.Lock()

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.model_id, "object": "model", "owned_by": "local"}],
                },
            )
        elif self.path == "/health":
            self._send(200, {"status": "ok", "model": self.model_id, "vision": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 200 * 1024 * 1024:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            messages = materialize_images(payload["messages"])
            max_tokens = min(int(payload.get("max_tokens", 512)), 2048)
            with self.lock:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
                trimmed = [
                    output[len(source) :]
                    for source, output in zip(inputs.input_ids, generated, strict=True)
                ]
                text = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            self._send(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
                },
            )
        except Exception as error:
            self._send(
                400,
                {"error": {"type": type(error).__name__, "message": str(error)}},
            )

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if not (args.model / "config.json").is_file():
        raise SystemExit("model config.json missing")
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
    )

    VisionServer.model_id = args.model.name
    VisionServer.processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    VisionServer.model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        local_files_only=True,
    )
    ThreadingHTTPServer((args.host, args.port), VisionServer).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
