# ruff: noqa: E501
"""仅绑定本机的轻量 Web Demo。"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from overseas_search_relevance.inference.jobs import BatchJobManager
from overseas_search_relevance.inference.service import infer_sample
from overseas_search_relevance.models.adapters import (
    EnsembleAdapter,
    LocalVLLMVisionAdapter,
    ModelAdapter,
)

PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Multimodal Relevance Studio</title><style>:root{--blue:#2563eb;--ink:#172033;--muted:#64748b}*{box-sizing:border-box}body{margin:0;font:15px system-ui;background:#f1f5f9;color:var(--ink)}header{background:linear-gradient(120deg,#0f172a,#1d4ed8);color:white;padding:34px max(5vw,24px)}header h1{margin:0 0 8px;font-size:28px}.wrap{max-width:1180px;margin:24px auto;padding:0 18px}.tabs{display:flex;gap:8px;margin-bottom:16px}.tabs button{background:white;color:var(--ink)}.tabs .active{background:var(--blue);color:white}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:white;padding:24px;border-radius:16px;box-shadow:0 8px 30px #17203312}.hidden{display:none}label{display:block;margin:13px 0 6px;font-weight:600}input,textarea{width:100%;padding:11px;border:1px solid #cbd5e1;border-radius:9px}textarea{min-height:72px}button{padding:11px 18px;border:0;border-radius:9px;background:var(--blue);color:white;cursor:pointer}.hint{color:var(--muted);font-size:13px}.drop{border:2px dashed #93c5fd;background:#eff6ff;padding:28px;text-align:center;border-radius:12px}pre{white-space:pre-wrap;min-height:180px;background:#0f172a;color:#dbeafe;padding:16px;border-radius:10px;overflow:auto}.bar{height:10px;background:#e2e8f0;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:#22c55e;width:0}@media(max-width:760px){.grid{grid-template-columns:1fr}}</style>
<header><h1>Multimodal Relevance Studio</h1><div>KML 私有模型 · 四档相关性 · A/B/C 分层 · 批量断点结果</div></header><main class="wrap"><div class="tabs"><button id="t1" class="active">单样本分析</button><button id="t2">批量任务</button></div>
<section id="single" class="grid"><div class="card"><form id="f"><label>Query *</label><input name="query" required placeholder="葡语或其他语言 query"><label>视频链接 *</label><input name="video_url" required placeholder="https://.../video.mp4"><details><summary>高级输入（可选）</summary><label>标题</label><input name="title"><label>本地视频路径（研发调试）</label><input name="video_path" placeholder="/path/to/video.mp4"><label>ASR</label><textarea name="asr"></textarea><label>OCR</label><textarea name="ocr"></textarea></details><p class="hint">系统自动下载受信任视频、抽取关键帧并通过 SSH 隧道发送到 KML。内部域名需配置 VIDEO_URL_ALLOWLIST。</p><button>开始多模态分析</button></form></div><div class="card"><h2>分析结果</h2><pre id="o">等待提交</pre></div></section>
<section id="batch" class="grid hidden"><div class="card"><h2>创建批量任务</h2><div class="drop"><input id="file" type="file" accept=".json,.jsonl,.csv,.xlsx,.zip"><p>支持单个 JSON/JSONL/CSV/XLSX，或包含数据文件与视频的 ZIP</p></div><p class="hint">数据至少包含 query；建议包含 sample_id/item_id、title、asr、ocr。Gold 字段不会进入模型 Prompt。</p><button id="upload">上传并运行</button></div><div class="card"><h2>任务状态</h2><div class="bar"><i id="progress"></i></div><p id="status">尚未创建任务</p><pre id="job">等待上传</pre><a id="download" class="hidden">下载 JSONL 结果</a></div></section></main>
<script>const show=(batch)=>{single.classList.toggle('hidden',batch);document.querySelector('#batch').classList.toggle('hidden',!batch);t1.classList.toggle('active',!batch);t2.classList.toggle('active',batch)};t1.onclick=()=>show(false);t2.onclick=()=>show(true);f.onsubmit=async(e)=>{e.preventDefault();o.textContent='推理中…';let p=Object.fromEntries(new FormData(f));let r=await fetch('/api/infer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});o.textContent=JSON.stringify(await r.json(),null,2)};upload.onclick=async()=>{let x=file.files[0];if(!x)return alert('请选择文件');status.textContent='上传中…';let r=await fetch('/api/jobs',{method:'POST',headers:{'X-Filename':encodeURIComponent(x.name),'Content-Type':'application/octet-stream'},body:x});let v=await r.json();job.textContent=JSON.stringify(v,null,2);if(v.job_id)poll(v.job_id)};async function poll(id){let r=await fetch('/api/jobs/'+id),v=await r.json();job.textContent=JSON.stringify(v,null,2);status.textContent=v.status+' '+(v.processed||0)+' / '+(v.total||'?');progress.style.width=(v.total?100*v.processed/v.total:0)+'%';if(v.status==='completed'){download.href=v.result;download.classList.remove('hidden');download.textContent='下载 JSONL 结果';let c=document.createElement('a');c.href=v.csv_result;c.textContent=' · 下载 CSV 结果';download.after(c)}else if(v.status!=='failed')setTimeout(()=>poll(id),1500)}</script></html>"""


class DemoHandler(BaseHTTPRequestHandler):
    adapter: ModelAdapter
    frame_dir: Path
    jobs: BatchJobManager

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path == "/health":
            self._send(200, "application/json", b'{"status":"ok"}')
        elif self.path.startswith("/api/jobs/"):
            parts = self.path.strip("/").split("/")
            job_id = parts[2] if len(parts) >= 3 else ""
            if len(parts) == 4 and parts[3] == "result":
                result = self.jobs.result_path(job_id)
                if result.is_file():
                    self._send(200, "application/x-ndjson", result.read_bytes())
                    return
            if len(parts) == 4 and parts[3] == "result.csv":
                result = self.jobs.root / job_id / "results.csv"
                if result.is_file():
                    self._send(200, "text/csv; charset=utf-8", result.read_bytes())
                    return
            status = self.jobs.status(job_id)
            if status:
                self._send(200, "application/json", json.dumps(status).encode())
                return
            self._send(404, "application/json", b'{"error":"job not found"}')
        else:
            self._send(404, "application/json", b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/jobs":
            self._upload_job()
            return
        if self.path not in {"/api/infer", "/api/label"}:
            self._send(404, "application/json", b'{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 1_000_000:
            self._send(413, "application/json", b'{"error":"invalid request size"}')
            return
        try:
            payload: dict[str, Any] = json.loads(self.rfile.read(length))
            result = infer_sample(payload, self.adapter, frame_dir=self.frame_dir)
            body = json.dumps(result, ensure_ascii=False).encode()
            self._send(200, "application/json; charset=utf-8", body)
        except Exception as error:
            error_body = json.dumps(
                {"error": type(error).__name__, "message": str(error)}, ensure_ascii=False
            )
            self._send(400, "application/json; charset=utf-8", error_body.encode())

    def _upload_job(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        filename = Path(unquote(self.headers.get("X-Filename", ""))).name
        if length < 1 or length > 20 * 1024**3 or not filename:
            self._send(413, "application/json", b'{"error":"invalid upload"}')
            return
        upload_dir = self.jobs.root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / f"{os.urandom(8).hex()}_{filename}"
        remaining = length
        with target.open("wb") as stream:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                stream.write(chunk)
                remaining -= len(chunk)
        job_id = self.jobs.submit(target)
        self._send(202, "application/json", json.dumps({"job_id": job_id}).encode())

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Web Demo 只允许绑定本机地址")
    endpoint = os.environ.get("MODEL_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
    model = os.environ.get("MODEL_NAME", "local-vlm")
    endpoints = [value.strip() for value in os.environ.get("MODEL_ENDPOINTS", endpoint).split(",")]
    adapters = [LocalVLLMVisionAdapter(value, model) for value in endpoints if value]
    DemoHandler.adapter = EnsembleAdapter(adapters) if len(adapters) > 1 else adapters[0]
    DemoHandler.frame_dir = Path(os.environ.get("FRAME_CACHE_DIR", "artifacts/frames/demo"))
    DemoHandler.jobs = BatchJobManager(Path("artifacts/jobs"), DemoHandler.adapter)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo: http://{args.host}:{args.port} -> {endpoint}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
