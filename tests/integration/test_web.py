import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from overseas_search_relevance.inference.jobs import BatchJobManager
from overseas_search_relevance.models.adapters import ModelResponse
from overseas_search_relevance.web import DemoHandler


class FakeAdapter:
    def infer(self, request):
        return ModelResponse(
            '{"relevance_label":2,"confidence":0.9,"reason":"ok","evidence":["title"]}',
            "fake",
            {},
        )


@pytest.fixture
def demo_server(tmp_path):
    DemoHandler.adapter = FakeAdapter()
    DemoHandler.frame_dir = tmp_path
    DemoHandler.jobs = BatchJobManager(tmp_path / "jobs", DemoHandler.adapter)
    server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_demo_page_health_and_inference(demo_server):
    assert "Multimodal Relevance Studio" in urllib.request.urlopen(demo_server).read().decode()
    assert json.load(urllib.request.urlopen(f"{demo_server}/health"))["status"] == "ok"
    request = urllib.request.Request(
        f"{demo_server}/api/infer",
        data=json.dumps({"query": "praia", "title": "praia"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    result = json.load(urllib.request.urlopen(request))
    assert result["relevance_label"] == 2
    assert result["tier"] == "B"


def test_demo_errors(demo_server):
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(f"{demo_server}/missing")
    assert missing.value.code == 404
    request = urllib.request.Request(f"{demo_server}/api/infer", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as invalid:
        urllib.request.urlopen(request)
    assert invalid.value.code == 400


def test_batch_upload(demo_server):
    request = urllib.request.Request(
        f"{demo_server}/api/jobs",
        data=b"sample_id,query,title\n1,praia,praia\n",
        headers={"X-Filename": "batch.csv", "Content-Type": "application/octet-stream"},
        method="POST",
    )
    submitted = json.load(urllib.request.urlopen(request))
    assert submitted["job_id"]
    deadline = time.monotonic() + 3
    while True:
        status = json.load(urllib.request.urlopen(f"{demo_server}/api/jobs/{submitted['job_id']}"))
        if status["status"] in {"completed", "failed"}:
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert status["status"] == "completed"
    result = urllib.request.urlopen(f"{demo_server}/api/jobs/{submitted['job_id']}/result").read()
    assert b'"relevance_label": 2' in result


def test_unknown_job_and_empty_upload(demo_server):
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(f"{demo_server}/api/jobs/unknown")
    assert missing.value.code == 404
    request = urllib.request.Request(
        f"{demo_server}/api/jobs",
        data=b"",
        headers={"X-Filename": "empty.csv"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as invalid:
        urllib.request.urlopen(request)
    assert invalid.value.code == 413
