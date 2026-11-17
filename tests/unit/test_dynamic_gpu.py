import pytest

from overseas_search_relevance.infrastructure.gpu import (
    audit_environment,
    choose_safe_profile,
    parse_float,
    parse_positive_int,
    resolve_runtime_config,
    write_audit,
)


def test_auto_configuration():
    config = resolve_runtime_config({})
    assert config["gpu_count"] is None
    assert config["tensor_parallel_size"] is None
    assert config["pipeline_parallel_size"] == 1


@pytest.mark.parametrize("count,tp", [(1, 1), (2, 2), (8, 4)])
def test_dynamic_gpu_counts(count, tp):
    config = resolve_runtime_config({"GPU_COUNT": str(count), "TENSOR_PARALLEL_SIZE": str(tp)})
    assert config["gpu_count"] == count


def test_invalid_parallelism():
    with pytest.raises(ValueError, match="整除"):
        resolve_runtime_config({"GPU_COUNT": "3", "TENSOR_PARALLEL_SIZE": "2"})


def test_invalid_boolean():
    with pytest.raises(ValueError, match="true 或 false"):
        resolve_runtime_config({"REMOTE_API_ENABLED": "yes"})


def test_scalar_validation():
    assert parse_positive_int(None, "X") is None
    assert parse_positive_int("3", "X") == 3
    assert parse_float("0.5", "X", minimum=0.1, maximum=1) == 0.5
    with pytest.raises(ValueError):
        parse_positive_int("0", "X")
    with pytest.raises(ValueError):
        parse_float("bad", "X", minimum=0.1, maximum=1)


@pytest.mark.parametrize(
    "devices, status",
    [([], "BLOCKED"), ([{"memory_mb": "4000"}], "BLOCKED"), ([{"memory_mb": "16000"}], "READY")],
)
def test_safe_profile(devices, status):
    assert choose_safe_profile({"gpu": {"devices": devices}})["status"] == status


def test_audit_and_write(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "overseas_search_relevance.infrastructure.gpu._run",
        lambda command: "GPU,0,16000,535" if "query-gpu=name" in " ".join(command) else None,
    )
    audit = audit_environment(tmp_path)
    assert audit["gpu"]["count"] == 1
    output = tmp_path / "audit.json"
    write_audit(output, audit, {"status": "READY"})
    assert output.exists()
