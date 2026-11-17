import pytest

from overseas_search_relevance.infrastructure.gpu import resolve_runtime_config


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
