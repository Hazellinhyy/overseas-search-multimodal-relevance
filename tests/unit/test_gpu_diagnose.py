import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("gpu_diagnose", Path("scripts/gpu_diagnose.py"))
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Tensor:
    def __matmul__(self, other):
        return _Tensor()


class _Cuda:
    def __init__(self, allocations):
        self.allocations = iter(allocations)

    def synchronize(self, device=None):
        return None

    def empty_cache(self):
        return None

    def memory_allocated(self, device):
        return next(self.allocations)


class _Torch:
    def __init__(self, allocations):
        self.cuda = _Cuda(allocations)

    @staticmethod
    def randn(shape, device):
        return _Tensor()


@pytest.mark.parametrize(
    "allocations, expected",
    [([8519680, 8519680, 8519680], True), ([8519680, 8519680, 10000000], False)],
)
def test_memory_release_baseline_and_growth(allocations, expected):
    details = _MODULE._memory_release_check(_Torch(allocations), "cuda:0")
    assert details["passed"] is expected
    assert details["baseline_allocated_bytes"] == allocations[0]
    assert details["final_allocated_bytes"] == allocations[-1]
    assert details["growth_bytes"] == allocations[-1] - allocations[0]
