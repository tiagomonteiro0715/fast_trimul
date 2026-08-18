# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""The dispatcher: _order() and run()'s fallback chain (CPU-only)."""

import pytest
import torch

from fast_trimul.core import dispatch, registry


def test_order_forced_torch():
    assert dispatch._order(torch.device("cpu"), "torch") == ["torch"]


def test_order_forced_cuda_adds_torch():
    assert dispatch._order(torch.device("cpu"), "cuda") == ["cuda", "torch"]


def test_order_auto_on_cpu():
    assert dispatch._order(torch.device("cpu"), "auto") == ["torch"]


def test_order_auto_on_cuda_device():
    assert dispatch._order(torch.device("cuda"), "auto") == ["cuda", "torch"]


def test_run_dispatches_to_chosen_backend():
    marker = object()

    @registry.backend("__run_ok", dtypes={torch.float32}, min_align=1)
    class _Ok:
        def __init__(self, caps):
            self.caps = caps

        def execute(self, inp, params):
            return marker

    try:
        out = dispatch.run(torch.randn(1, 8, 8, 4), None, None, prefer="__run_ok")
        assert out is marker
    finally:
        registry._BACKENDS.pop("__run_ok", None)


def test_run_raises_when_nothing_can_run():
    # A 2-D input: every backend's guard returns None, so nothing runs.
    with pytest.raises(RuntimeError):
        dispatch.run(torch.randn(2, 2), None, None, prefer="torch")
