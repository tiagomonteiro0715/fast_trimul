# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Scoped acceleration: @accelerate / accelerated() (CPU-only)."""

from fast_trimul import accelerate, accelerated
from fast_trimul.core.decorators import active_backend


def test_default_is_auto():
    assert active_backend() == "auto"


def test_accelerated_sets_and_reverts():
    assert active_backend() == "auto"
    with accelerated("torch"):
        assert active_backend() == "torch"
    assert active_backend() == "auto"


def test_accelerated_nesting():
    with accelerated("cuda"):
        assert active_backend() == "cuda"
        with accelerated("torch"):
            assert active_backend() == "torch"
        assert active_backend() == "cuda"
    assert active_backend() == "auto"


def test_accelerate_with_backend():
    @accelerate(backend="torch")
    def where():
        return active_backend()

    assert where() == "torch"
    assert active_backend() == "auto"


def test_accelerate_bare():
    @accelerate
    def where():
        return active_backend()

    assert where() == "auto"
