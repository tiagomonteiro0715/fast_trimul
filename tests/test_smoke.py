# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Import smoke tests -- run anywhere, no GPU needed."""

import fast_trimul


def test_import_and_version():
    assert isinstance(fast_trimul.__version__, str)
    assert fast_trimul.__version__


def test_public_api_exported():
    for name in ("FastTriangleMultiplication", "accelerate", "accelerated",
                 "verify", "list_backends", "patch_openfold", "patch_openfold3",
                 "patch_boltz", "patch_protenix", "functional"):
        assert hasattr(fast_trimul, name)


def test_torch_backend_registered():
    assert "torch" in fast_trimul.list_backends()


def test_functional_has_entrypoint():
    assert hasattr(fast_trimul.functional, "triangle_multiplication")
