# Copyright (c) 2026 Tiago Monteiro. Apache License 2.0.
"""Make the CuTe DSL `cutlass` module importable without a kernel restart.

`nvidia-cutlass-dsl` ships the `cutlass` package inside a `python_packages`
subdirectory that its `.pth` file adds to `sys.path`. In a freshly `pip install`ed
Jupyter/Colab kernel that `.pth` hasn't been processed yet, so `import cutlass`
fails until the interpreter restarts. This re-runs the site setup so `import
cutlass` works immediately.
"""

import glob
import importlib
import site
import sys
from importlib.metadata import PackageNotFoundError, distribution

_done = False


def ensure_cutlass_on_path() -> None:
    global _done
    if _done or "cutlass" in sys.modules:
        return
    try:
        sp = str(distribution("nvidia-cutlass-dsl").locate_file(""))
    except PackageNotFoundError:
        return  # not installed -- let the real ImportError surface with a clear name
    site.addsitedir(sp)  # process the .pth the way interpreter startup would
    for d in glob.glob(f"{sp}/nvidia_cutlass_dsl*/python_packages"):
        if d not in sys.path:
            sys.path.insert(0, d)
    importlib.invalidate_caches()
    _done = True
