"""TSM package. Caps CPU threads by default so background stages never starve the machine."""
import os as _os

_cap = _os.environ.get("TSM_CPU_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, _cap)
try:
    import torch as _torch

    _torch.set_num_threads(int(_cap))
except Exception:  # torch missing or already initialised
    pass
