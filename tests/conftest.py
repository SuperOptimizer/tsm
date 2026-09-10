"""Keep tests from taking the whole machine: cap BLAS/torch threads before torch is imported."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import torch  # noqa: E402

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
