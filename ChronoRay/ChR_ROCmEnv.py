"""ROCm / PyTorch HIP + Ray environment setup.

Call :func:`prepare_rocm_ray_env` **before** ``import ray`` on Ray 2.45+ when using
the PyTorch ROCm stack. The :mod:`ChronoRay` package runs this automatically on
import; you may also call it explicitly in scripts that import ``ray`` first.
"""

from __future__ import annotations

import os


def prepare_rocm_ray_env() -> None:
    """Set Ray + device env defaults for ROCm / HIP hosts (e.g. AMD MI300).

    - Sets Ray experimental flags so Ray does not clobber HIP/ROCR visibility.
    - Clears ``ROCR_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES`` for the initial
      ``import ray`` / ``ray.init`` handshake (see ``docs/AMD_ROCM.md`` for when
      to re-export ``ROCR_VISIBLE_DEVICES`` for RCCL / multi-node).
    """
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "1")
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.setdefault(
        "HIP_VISIBLE_DEVICES",
        os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
    )