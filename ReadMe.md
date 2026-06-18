# ChronoRay

PyChrono + Ray Integration for Distributed Simulation Workflows

---

## Installation

There are two installation options depending on whether you already have PyChrono installed.

---

### Option 1 — You already have PyChrono

Use this if PyChrono is already installed in your environment.

```bash
pip install -e .
```

This will install ChronoRay along with its remaining dependencies (`ray`, `numpy`).

---

### AMD ROCm + Ray (PyTorch HIP)

If you use **PyTorch ROCm** and **Ray** on AMD GPUs, read **[`docs/AMD_ROCM.md`](docs/AMD_ROCM.md)**. The `ChronoRay` package sets the recommended **`RAY_EXPERIMENTAL_NOSET_*`** environment variables on import; you can also call **`prepare_rocm_ray_env()`** from `ChronoRay` before importing Ray in standalone scripts.

---

### Option 2 — You need everything (including PyChrono)

Use this if you are starting fresh. This creates a full conda environment with PyChrono and all dependencies.

**1. Create the environment (miniconda or equivalent required):**
```bash
conda env create --file environment.yml -n chrono-ray
```

**2. Activate it:**
```bash
conda activate chrono-ray
```

**3. Install ChronoRay:**
```bash
pip install -e .
```

---

## More Info To Come Later
