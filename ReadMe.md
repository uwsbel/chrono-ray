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

### Option 2 — You need everything (including PyChrono)

Use this if you are starting fresh. This creates a full conda environment with PyChrono and all dependencies.

**1. Create the environment (miniconda or equivalent required):**
```bash
conda env create -f environment.yml
```

**2. Activate it:**
```bash
conda activate ChronoRay
```

**3. Install ChronoRay:**
```bash
pip install -e .
```

---

## Usage

```python
from ChronoRay import ChRParamEst

est = ChRParamEst(
    simulate_fn        = my_sim,
    est_rule           = ChRParamEst.EstRule.LS,
    param_sample_space = {
        "k": ChRParamEst.ChR_Distr.uniform(0, 10),
    },
    target_sim_outputs = {
        "output_1": 5.0,
    }
)
```

Run `ChRParamEst.info()` for full documentation.