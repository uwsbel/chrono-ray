# Example 1 — Parameter estimation (honeycomb lander)

## Headless / cluster notes

- **Working directory:** `simulate_fn` changes into this directory before loading meshes so you can launch from any path, e.g. `python3 /path/to/chrono-ray/ex1-paramest-lander/paramest_lander.py`.
- **Irrlicht:** `HoneycombForceFunctor` does not require `pychrono.irrlicht` at import time; optional visualization imports are skipped on minimal images.
- **Quick check:** `python3 paramest_lander.py --smoke` runs two Tune trials with reduced logging.

## Dependencies

- `bayesian-optimization` is pinned in the repo `pyproject.toml` (`>=1.4.3,<2`) for compatibility with Ray Tune searchers.

## ROCm + Ray

See the repository **[`docs/AMD_ROCM.md`](../docs/AMD_ROCM.md)**.
