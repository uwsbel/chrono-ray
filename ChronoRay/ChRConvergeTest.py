from collections.abc import Callable
import math
import os
import csv
import pickle

import ray

from ChronoRay.ChR_Config import ChR_Distr


class ChRConversionTest:

    @staticmethod
    def info() -> None:
        print("========================================================")
        print("ChRConversionTest - Sobol dispersion sample-size (N) convergence study")
        print("========================================================")
        print("""
            DESCRIPTION:
                Finds the smallest number of dispersion samples N at which the
                robustness metrics stop moving, so N can be fixed for the inner
                loop of ChRDispersionOpt.

                Holds ONE fixed design and evaluates it across a growing number of
                Sobol dispersion samples: N = 2, 4, 8, ... , 2^max_power. Because
                the Sobol sequence is EXTENSIBLE (the first 2^k points are a
                subset of the first 2^(k+1)), all 2^max_power sims are run ONCE and
                the metrics are recomputed on growing PREFIXES of that single
                result set -- no re-simulating per N.

                Self-contained: it owns its Sobol ticket generation and its
                one-sim-per-GPU fan-out (mirroring ChRDispersionOpt's internals),
                so it does not depend on any other workflow module.

            CONSTRUCTOR ARGUMENTS:
                simulate_fn (Callable) [REQUIRED]
                    simulate_fn(config) -> per-run raw output. config is the fixed
                    design merged with one dispersion ticket (standard ChR config).

                metrics_fn (Callable) [REQUIRED]
                    metrics_fn(outputs) -> dict[str, float]. Given the list of
                    per-run raw outputs, return the AGGREGATE metrics to watch
                    (e.g. {"p95_accel": ..., "p05_margin": ..., "crash_rate": ...}).
                    The study tracks the convergence of EACH returned key
                    separately. This is the UNFOLDED sibling of objective_fn:
                    same aggregation math, but returns the pieces as a dict
                    instead of summing them into one scalar.

                dispersion_sample_space (dict[str, ChR_Distr]) [REQUIRED]
                    Uncertain conditions to Sobol-sample.

                fixed_params (dict) [REQUIRED]
                    Concrete values of the design variables to hold fixed. Keys
                    must be disjoint from the dispersion keys. Pick a plausibly-
                    STRESSED design (near the constraint edges) so N has margin
                    for the real optimization.

                num_gpus (int) [REQUIRED]
                    G -- GPUs to fan the samples across.

                max_power (int) [OPTIONAL, default 8]  -> largest N = 2^max_power
                min_power (int) [OPTIONAL, default 1]  -> smallest N = 2^min_power

                convergence_tol (float) [OPTIONAL, default 0.02]
                    A metric counts as settled at N when every further doubling
                    changes it by less than this fraction of its final value.
                convergence_streak (int) [OPTIONAL, default 2]
                    Require at least this many stable doublings after N.

                cpus_per_sample / gpus_per_sample (int) [OPTIONAL, default 1 / 1]
                sobol_seed (int) [OPTIONAL, default 42]
                output_prefix (str) [OPTIONAL] -- stem for the .csv/.png/.pkl outputs
                FLAG_auto_run (bool) [OPTIONAL, default True]
        """)
        print("========================================================")

    ChR_Distr = ChR_Distr

    def __init__(
                self,
                simulate_fn: Callable,
                metrics_fn: Callable,
                dispersion_sample_space: dict[str, ChR_Distr],
                fixed_params: dict,
                num_gpus: int,
                max_power: int = 8,
                min_power: int = 1,
                convergence_tol: float = 0.02,
                convergence_streak: int = 2,
                cpus_per_sample: int = 1,
                gpus_per_sample: int = 1,
                sobol_seed: int = 42,
                output_prefix: str = "chr_conversion_test",
                FLAG_auto_run: bool = True) -> None:

        #1. mandatory parameters
        self.simulate_fn             = simulate_fn
        self.metrics_fn              = metrics_fn
        self.dispersion_sample_space = dispersion_sample_space
        self.fixed_params            = fixed_params
        self.num_gpus                = num_gpus

        #2. optional parameters
        self.max_power          = max_power
        self.min_power          = min_power
        self.convergence_tol    = convergence_tol
        self.convergence_streak = convergence_streak
        self.cpus_per_sample    = cpus_per_sample
        self.gpus_per_sample    = gpus_per_sample
        self.sobol_seed         = sobol_seed
        self.output_prefix      = output_prefix

        self._validate_inputs()

        #generate the FULL 2^max_power Sobol ticket set ONCE (frozen; prefixed later)
        self.num_dispersion_samples = 2 ** self.max_power
        self._tickets = self._generate_dispersion_tickets()

        self._report_config()

        #3. immediately run if FLAG_auto_run
        self.FLAG_auto_run = FLAG_auto_run
        self.results = None
        if FLAG_auto_run:
            self.results = self.run()

    def _validate_inputs(self) -> None:
        print("************************************************************")
        print("Validating inputs...")
        print("run ChRConversionTest.info() for usage.")
        print("************************************************************")

        if not callable(self.simulate_fn):
            raise ValueError("simulate_fn must be callable")
        if not callable(self.metrics_fn):
            raise ValueError("metrics_fn must be callable")

        for nm, sp in (("dispersion_sample_space", self.dispersion_sample_space),):
            if not isinstance(sp, dict):
                raise TypeError(f"{nm} must be a dict")
            if not all(isinstance(k, str) for k in sp):
                raise TypeError(f"{nm} keys must all be strings")
            if not all(getattr(v, "FLAG_is_chr_distr", False) for v in sp.values()):
                raise TypeError(f"{nm} values must all be ChR_Distr distributions")

        #dispersion distributions must be Sobol-samplable
        for name, d in self.dispersion_sample_space.items():
            if not ChRConversionTest._is_sobol_compatible(d):
                raise TypeError(
                    f"Dispersion parameter '{name}' uses a distribution incompatible with Sobol "
                    f"sampling. Compatible: uniform, loguniform, randint, choice, grid_search."
                )

        if not isinstance(self.fixed_params, dict):
            raise TypeError("fixed_params must be a dict of concrete design values")
        overlap = set(self.fixed_params) & set(self.dispersion_sample_space)
        if overlap:
            raise ValueError(f"fixed_params shares key(s) with dispersion space: {sorted(overlap)}")

        if not (isinstance(self.num_gpus, int) and self.num_gpus > 0):
            raise ValueError("num_gpus must be a positive integer")
        if not (0 <= self.min_power <= self.max_power):
            raise ValueError("require 0 <= min_power <= max_power")
        if self.convergence_streak < 1:
            raise ValueError("convergence_streak must be >= 1")

    def _report_config(self) -> None:
        Ns = self._powers(self.min_power, self.max_power)
        print("************************************************************")
        print("========================================================")
        print("ChRConversionTest configuration:")
        print("========================================================")
        print(f"  1. simulate_fn       : {self.simulate_fn.__name__}")
        print(f"  2. metrics_fn        : {self.metrics_fn.__name__}")
        print(f"  3. fixed_params      : {self.fixed_params}")
        print(f"  4. dispersion vars   :")
        for name, distr in self.dispersion_sample_space.items():
            print(f"       {name} : {ChR_Distr._format_distr(distr)}")
        print(f"  5. N schedule        : {Ns}  (all {Ns[-1]} sims run once, then prefixed)")
        print(f"  6. num_gpus          : {self.num_gpus}")
        print(f"  7. convergence       : rel-change < {self.convergence_tol} for >= {self.convergence_streak} doublings")
        print(f"  8. output_prefix     : {self.output_prefix}")
        print("************************************************************")

    # -------------------------------------------------------------------------
    # DISPERSION TICKET SAMPLING (Sobol, in-house)
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_distr_info(d) -> tuple:
        if isinstance(d, ChR_Distr._ChRDistrDict):
            return "GridSearch", None
        ray_tune_distr_type = type(d).__name__
        sampler = type(d.sampler).__name__.strip("_") if hasattr(d, "sampler") else None
        return ray_tune_distr_type, sampler

    @staticmethod
    def _is_sobol_compatible(d) -> bool:
        if isinstance(d, ChR_Distr._ChRDistrDict):
            return True
        distr_type, sampler = ChRConversionTest._get_distr_info(d)
        if distr_type == "Categorical":
            return True
        if distr_type == "Integer" and sampler == "Uniform":
            return True
        if distr_type == "Float" and sampler in ("Uniform", "LogUniform"):
            return True
        return False

    def _transform_unit_samples(self, names: list, samples) -> list:
        tickets = []
        for i in range(len(samples)):
            ticket = {}
            for j, name in enumerate(names):
                d = self.dispersion_sample_space[name]
                u = samples[i, j]
                distr_type, sampler = ChRConversionTest._get_distr_info(d)

                if distr_type == "Float" and sampler == "Uniform":
                    ticket[name] = d.lower + u * (d.upper - d.lower)
                elif distr_type == "Float" and sampler == "LogUniform":
                    ticket[name] = d.lower * (d.upper / d.lower) ** u
                elif distr_type == "Integer" and sampler == "Uniform":
                    ticket[name] = int(d.lower + u * (d.upper - d.lower))
                elif distr_type == "Categorical":
                    idx = min(int(u * len(d.categories)), len(d.categories) - 1)
                    ticket[name] = d.categories[idx]
                elif isinstance(d, ChR_Distr._ChRDistrDict):
                    values = d['grid_search']
                    idx = min(int(u * len(values)), len(values) - 1)
                    ticket[name] = values[idx]

            tickets.append(ticket)
        return tickets

    def _generate_dispersion_tickets(self) -> list:
        from scipy.stats import qmc
        names   = list(self.dispersion_sample_space.keys())
        n       = 2 ** self.max_power   # already a power of two -> balanced, prefix-safe
        sampler = qmc.Sobol(d=len(names), scramble=True, seed=self.sobol_seed)
        samples = sampler.random(n=n)
        return self._transform_unit_samples(names, samples)

    # -------------------------------------------------------------------------
    # FAN-OUT (in-house; one sim per GPU, run from the driver)
    # -------------------------------------------------------------------------
    @staticmethod
    def _run_dispersion_sample(simulate_fn, config):
        #run the user's sim for ONE merged config on ONE GPU, pinning the device
        #the way ChRDoE._execute_trial does (ROCm HIP vs CUDA aware).
        import os
        gpu_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
        if gpu_ids:
            gpu_id = str(gpu_ids[0])
            try:
                import torch
                is_hip = torch.version.hip is not None
            except ImportError:
                is_hip = False
            if is_hip:
                os.environ["HIP_VISIBLE_DEVICES"] = gpu_id
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return simulate_fn(config)

    def _run_dispersion(self, design: dict) -> list:
        #fan the frozen tickets across the GPUs for the fixed design. Called from
        #the driver (not inside a Tune trial), so num_gpus tasks schedule against
        #the whole cluster: G run at a time, the rest queue -> waves happen on
        #their own. Each ticket is MERGED with the design into one config.
        design = dict(design)

        remote_sample = ray.remote(max_calls=1, max_retries=0)(
            ChRConversionTest._run_dispersion_sample
        ).options(num_gpus=self.gpus_per_sample, num_cpus=self.cpus_per_sample)

        futures = [
            remote_sample.remote(self.simulate_fn, {**design, **ticket})
            for ticket in self._tickets
        ]
        return ray.get(futures)

    # -------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------
    def run(self) -> dict:
        Ns    = self._powers(self.min_power, self.max_power)
        N_max = Ns[-1]

        if not ray.is_initialized():
            ray.init()

        #fan out ALL N_max samples for the fixed design, ONCE.
        print(f"\nrunning {N_max} dispersion sims for the fixed design (fanned across {self.num_gpus} GPUs)...")
        raw_outputs = self._run_dispersion(self.fixed_params)

        #recompute metrics on growing prefixes of the single result set.
        metrics_by_N = self._compute_prefix_metrics(raw_outputs, self.metrics_fn, Ns)

        converged, recommended = self._detect_convergence(
            metrics_by_N, self.convergence_tol, self.convergence_streak)

        #persist: raw outputs (re-analysis w/o re-sim), metrics table, plot.
        pkl_path = f"{self.output_prefix}_raw.pkl"
        csv_path = f"{self.output_prefix}_metrics.csv"
        png_path = f"{self.output_prefix}_plot.png"
        with open(pkl_path, "wb") as f:
            pickle.dump({"Ns": Ns, "raw_outputs": raw_outputs, "metrics_by_N": metrics_by_N}, f)
        self._save_csv(metrics_by_N, csv_path)
        try:
            self._save_plot(metrics_by_N, converged, self.convergence_tol, png_path)
        except Exception as e:
            print(f"(plot skipped: {e})")

        self._report_results(metrics_by_N, converged, recommended)

        return {
            "Ns": Ns,
            "metrics_by_N": metrics_by_N,
            "converged_N_per_metric": converged,
            "recommended_N": recommended,
            "files": {"raw": pkl_path, "csv": csv_path, "plot": png_path},
        }

    # -------------------------------------------------------------------------
    # ANALYSIS (static, Ray-free -> unit-testable)
    # -------------------------------------------------------------------------
    @staticmethod
    def _powers(min_power, max_power):
        return [2 ** k for k in range(min_power, max_power + 1)]

    @staticmethod
    def _compute_prefix_metrics(raw_outputs, metrics_fn, Ns):
        return {N: metrics_fn(raw_outputs[:N]) for N in Ns}

    @staticmethod
    def _detect_convergence(metrics_by_N, tol, streak):
        """For each metric: the smallest N after which EVERY further doubling
        changes it by < tol (relative to its final value), with >= streak such
        stable doublings. Overall recommended N = the slowest metric's N (max);
        None if any metric hasn't settled by the largest N tested."""
        Ns   = sorted(metrics_by_N)
        keys = list(metrics_by_N[Ns[-1]].keys())
        converged = {}
        for key in keys:
            vals  = [metrics_by_N[N][key] for N in Ns]
            scale = abs(vals[-1]) + 1e-12
            rel   = [abs(vals[i] - vals[i - 1]) / scale for i in range(1, len(vals))]
            conv_N = None
            for i in range(len(vals)):
                tail = rel[i:]
                if len(tail) >= streak and all(r < tol for r in tail):
                    conv_N = Ns[i]
                    break
            converged[key] = conv_N
        ok = [v for v in converged.values() if v is not None]
        recommended = None if (any(v is None for v in converged.values()) or not ok) else max(ok)
        return converged, recommended

    @staticmethod
    def _save_csv(metrics_by_N, path):
        Ns   = sorted(metrics_by_N)
        keys = list(metrics_by_N[Ns[-1]].keys())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N"] + keys)
            for N in Ns:
                w.writerow([N] + [metrics_by_N[N].get(k, "") for k in keys])

    @staticmethod
    def _save_plot(metrics_by_N, converged, tol, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Ns   = sorted(metrics_by_N)
        keys = list(metrics_by_N[Ns[-1]].keys())
        n    = len(keys)
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False)

        for idx, key in enumerate(keys):
            ax   = axes[idx // cols][idx % cols]
            vals = [metrics_by_N[N][key] for N in Ns]
            ax.plot(Ns, vals, "o-", color="#3b6fb5")
            ax.axhline(vals[-1], ls="--", lw=1, color="#888", label=f"final = {vals[-1]:.4g}")
            cN = converged.get(key)
            if cN is not None:
                ax.axvline(cN, ls=":", lw=1.5, color="#c0392b", label=f"converged N = {cN}")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("N (dispersion samples)")
            ax.set_title(key)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis("off")

        fig.suptitle(f"Dispersion metric convergence (rel-change tol = {tol})", y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    def _report_results(self, metrics_by_N, converged, recommended):
        print("\n================ CONVERGENCE RESULTS ================")
        print("per-metric converged N:")
        for k, v in converged.items():
            print(f"    {k:20s}: {'NOT CONVERGED' if v is None else v}")
        if recommended is None:
            print("\nOVERALL: at least one metric had not settled by the largest N tested.")
            print("         -> increase max_power and re-run.")
        else:
            print(f"\nOVERALL recommended N (slowest metric): {recommended}")
            print(f"         -> inflate by ~1.5-2x for the optimization: use ~{2*recommended}.")
            print(f"         -> and RE-VERIFY at the optimized winner afterward.")
        print("====================================================")