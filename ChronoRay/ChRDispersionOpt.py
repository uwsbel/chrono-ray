from collections.abc import Callable
import math
import textwrap

import ray

from ChronoRay.ChR_ChronoRay import ChR_ChronoRay
from ChronoRay.ChR_Config import ChR_Distr, ChR_SearchAlg


class ChRDispersionOpt:

    @staticmethod
    def info() -> None:
        print("========================================================")
        print("ChRDispersionOpt - ChronoRay (PyChrono + Ray) Robust Design Optimization Workflow")
        print("========================================================")
        print(textwrap.dedent("""
            DESCRIPTION:
                ChRDispersionOpt performs ROBUST design optimization: it searches
                for a design that performs well across a RANGE of uncertain
                conditions, not just at a single nominal point.

                It is a nested loop:
                    OUTER : Bayesian optimization proposes a design.
                    INNER : that design is simulated across N sampled uncertain
                            conditions ("dispersion tickets"), fanned out across
                            the GPUs (one sim per GPU).
                    SCORE : the user's objective_fn reduces the N raw outputs to
                            one robustness scalar (e.g. 95th-percentile of a
                            metric, tail probabilities, penalties for constraint
                            violations) which the optimizer minimizes/maximizes.

                The N dispersion tickets are drawn ONCE in the constructor (Sobol,
                pushed through the dispersion distributions) and REUSED for every
                design, so each design is judged against the same conditions
                (common random numbers) and the objective the optimizer sees is
                deterministic.

                The outer loop rides on the same ChR_ChronoRay backend as
                ChRBayesOpt, so it inherits pause / restore / resume for cluster
                wall-time limits.
                NOTE: resume is at DESIGN granularity -- a design interrupted
                mid-dispersion re-runs all N of its tickets on resume.

            --------------------------------------------------------
            CONSTRUCTOR ARGUMENTS:
            --------------------------------------------------------

                simulate_fn (Callable) [REQUIRED]
                    A PyChrono simulation for ONE design under ONE condition.
                    INPUT:   config (dict) -- the design-variable values AND one
                             sampled set of uncertain conditions, MERGED into a
                             single dict (same config style as every other ChR
                             workflow; the sim does not distinguish the two).
                    RETURNS: raw_output (any) -- consumed by objective_fn.
                    NOTE:    runs on a single GPU; no visualization (Irrlicht/VSG).

                objective_fn (Callable) [REQUIRED]
                    Reduces the N per-ticket outputs to one robustness scalar.
                    INPUT:   outputs (list) -- the N raw_output values, one per
                                               dispersion ticket, in ticket order.
                    RETURNS: score (float) -- scalar to minimize or maximize.
                    NOTE:    all robustness/constraint logic (percentiles, tail
                             probabilities, penalty terms) lives here.

                param_sample_space (dict[str, ChR_Distr]) [REQUIRED]
                    The DESIGN variables the optimizer searches over.
                    Keys   : design-variable names (match simulate_fn's design).
                    Values : ChR_Distr distributions defining the search ranges.

                dispersion_sample_space (dict[str, ChR_Distr]) [REQUIRED]
                    The UNCERTAIN conditions to sample for the inner loop.
                    Keys   : condition names (match simulate_fn's ticket).
                    Values : ChR_Distr distributions encoding their uncertainty.
                    NOTE:    Sobol-compatible distributions only (uniform,
                             loguniform, randint, choice, grid_search).
                    NOTE:    key names must NOT overlap with param_sample_space
                             (design + dispersion are merged into one config).

                mode (str) [REQUIRED]
                    Whether to minimize or maximize the objective. "min" or "max".

                num_gpus (int) [REQUIRED]
                    G -- number of GPUs the inner dispersion loop fans out across
                    per design. The trial reserves G GPUs; one sim per GPU.

                num_dispersion_samples (int) [OPTIONAL, default: 16]
                    N -- number of dispersion tickets per design. Rounded UP to
                    the next power of two (Sobol requirement).

                total_trials (int) [OPTIONAL, default: 10]
                    Number of designs the outer optimizer evaluates.

                max_concurrent_trials (int) [OPTIONAL, default: 1]
                    Designs evaluated at once. Default 1: one design owns all G
                    GPUs, and BayesOpt (sequential) prefers low concurrency.

                cpus_per_sample / gpus_per_sample (int) [OPTIONAL, default: 1 / 1]
                    Resources each inner sim requests. gpus_per_sample=1 gives
                    one-GPU-per-sim pinning.

                sobol_seed (int) [OPTIONAL, default: 42]
                    Seed for the (scrambled) Sobol tickets. Fixing it keeps the
                    tickets identical across runs and on resume.

                FLAG_log_to_file (bool) [OPTIONAL, default: False]
                    Redirect Ray/console output to a timestamped .txt file.

                FLAG_auto_run (bool) [OPTIONAL, default: True]
                    Controls the operating mode. See OPERATING MODES below.

                FLAG_start_restore_session (bool) [OPTIONAL, default: False]
                    Write this run to disk as a restorable experiment (use for the
                    FIRST run of a job you expect to be interrupted).

                FLAG_restore_from_previous_session (bool) [OPTIONAL, default: False]
                    Resume a previously started restorable experiment.
                    NOTE: mutually exclusive with FLAG_start_restore_session; if
                          both are True, start takes precedence.

                restore_experiment_name (str) [OPTIONAL]
                    Name of the restorable experiment. Must match across runs.

                restore_experiment_storage_path (str) [OPTIONAL]
                    Directory (relative to cwd) holding the restorable experiment.

            --------------------------------------------------------
            OPERATING MODES:
            --------------------------------------------------------

                MODE 1 - FLAG_auto_run=True (default)
                    The optimization runs immediately on construction.

                    Example:
                        opt = ChRDispersionOpt(
                            simulate_fn      = my_sim,          # (design, ticket) -> output
                            objective_fn     = my_robust_score, # (list of N outputs) -> float
                            param_sample_space = {
                                "strut_angle": ChRDispersionOpt.ChR_Distr.uniform(20, 40),
                                "crush_force": ChRDispersionOpt.ChR_Distr.uniform(1e3, 1e4),
                            },
                            dispersion_sample_space = {
                                "cohesion": ChRDispersionOpt.ChR_Distr.uniform(0.1, 5.0),
                                "slope":    ChRDispersionOpt.ChR_Distr.uniform(0.0, 15.0),
                            },
                            mode                   = "min",
                            num_gpus               = 8,
                            num_dispersion_samples = 64,
                            total_trials           = 30,
                        )

                MODE 2 - FLAG_auto_run=False
                    Construction stops after validation, ticket generation, and
                    config report. The user can configure optional settings before
                    running.

                    Example:
                        opt = ChRDispersionOpt(..., FLAG_auto_run=False)
                        opt.set_FLAG_log_to_file(True)
                        opt._build_backend()
                        opt.run()

            --------------------------------------------------------
            OPTIONAL SETTERS (MODE 2 only):
            --------------------------------------------------------

                set_FLAG_log_to_file(flag: bool)
                    Toggle whether Ray output is redirected to a file. Default: False
        """))
        print("========================================================")

    ChR_Distr = ChR_Distr

    def __init__(
                self,
                simulate_fn: Callable,
                objective_fn: Callable,
                param_sample_space: dict[str, ChR_Distr],
                dispersion_sample_space: dict[str, ChR_Distr],
                mode: str,
                num_gpus: int,
                num_dispersion_samples: int = 16,
                total_trials: int = 10,
                max_concurrent_trials: int = 1,
                cpus_per_sample: int = 1,
                gpus_per_sample: int = 1,
                sobol_seed: int = 42,
                FLAG_log_to_file: bool = False,
                FLAG_auto_run: bool = True,
                FLAG_start_restore_session: bool = False,
                FLAG_restore_from_previous_session: bool = False,
                restore_experiment_name: str = "chronoray_dispersion_experiment",
                restore_experiment_storage_path: str = "chronoray_dispersion_storage") -> None:

        #1. mandatory parameters
        self.simulate_fn             = simulate_fn
        self.objective_fn            = objective_fn
        self.param_sample_space      = param_sample_space
        self.dispersion_sample_space = dispersion_sample_space
        self.mode                    = mode
        self.num_gpus                = num_gpus
        self.num_dispersion_samples  = num_dispersion_samples
        self.total_trials            = total_trials
        self.max_concurrent_trials   = max_concurrent_trials

        #2. optional parameters
        self.cpus_per_sample = cpus_per_sample
        self.gpus_per_sample = gpus_per_sample
        self.sobol_seed      = sobol_seed
        self.FLAG_log_to_file = FLAG_log_to_file

        self.FLAG_start_restore_session = FLAG_start_restore_session
        self.FLAG_restore_from_previous_session = FLAG_restore_from_previous_session
        self.restore_experiment_name = restore_experiment_name
        self.restore_experiment_storage_path = restore_experiment_storage_path

        self._validate_inputs()

        #generate the N frozen dispersion tickets ONCE, here in the constructor.
        #reused for every design (common random numbers). may round N up to a
        #power of two (Sobol requirement) and update num_dispersion_samples.
        self._tickets = self._generate_dispersion_tickets()

        self._report_config()

        #3. immediately run if FLAG_auto_run is True
        self.FLAG_auto_run = FLAG_auto_run
        self.backend = None
        if FLAG_auto_run:
            self.backend = self._build_backend()
            self.run()

    def _validate_inputs(self) -> None:

        print("************************************************************")
        print("Validating inputs...")
        print("run ChRDispersionOpt.info() for information on how to use this workflow.")
        print("************************************************************")

        #1. simulate_fn / objective_fn checks
        if not callable(self.simulate_fn):
            raise ValueError("simulate_fn must be callable")
        if not callable(self.objective_fn):
            raise ValueError("objective_fn must be callable")

        #2. design space checks
        if not isinstance(self.param_sample_space, dict):
            raise TypeError("param_sample_space must be a dict")
        if not all(isinstance(k, str) for k in self.param_sample_space.keys()):
            raise TypeError("param_sample_space keys must all be strings")
        if not all(getattr(v, "FLAG_is_chr_distr", False) for v in self.param_sample_space.values()):
            raise TypeError("param_sample_space values must all be ChR_Distr distributions. Use ChRDispersionOpt.ChR_Distr.info() for available options.")

        #3. dispersion space checks
        if not isinstance(self.dispersion_sample_space, dict):
            raise TypeError("dispersion_sample_space must be a dict")
        if not all(isinstance(k, str) for k in self.dispersion_sample_space.keys()):
            raise TypeError("dispersion_sample_space keys must all be strings")
        if not all(getattr(v, "FLAG_is_chr_distr", False) for v in self.dispersion_sample_space.values()):
            raise TypeError("dispersion_sample_space values must all be ChR_Distr distributions. Use ChRDispersionOpt.ChR_Distr.info() for available options.")

        #3a. dispersion distributions must be Sobol-samplable
        for name, d in self.dispersion_sample_space.items():
            if not ChRDispersionOpt._is_sobol_compatible(d):
                raise TypeError(
                    f"Dispersion parameter '{name}' uses a distribution incompatible with Sobol sampling. "
                    f"Compatible distributions: uniform, loguniform, randint, choice, grid_search."
                )

        #3b. design and dispersion spaces must not share key names (they get
        #    merged into one config; a shared key would silently collide).
        shared = set(self.param_sample_space) & set(self.dispersion_sample_space)
        if shared:
            raise ValueError(
                f"param_sample_space and dispersion_sample_space share key(s) {sorted(shared)}. "
                f"Design and dispersion variables are merged into one config and must be disjoint."
            )

        #4. mode check
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")

        #5. resource / count checks
        if not (isinstance(self.num_gpus, int) and self.num_gpus > 0):
            raise ValueError("num_gpus must be a positive integer")
        if not (isinstance(self.num_dispersion_samples, int) and self.num_dispersion_samples > 0):
            raise ValueError("num_dispersion_samples must be a positive integer")
        if not (isinstance(self.total_trials, int) and self.total_trials > 0):
            raise ValueError("total_trials must be a positive integer")
        if self.gpus_per_sample <= 0 or self.cpus_per_sample <= 0:
            raise ValueError("cpus_per_sample and gpus_per_sample must be positive")

    def _report_config(self) -> None:

        print("************************************************************")
        print("========================================================")
        print("ChRDispersionOpt (\"ChronoRay Robust Design Optimization Workflow\") configuration:")
        print("========================================================")
        print(f"  1. simulate_fn             : {self.simulate_fn.__name__}")
        print(f"  2. objective_fn            : {self.objective_fn.__name__}")
        print(f"  3. param_sample_space (design) :")
        for name, distr in self.param_sample_space.items():
            print(f"       {name} : {ChR_Distr._format_distr(distr)}")
        print(f"  4. dispersion_sample_space (uncertain) :")
        for name, distr in self.dispersion_sample_space.items():
            print(f"       {name} : {ChR_Distr._format_distr(distr)}")
        print(f"  5. mode                    : {self.mode}")
        print(f"  6. total_trials (designs)  : {self.total_trials}")
        print(f"  7. num_dispersion_samples  : {self.num_dispersion_samples} (per design, Sobol)")
        print(f"  8. num_gpus (fan-out)      : {self.num_gpus}")
        print(f"  9. per-sample resources    : {{'cpu': {self.cpus_per_sample}, 'gpu': {self.gpus_per_sample}}}")
        print(f" 10. search_algorithm        : BAYESOPT (fixed)")
        print(f" 11. FLAG_log_to_file        : {self.FLAG_log_to_file}")
        print(f" 12. FLAG_start_restore_session         : {self.FLAG_start_restore_session}")
        print(f" 13. FLAG_restore_from_previous_session : {self.FLAG_restore_from_previous_session}")
        print(f" 14. restore_experiment_name           : {self.restore_experiment_name}")
        print(f" 15. restore_experiment_storage_path   : {self.restore_experiment_storage_path}")
        print("************************************************************")

    # -------------------------------------------------------------------------
    # DISPERSION TICKET SAMPLING (Sobol, run once in the constructor)
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
        distr_type, sampler = ChRDispersionOpt._get_distr_info(d)
        if distr_type == "Categorical":
            return True
        if distr_type == "Integer" and sampler == "Uniform":
            return True
        if distr_type == "Float" and sampler in ("Uniform", "LogUniform"):
            return True
        return False

    def _transform_unit_samples(self, names: list, samples) -> list:
        #push [0,1]^d Sobol coordinates through the dispersion distributions
        tickets = []
        for i in range(len(samples)):
            ticket = {}
            for j, name in enumerate(names):
                d = self.dispersion_sample_space[name]
                u = samples[i, j]
                distr_type, sampler = ChRDispersionOpt._get_distr_info(d)

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
        names  = list(self.dispersion_sample_space.keys())

        #Sobol is balanced only at powers of two -- round N up if needed
        n_pow2 = 2 ** math.ceil(math.log2(self.num_dispersion_samples))
        if n_pow2 != self.num_dispersion_samples:
            print(f"  NOTE: Sobol requires power-of-2 samples. "
                  f"Rounding num_dispersion_samples {self.num_dispersion_samples} up to {n_pow2}.")
            self.num_dispersion_samples = n_pow2

        sampler = qmc.Sobol(d=len(names), scramble=True, seed=self.sobol_seed)
        samples = sampler.random(n=self.num_dispersion_samples)
        return self._transform_unit_samples(names, samples)

    # -------------------------------------------------------------------------
    # INNER-LOOP FAN-OUT (one design -> N sims across the GPUs)
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
        #The per-design sim handed to the backend as simulate_fn. Passed as a
        #BOUND METHOD (self._run_dispersion) -- exactly how ChR_ChronoRay hands
        #self._trial to Tune -- so the backend calls it directly like any ordinary
        #simulate_fn. The workflow class itself is NOT a callable.
        #
        #For the given design, it fans the N FROZEN dispersion tickets across the
        #GPUs and returns the list of N raw outputs (for the user's objective_fn).
        #Each ticket is MERGED with the design into one config, so the user's sim
        #keeps the standard simulate_fn(config) signature.
        #
        #PLACEMENT: these tasks are launched from inside a Tune trial. Ray Tune sets
        #placement_group_capture_child_tasks=True, so each GPU task is captured into
        #the trial's placement group and lands on one of the G GPU bundles reserved
        #for that trial (see _build_backend). With only G GPUs reserved, Ray runs G
        #at a time and queues the rest, so the waves of N = b*G happen on their own.
        design = dict(design)

        remote_sample = ray.remote(max_calls=1, max_retries=0)(
            ChRDispersionOpt._run_dispersion_sample
        ).options(num_gpus=self.gpus_per_sample, num_cpus=self.cpus_per_sample)

        futures = [
            remote_sample.remote(self.simulate_fn, {**design, **ticket})
            for ticket in self._tickets
        ]

        #list of N raw outputs, in ticket order -> handed to the user's objective_fn.
        #SEAM: for sim-level resume (skip already-done tickets on restart), a
        #ChRDoE-style hash/skip list would slot in here. Left out until benchmarks
        #show a single design's inner loop can exceed a wall-time window.
        return ray.get(futures)

    # -------------------------------------------------------------------------
    # <3 METHODS IN USER INTERFACE
    # -------------------------------------------------------------------------

    def set_FLAG_log_to_file(self, flag: bool) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set FLAG_log_to_file after (auto-)run has started")
        self.FLAG_log_to_file = flag

    def _build_backend(self) -> ChR_ChronoRay:

        if self.backend is not None and self.FLAG_auto_run:
            raise ValueError("Cannot rebuild since backend has already been built and (auto-)run has started.")

        from ray.tune import PlacementGroupFactory

        #reserve, per design (per trial): 1 orchestrator bundle + G one-GPU worker
        #bundles. The trial only starts once all G GPUs are free, and the inner
        #dispersion tasks schedule into these reserved bundles (no PG deadlock).
        bundles = [{"CPU": 1}] + [
            {"CPU": self.cpus_per_sample, "GPU": self.gpus_per_sample}
            for _ in range(self.num_gpus)
        ]
        resources_per_trial = PlacementGroupFactory(bundles, strategy="PACK")

        #simulate_fn = self._run_dispersion (bound method): the backend calls it
        #per design -> [N raw outputs]; objective_fn = the user's, which reduces
        #those N outputs to a scalar.
        return ChR_ChronoRay(
            simulate_fn=self._run_dispersion,
            objective_fn=self.objective_fn,
            param_space=self.param_sample_space,
            resources_per_trial=resources_per_trial,
            num_trials=self.total_trials,
            max_concurrent_trials=self.max_concurrent_trials,
            mode=self.mode,
            search_algorithm=ChR_SearchAlg.BAYESOPT,
            search_kwargs={},
            FLAG_start_restore_session=self.FLAG_start_restore_session,
            FLAG_restore_from_previous_session=self.FLAG_restore_from_previous_session,
            restore_experiment_name=self.restore_experiment_name,
            restore_experiment_storage_path=self.restore_experiment_storage_path,
        )

    def run(self) -> None:
        if self.backend is None:
            raise ValueError("backend has not been built yet. Call _build_backend() first.")

        self.FLAG_auto_run = True
        self.backend.run(FLAG_log_to_file=self.FLAG_log_to_file)