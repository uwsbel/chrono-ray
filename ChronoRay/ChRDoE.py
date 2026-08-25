from collections.abc import Callable
from enum import Enum, auto
import textwrap
import itertools

import ray
import numpy as np

from ChronoRay.ChR_Config import ChR_Distr


class ChRDoE:

    @staticmethod
    def info() -> None:
        print("========================================================")
        print("ChRDoE - ChronoRay (PyChrono + Ray) Design of Experiments Workflow")
        print("========================================================")
        print(textwrap.dedent("""
            DESCRIPTION:
                ChRDoE is a design of experiments framework for PyChrono simulations.
                It generates a set of parameter configurations according to a chosen
                sampling design and runs all configurations in parallel using Ray.
                Each trial reserves the resources given by resources_per_trial
                (default 1 CPU + 1 GPU). Concurrency is derived automatically from
                the resources Ray was granted, running floor(detected / per-trial)
                trials at once for maximum utilization.

                Results are logged to disk as each trial finishes (a CSV of successes
                and a JSON-lines file of failures). A sweep that dies partway can be
                re-run: any config already logged (success OR failure) is skipped on
                restart, matched by a hash of its parameter values.

                Unlike ChRParamEst and ChRBayesOpt, there is no objective function
                or scoring: the user's simulate_fn is responsible for logging
                any outputs of interest internally.

            --------------------------------------------------------
            CONSTRUCTOR ARGUMENTS:
            --------------------------------------------------------

                simulate_fn (Callable) [REQUIRED]
                    A PyChrono simulation function.
                    INPUT:   config (dict): parameter names and their sampled values.
                    RETURNS: None (logging must be handled inside simulate_fn).
                    NOTE:    no visualization (e.g. Irrlicht, VSG) should take place.

                param_sample_space (dict[str, ChR_Distr]) [REQUIRED]
                    Parameters to vary and their distributions defining the search space.
                    Keys   : parameter names (must match keys expected by simulate_fn).
                    Values : ChR_Distr distributions.
                    NOTE:    run ChRDoE.ChR_Distr.info() for available distributions.
                    NOTE:    for FULL_FACTORIAL, only the following distributions are
                             compatible: uniform, loguniform, randint, choice, grid_search.

                sampling_design (ChRDoE.SamplingDesign) [REQUIRED]
                    The sampling strategy used to generate parameter configurations.
                    Available options:
                        ChRDoE.SamplingDesign.FULL_FACTORIAL  : all combinations of discrete parameter values
                        ChRDoE.SamplingDesign.LATIN_HYPERCUBE : space-filling Latin Hypercube Sampling (LHS)
                        ChRDoE.SamplingDesign.SOBOL           : space-filling Sobol sequence

                num_trials (int) [OPTIONAL, default: 10]
                    Number of total configurations to generate.
                    Used by LATIN_HYPERCUBE and SOBOL.
                    Ignored for FULL_FACTORIAL (determined by factorial_level_spacing).
                    NOTE:    SOBOL requires a power-of-2 value (will auto round up).

                resources_per_trial (dict[str, int]) [OPTIONAL, default: {"cpu": 1, "gpu": 0}]
                    CPU/GPU resources reserved per trial. Concurrency is derived
                    automatically from this: the workflow detects the resources Ray
                    was granted and runs floor(detected / per-trial) trials at once,
                    for whichever of CPU / GPU runs out first. Set "gpu" to 0 for
                    CPU-only simulations (then CPU is the binding resource).
                    In auto-run mode this default is used as-is. In manual mode
                    (FLAG_auto_run=False) it can be overridden via
                    set_resources_per_trial before running, which is how a scaling
                    sweep is expressed: larger per-trial reservations yield fewer
                    concurrent trials, smaller ones yield more.

                factorial_level_spacing (dict[str, int]) [OPTIONAL, default: None]
                    Number of discrete levels per parameter for FULL_FACTORIAL.
                    Each continuous parameter is evenly divided into this many values
                    across its range. Required for uniform and loguniform distributions.
                    Not required for randint, choice, or grid_search (those are
                    already discrete).
                    Example: {"k": 5, "m": 4}
                        "k" → 5 evenly spaced values across its range
                        "m" → 4 log-spaced values across its range

                results_prefix (str) [OPTIONAL, default: "chr_doe"]
                    Filename prefix for the persistence logs, written to the current
                    working directory:
                        <prefix>_results.csv    : one row per completed trial
                        <prefix>_failures.jsonl : one record per failed trial
                    On restart, any config already present in either file (matched by
                    a hash of its parameter values) is skipped, so a crashed sweep
                    resumes instead of starting over. Use a distinct prefix per sweep
                    to keep separate campaigns from sharing logs.

                FLAG_log_to_file (bool) [OPTIONAL, default: False]
                    If True, Ray's console output is redirected to a timestamped
                    .txt file in the current working directory. User prints are also 
                    redirected to .txt file. Default keeps all output in the console.

                FLAG_auto_run (bool) [OPTIONAL, default: True]
                    Controls the operating mode. See OPERATING MODES below.

            --------------------------------------------------------
            OPERATING MODES:
            --------------------------------------------------------

                MODE 1 (FLAG_auto_run=True, default)
                    Configurations are generated and all trials run immediately
                    on construction. No further configuration is possible.

                    Example (Latin hypercube):
                        doe = ChRDoE(
                            simulate_fn        = my_sim,
                            param_sample_space = {
                                "k": ChRDoE.ChR_Distr.uniform(0, 10),
                                "m": ChRDoE.ChR_Distr.loguniform(1e-2, 1e2),
                            },
                            sampling_design = ChRDoE.SamplingDesign.LATIN_HYPERCUBE,
                            num_trials     = 20,
                        )

                    Example (full factorial):
                        doe = ChRDoE(
                            simulate_fn        = my_sim,
                            param_sample_space = {
                                "k": ChRDoE.ChR_Distr.uniform(0, 10),
                                "m": ChRDoE.ChR_Distr.loguniform(1e-2, 1e2),
                            },
                            sampling_design = ChRDoE.SamplingDesign.FULL_FACTORIAL,
                            factorial_level_spacing       = {"k": 5, "m": 4},
                        )

                MODE 2 (FLAG_auto_run=False)
                    Construction stops after validation and config report.
                    The user can configure optional settings before running.

                    Example:
                        doe = ChRDoE(..., FLAG_auto_run=False)
                        doe.set_resources_per_trial(cpu=4, gpu=0)
                        doe.set_FLAG_log_to_file(True)
                        doe._build()
                        doe.run()

            --------------------------------------------------------
            OPTIONAL SETTERS (MODE 2 only):
            --------------------------------------------------------

                set_resources_per_trial(cpu: int, gpu: int)
                    Set the CPU/GPU resources reserved per trial, which in turn
                    sets concurrency (floor of detected resources / per-trial).
                    Default: cpu=1, gpu=1

                set_FLAG_log_to_file(flag: bool)
                    Toggle whether Ray output is redirected to a file. Default: False
        """))
        print("========================================================")

    ChR_Distr = ChR_Distr

    class SamplingDesign(Enum):
        FULL_FACTORIAL  = auto()
        LATIN_HYPERCUBE = auto()
        SOBOL           = auto()

    def __init__(
                self,
                simulate_fn: Callable,
                param_sample_space: dict[str, ChR_Distr],
                sampling_design: SamplingDesign,
                num_trials: int = 10,
                resources_per_trial: dict[str, int] = {"cpu": 1, "gpu": 0},
                factorial_level_spacing: dict[str, int] = None,
                results_prefix: str = "chr_doe",
                FLAG_log_to_file: bool = False,
                FLAG_auto_run: bool = True) -> None:

        #1. mandatory parameters
        self.simulate_fn                 = simulate_fn
        self.param_sample_space          = param_sample_space
        self.sampling_design             = sampling_design
        self.num_trials                  = num_trials
        self.factorial_level_spacing     = factorial_level_spacing or {}

        #2. optional parameters
        self.resources_per_trial   = resources_per_trial
        self.FLAG_log_to_file           = FLAG_log_to_file

        #persistence: results + failures logged here; used to skip done configs on restart
        self.results_prefix    = results_prefix
        self._success_log_path = f"{results_prefix}_results.csv"
        self._failure_log_path = f"{results_prefix}_failures.jsonl"

        self._validate_inputs()
        self._report_config()

        #3. immediately run if FLAG_auto_run is True
        self.FLAG_auto_run = FLAG_auto_run
        self._configs = None

        if FLAG_auto_run:
            self._build()
            self.run()


    def _validate_inputs(self) -> None:

        print("************************************************************")
        print("Validating inputs...")
        print("run ChRDoE.info() for information on how to use this workflow.")
        print("************************************************************")

        #1. simulate_fn check
        if not callable(self.simulate_fn):
            raise ValueError("simulate_fn must be callable")

        #2. param_sample_space checks
        if not isinstance(self.param_sample_space, dict):
            raise TypeError("param_sample_space must be a dict")

        if not all(isinstance(k, str) for k in self.param_sample_space.keys()):
            raise TypeError("param_sample_space keys must all be strings")

        if not all(getattr(v, "FLAG_is_chr_distr", False) for v in self.param_sample_space.values()):
            raise TypeError("param_sample_space values must all be ChR_Distr distributions. Use ChRDoE.ChR_Distr.info() for available options.")

        #3. sampling design check
        if not isinstance(self.sampling_design, ChRDoE.SamplingDesign):
            raise TypeError("sampling_design must be a ChRDoE.SamplingDesign enum member.")

        #3a. resources_per_trial check. Concurrency is derived from this against
        #    the resources Ray detects, so it must have int cpu/gpu counts with a
        #    positive CPU (gpu may be 0 for CPU-only simulations).
        if not isinstance(self.resources_per_trial, dict):
            raise TypeError("resources_per_trial must be a dict, e.g. {'cpu': 1, 'gpu': 1}")
        if "cpu" not in self.resources_per_trial or "gpu" not in self.resources_per_trial:
            raise ValueError("resources_per_trial must contain both 'cpu' and 'gpu' keys")
        if not (isinstance(self.resources_per_trial["cpu"], int) and self.resources_per_trial["cpu"] > 0):
            raise ValueError("resources_per_trial['cpu'] must be a positive integer")
        if not (isinstance(self.resources_per_trial["gpu"], int) and self.resources_per_trial["gpu"] >= 0):
            raise ValueError("resources_per_trial['gpu'] must be a non-negative integer")

        #4. full factorial specific checks
        if self.sampling_design == ChRDoE.SamplingDesign.FULL_FACTORIAL:

            #4a. check all distributions are factorial compatible
            for name, d in self.param_sample_space.items():
                if not ChRDoE._is_factorial_compatible(d):
                    raise TypeError(
                        f"Parameter '{name}' uses a distribution incompatible with FULL_FACTORIAL. "
                        f"Compatible distributions: uniform, loguniform, randint, choice, grid_search."
                    )

            #4b. check factorial_level_spacing keys are strings
            if not all(isinstance(k, str) for k in self.factorial_level_spacing.keys()):
                raise TypeError("factorial_level_spacing keys must all be strings.")

            #4c. check factorial_level_spacing values are positive integers
            if not all(isinstance(v, int) and v > 0 for v in self.factorial_level_spacing.values()):
                raise TypeError("factorial_level_spacing values must all be positive integers.")

            #4d. check every continuous param has an entry
            for name, d in self.param_sample_space.items():
                if ChRDoE._needs_factorial_level_spacing(d) and name not in self.factorial_level_spacing:
                    raise ValueError(
                        f"Parameter '{name}' requires a factorial_level_spacing entry for FULL_FACTORIAL. "
                        f"Example: factorial_level_spacing={{'{name}': 5}}"
                    )

            #4e. check no extra keys in factorial_level_spacing
            for name in self.factorial_level_spacing.keys():
                if name not in self.param_sample_space:
                    raise ValueError(
                        f"factorial_level_spacing contains key '{name}' which is not in param_sample_space."
                    )


    def _report_config(self) -> None:

        print("************************************************************")
        print("========================================================")
        print("ChRDoE (\"ChronoRay Design of Experiments Workflow\") configuration:")
        print("========================================================")
        print(f"  1. simulate_fn        : {self.simulate_fn.__name__}")
        print(f"  2. sampling_design    : {self.sampling_design.name}")
        print(f"  3. param_sample_space :")

        for name, distr in self.param_sample_space.items():
            line = f"       {name} : {ChR_Distr._format_distr(distr)}"
            if self.sampling_design == ChRDoE.SamplingDesign.FULL_FACTORIAL and ChRDoE._needs_factorial_level_spacing(distr):
                line += f"  [{self.factorial_level_spacing.get(name, '?')} steps]"
            print(line)

        if self.sampling_design == ChRDoE.SamplingDesign.FULL_FACTORIAL:
            total = 1
            for name, d in self.param_sample_space.items():
                ray_tune_disr_type, sampler = ChRDoE._get_distr_info(d)
                if ChRDoE._needs_factorial_level_spacing(d):
                    total *= self.factorial_level_spacing.get(name, 1)
                elif isinstance(d, ChR_Distr._ChRDistrDict):
                    total *= len(d['grid_search'])
                elif ray_tune_disr_type == "Categorical":
                    total *= len(d.categories)
                elif ray_tune_disr_type == "Integer":
                    total *= (d.upper - d.lower)
            print(f"  4. total configs      : {total} (full factorial)")
        else:
            print(f"  4. num_trials        : {self.num_trials}")

        print(f"  5. resources/config        : {self.resources_per_trial}")
        print(f"  6. concurrency             : auto (floor of detected resources / resources_per_trial)")
        print(f"  7. FLAG_log_to_file        : {self.FLAG_log_to_file}")
        print(f"  8. results_prefix          : {self.results_prefix}")
        if self.FLAG_auto_run:
            print(f"  9. FLAG_auto_run          : True -- concurrency is auto detected")
        else:
            print(f"  9. FLAG_auto_run          : False -- concurrency must be set manually before the backend is ran, see info() for help.")

        print("************************************************************")


    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _resolve_concurrency(self) -> int:
        #derive how many trials to run at once from the resources Ray was granted
        #and the per-trial reservation. concurrency = floor(detected / per-trial)
        #for whichever of CPU / GPU runs out first (GPU dropped when gpu == 0).
        #NOTE: must be called AFTER ray.init() so cluster_resources() is accurate.
        detected = ray.cluster_resources()
        detected_cpus = int(detected.get("CPU", 0))
        detected_gpus = int(detected.get("GPU", 0))

        cpu_per_trial = self.resources_per_trial["cpu"]
        gpu_per_trial = self.resources_per_trial["gpu"]

        cap_by_cpu = detected_cpus // cpu_per_trial

        if gpu_per_trial > 0:
            cap_by_gpu = detected_gpus // gpu_per_trial
            concurrency = min(cap_by_cpu, cap_by_gpu)
        else:
            #CPU-only trials: GPUs do not bind
            concurrency = cap_by_cpu

        #never drop below 1 (e.g. a trial that reserves more than the node has)
        concurrency = max(concurrency, 1)

        print(f"  Auto-concurrency: running {concurrency} trials at once "
              f"(detected {detected_cpus} CPU, {detected_gpus} GPU; "
              f"per trial {self.resources_per_trial}).")

        return concurrency

    @staticmethod
    def _get_distr_info(d) -> tuple[str, str]:
        if isinstance(d, ChR_Distr._ChRDistrDict):
            return "GridSearch", None
        ray_tune_disr_type   = type(d).__name__
        sampler = type(d.sampler).__name__.strip("_") if hasattr(d, "sampler") else None
        return ray_tune_disr_type, sampler

    @staticmethod
    def _is_factorial_compatible(d) -> bool:
        if isinstance(d, ChR_Distr._ChRDistrDict):
            return True
        ray_tune_disr_type, sampler = ChRDoE._get_distr_info(d)
        if ray_tune_disr_type == "Categorical":
            return True
        if ray_tune_disr_type == "Integer" and sampler == "Uniform":
            return True
        if ray_tune_disr_type == "Float" and sampler in ("Uniform", "LogUniform"):
            return True
        return False

    @staticmethod
    def _needs_factorial_level_spacing(d) -> bool:
        if isinstance(d, ChR_Distr._ChRDistrDict):
            return False
        ray_tune_disr_type, sampler = ChRDoE._get_distr_info(d)
        if ray_tune_disr_type == "Categorical":
            return False
        if ray_tune_disr_type == "Integer":
            return False
        return True

    def _generate_configs(self) -> list[dict]:
        if self.sampling_design == ChRDoE.SamplingDesign.FULL_FACTORIAL:
            return self._generate_factorial_configs()
        elif self.sampling_design == ChRDoE.SamplingDesign.LATIN_HYPERCUBE:
            return self._generate_lhs_configs()
        elif self.sampling_design == ChRDoE.SamplingDesign.SOBOL:
            return self._generate_sobol_configs()

    def _generate_factorial_configs(self) -> list[dict]:
        param_values = {}
        for name, d in self.param_sample_space.items():
            ray_tune_disr_type, sampler = ChRDoE._get_distr_info(d)
            if isinstance(d, ChR_Distr._ChRDistrDict):
                param_values[name] = d['grid_search']
            elif ray_tune_disr_type == "Categorical":
                param_values[name] = list(d.categories)
            elif ray_tune_disr_type == "Integer" and sampler == "Uniform":
                param_values[name] = list(range(d.lower, d.upper))
            elif ray_tune_disr_type == "Float" and sampler == "Uniform":
                param_values[name] = np.linspace(d.lower, d.upper, self.factorial_level_spacing[name]).tolist()
            elif ray_tune_disr_type == "Float" and sampler == "LogUniform":
                param_values[name] = np.logspace(np.log10(d.lower), np.log10(d.upper), self.factorial_level_spacing[name]).tolist()

        keys = list(param_values.keys())
        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*param_values.values())
        ]

    def _generate_lhs_configs(self) -> list[dict]:
        from scipy.stats import qmc
        names   = list(self.param_sample_space.keys())
        sampler = qmc.LatinHypercube(d=len(names), seed=42)
        samples = sampler.random(n=self.num_trials)
        return self._transform_unit_samples(names, samples)

    def _generate_sobol_configs(self) -> list[dict]:
        from scipy.stats import qmc
        import math
        names  = list(self.param_sample_space.keys())
        n_pow2 = 2 ** math.ceil(math.log2(self.num_trials))
        if n_pow2 != self.num_trials:
            print(f"  NOTE: Sobol requires power-of-2 samples. Rounding {self.num_trials} up to {n_pow2}.")
            self.num_trials = n_pow2
        sampler = qmc.Sobol(d=len(names), scramble=True, seed=42)
        samples = sampler.random(n=self.num_trials)
        return self._transform_unit_samples(names, samples)

    def _transform_unit_samples(self, names: list, samples: np.ndarray) -> list[dict]:
        configs = []
        for i in range(len(samples)):
            config = {}
            for j, name in enumerate(names):
                d = self.param_sample_space[name]
                u = samples[i, j]
                ray_tune_disr_type, sampler = ChRDoE._get_distr_info(d)

                if ray_tune_disr_type == "Float" and sampler == "Uniform":
                    config[name] = d.lower + u * (d.upper - d.lower)
                elif ray_tune_disr_type == "Float" and sampler == "LogUniform":
                    config[name] = d.lower * (d.upper / d.lower) ** u
                elif ray_tune_disr_type == "Integer" and sampler == "Uniform":
                    config[name] = int(d.lower + u * (d.upper - d.lower))
                elif ray_tune_disr_type == "Categorical":
                    idx = min(int(u * len(d.categories)), len(d.categories) - 1)
                    config[name] = d.categories[idx]
                elif isinstance(d, ChR_Distr._ChRDistrDict):
                    values = d['grid_search']
                    idx = min(int(u * len(values)), len(values) - 1)
                    config[name] = values[idx]

            configs.append(config)
        return configs

    @staticmethod
    def _execute_trial(simulate_fn, config):
        # One Ray task == one DoE configuration. If the trial reserved a GPU,
        # Ray decides which one it owns and we pin Chrono to that device. For
        # CPU-only trials (resources_per_trial gpu=0) no GPU is assigned, so we
        # skip pinning and just run the simulation.
        import os

        gpu_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])

        if gpu_ids:
            gpu_id = str(gpu_ids[0])

            # Match the robust-dispersion workflow: detect ROCm vs CUDA if torch exists.
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


    # -------------------------------------------------------------------------
    # PERSISTENCE + RESUME
    # -------------------------------------------------------------------------

    @staticmethod
    def _config_hash(config) -> str:
        #stable id for a config = hash of its parameter values (sorted keys so the
        #hash is order-independent; default=str handles numpy floats / categoricals).
        #NOTE: keyed on VALUES, so two configs with identical params share a hash.
        import hashlib, json
        blob = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _load_done_hashes(self) -> set:
        #gather config hashes already logged (successes AND failures) so restart can
        #skip them. missing files -> empty set (fresh run).
        import os, csv, json
        done = set()

        if os.path.exists(self._success_log_path):
            with open(self._success_log_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    h = row.get("config_hash")
                    if h:
                        done.add(h)

        if os.path.exists(self._failure_log_path):
            with open(self._failure_log_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line)["config_hash"])
                    except (json.JSONDecodeError, KeyError):
                        continue

        return done

    def _log_success(self, config_hash, trial_index, config) -> None:
        #append one row per completed trial; write the header once on file creation.
        #open/append/close per write so each record is flushed before the next trial.
        import os, csv
        from datetime import datetime
        names = list(self.param_sample_space.keys())
        write_header = not os.path.exists(self._success_log_path)
        with open(self._success_log_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(["timestamp", "trial_index", "config_hash"] + names)
            writer.writerow(
                [datetime.now().isoformat(), trial_index, config_hash]
                + [config.get(n) for n in names]
            )

    def _log_failure(self, config_hash, trial_index, config, error) -> None:
        #append one JSON record per failed trial (config values stringified so numpy
        #types serialize cleanly).
        import json
        from datetime import datetime
        record = {
            "timestamp":   datetime.now().isoformat(),
            "trial_index": trial_index,
            "config_hash": config_hash,
            "error_type":  type(error).__name__,
            "error":       str(error),
            "config":      {k: str(v) for k, v in config.items()},
        }
        with open(self._failure_log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


    # -------------------------------------------------------------------------
    # <3 METHODS IN USER INTERFACE
    # -------------------------------------------------------------------------


    def set_resources_per_trial(self, cpu: int, gpu: int) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set resources_per_trial after (auto-)run has started")
        self.resources_per_trial = {"cpu": cpu, "gpu": gpu}

    def set_FLAG_log_to_file(self, flag: bool) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set FLAG_log_to_file after (auto-)run has started")
        self.FLAG_log_to_file = flag

    def _build(self) -> None:
        if self._configs is not None and self.FLAG_auto_run:
            raise ValueError("Cannot rebuild since configs have already been generated and (auto-)run has started.")
        self._configs = self._generate_configs()
        print(f"  Generated {len(self._configs)} configurations.")

    def run(self) -> None:
        if self._configs is None:
            raise ValueError("Configs have not been generated yet. Call _build() first.")

        import os
        import sys
        import logging
        from datetime import datetime

        # suppress Ray console output
        os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

        for name in ["ray", "ray.tune", "ray.tune.registry", "ray._private"]:
            logging.getLogger(name).setLevel(logging.ERROR)
            logging.getLogger(name).propagate = False

        # redirect Ray output to a timestamped file only if explicitly requested.
        # keep references to the originals + file so they can be restored/closed
        # in the finally block once the run completes.
        log_file        = None
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        if self.FLAG_log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path  = os.path.join(os.getcwd(), f"chr_doe_log_{timestamp}.txt")
            logging.basicConfig(filename=log_path, level=logging.ERROR)

            #tee: write to BOTH the original console stream and the file, and flush
            #every write so output reaches disk even if the process is interrupted.
            class _TeeStream:
                def __init__(self, console, file):
                    self.console = console
                    self.file = file
                def write(self, msg):
                    self.console.write(msg)
                    self.file.write(msg)
                    self.file.flush()
                def flush(self):
                    self.console.flush()
                    self.file.flush()
                def fileno(self):
                    return self.file.fileno()

            log_file = open(log_path, "w")
            sys.stdout = _TeeStream(original_stdout, log_file)
            sys.stderr = _TeeStream(original_stderr, log_file)

        try:
            if not ray.is_initialized():
                # let Ray detect the resources exposed by the scheduler/partition;
                # concurrency is then derived from those resources and
                # resources_per_trial via _resolve_concurrency().
                ray.init(
                    logging_level=logging.ERROR,
                    log_to_driver=False,
                    configure_logging=False,
                )

            self.FLAG_auto_run = True

            # derive how many trials to run at once from detected resources
            concurrency = self._resolve_concurrency()

            # wrap _execute_trial as a Ray remote function, reserving resources per trial
            remote_fn = ray.remote(max_calls=1, max_retries=0)(ChRDoE._execute_trial).options(
                num_cpus=self.resources_per_trial["cpu"],
                num_gpus=self.resources_per_trial["gpu"],
            )

            # resume: skip any config already logged (success OR failure) on a previous
            # run, matched by a hash of its parameter values. keep the ORIGINAL index so
            # console/log records line up across runs.
            done_hashes = self._load_done_hashes()
            pending = [
                (i, c) for i, c in enumerate(self._configs)
                if ChRDoE._config_hash(c) not in done_hashes
            ]
            n_skipped = len(self._configs) - len(pending)
            if n_skipped:
                print(f"  Resuming: {n_skipped} configs already logged, {len(pending)} remaining.")

            # Submit trials in a sliding window so at most `concurrency` are in
            # flight at once. .remote() returns immediately with a future; each
            # future is paired with its original index + config so results can be
            # logged as they complete. Resource reservations (resources_per_trial)
            # further bound how many actually run in parallel on the node.
            n_ok = 0
            n_failed = 0

            pending_iter = iter(pending)
            future_meta = {}

            # prime the window up to the resolved concurrency
            for (i, c) in itertools.islice(pending_iter, concurrency):
                future_meta[remote_fn.remote(self.simulate_fn, c)] = (i, c)

            # process completions in finish order, back-filling the window with the
            # next queued config each time a trial finishes.
            while future_meta:
                ready, _ = ray.wait(list(future_meta.keys()), num_returns=1)
                f = ready[0]
                i, c = future_meta.pop(f)
                h = ChRDoE._config_hash(c)

                try:
                    ray.get(f)
                    self._log_success(h, i, c)
                    n_ok += 1
                except Exception as e:
                    self._log_failure(h, i, c, e)
                    n_failed += 1
                    print(f"  TRIAL {i} FAILED ({type(e).__name__}): {e}", flush=True)

                # back-fill: submit the next pending config, if any remain
                nxt = next(pending_iter, None)
                if nxt is not None:
                    ni, nc = nxt
                    future_meta[remote_fn.remote(self.simulate_fn, nc)] = (ni, nc)

            print(f"  DoE complete: {n_ok} succeeded, {n_failed} failed, "
                  f"{n_skipped} skipped (already logged). "
                  f"Results -> {self._success_log_path}")

        finally:
            # restore console streams and flush/close the log file so its
            # contents are guaranteed to reach disk.
            if self.FLAG_log_to_file:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                if log_file is not None:
                    log_file.flush()
                    log_file.close()