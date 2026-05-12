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

                Unlike ChRParamEst and ChRBayesOpt, there is no objective function
                or scoring — the user's simulate_fn is responsible for logging
                any outputs of interest internally.

            --------------------------------------------------------
            CONSTRUCTOR ARGUMENTS:
            --------------------------------------------------------

                simulate_fn (Callable) [REQUIRED]
                    A PyChrono simulation function.
                    INPUT:   config (dict) — parameter names and their sampled values.
                    RETURNS: None — logging must be handled inside simulate_fn.
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
                        ChRDoE.SamplingDesign.FULL_FACTORIAL  — all combinations of discrete parameter values
                        ChRDoE.SamplingDesign.LATIN_HYPERCUBE — space-filling Latin Hypercube Sampling (LHS)
                        ChRDoE.SamplingDesign.SOBOL           — space-filling Sobol sequence

                num_trials (int) [OPTIONAL, default: 10]
                    Number of total configurations to generate.
                    Used by LATIN_HYPERCUBE and SOBOL.
                    Ignored for FULL_FACTORIAL (determined by factorial_level_spacing).
                    NOTE:    SOBOL requires a power-of-2 value — will auto round up.

                max_concurrent_trials (int) [OPTIONAL, default: 2]
                    Maximum number of trials running simultaneously.

                factorial_level_spacing (dict[str, int]) [OPTIONAL, default: None]
                    Number of discrete levels per parameter for FULL_FACTORIAL.
                    Each continuous parameter is evenly divided into this many values
                    across its range. Required for uniform and loguniform distributions.
                    Not required for randint, choice, or grid_search — those are
                    already discrete.
                    Example: {"k": 5, "m": 4}
                        "k" → 5 evenly spaced values across its range
                        "m" → 4 log-spaced values across its range

                FLAG_log_to_file (bool) [OPTIONAL, default: False]
                    If True, Ray's console output is redirected to a timestamped
                    .txt file in the current working directory. User prints are also 
                    redirected to .txt file. Default keeps all output in the console.

                FLAG_auto_run (bool) [OPTIONAL, default: True]
                    Controls the operating mode. See OPERATING MODES below.

            --------------------------------------------------------
            OPERATING MODES:
            --------------------------------------------------------

                MODE 1 — FLAG_auto_run=True (default)
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

                MODE 2 — FLAG_auto_run=False
                    Construction stops after validation and config report.
                    The user can configure optional settings before running.

                    Example:
                        doe = ChRDoE(..., FLAG_auto_run=False)
                        doe.set_resources_per_trial(cpu=2, gpu=0)
                        doe.set_FLAG_log_to_file(True)
                        doe._build()
                        doe.run()

            --------------------------------------------------------
            OPTIONAL SETTERS (MODE 2 only):
            --------------------------------------------------------

                set_resources_per_trial(cpu: int, gpu: int)
                    Set the CPU/GPU resources allocated per trial.
                    Default: cpu=1, gpu=0

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
                max_concurrent_trials: int = 2,
                factorial_level_spacing: dict[str, int] = None,
                FLAG_log_to_file: bool = False,
                FLAG_auto_run: bool = True) -> None:

        #1. mandatory parameters
        self.simulate_fn                 = simulate_fn
        self.param_sample_space          = param_sample_space
        self.sampling_design             = sampling_design
        self.num_trials                  = num_trials
        self.max_concurrent_trials       = max_concurrent_trials
        self.factorial_level_spacing     = factorial_level_spacing or {}

        #2. optional parameters
        self.resources_per_trial   = {"cpu": 1, "gpu": 0}
        self.FLAG_log_to_file           = FLAG_log_to_file

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

        print(f"  5. FLAG_log_to_file        : {self.FLAG_log_to_file}")

        print("************************************************************")


    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

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
        simulate_fn(config)


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

        # redirect Ray output to a timestamped file only if explicitly requested
        if self.FLAG_log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path  = os.path.join(os.getcwd(), f"chr_doe_log_{timestamp}.txt")
            logging.basicConfig(filename=log_path, level=logging.ERROR)

            class _TeeStream:
                def __init__(self, file):
                    self.file = file
                def write(self, msg):
                    self.file.write(msg)
                def flush(self):
                    self.file.flush()
                def fileno(self):
                    return self.file.fileno()

            log_file = open(log_path, "w")
            sys.stdout = _TeeStream(log_file)
            sys.stderr = _TeeStream(log_file)

        if not ray.is_initialized():
            ray.init(
                logging_level=logging.ERROR,
                log_to_driver=False,
                configure_logging=False
            )

        self.FLAG_auto_run = True

        # wrap _execute_trial as a Ray remote function, reserving resources per trial
        remote_fn = ray.remote(ChRDoE._execute_trial).options(
            num_cpus=self.resources_per_trial["cpu"],
            num_gpus=self.resources_per_trial["gpu"]
        )

        # fire off all trials in parallel — .remote() returns immediately with a future
        # Ray queues and runs them concurrently up to the available resource limit
        for i in range(0, len(self._configs), self.max_concurrent_trials):
            batch = self._configs[i:i + self.max_concurrent_trials]
            futures = [remote_fn.remote(self.simulate_fn, config) for config in batch]
            ray.get(futures)