from collections.abc import Callable
import textwrap

from ChronoRay.ChR_ChronoRay import ChR_ChronoRay
from ChronoRay.ChR_Config import ChR_Distr, ChR_SearchAlg

class ChROpt:

    @staticmethod
    def info() -> None:
        print("========================================================")
        print("ChROpt - ChronoRay (PyChrono + Ray) Optimization Workflow")
        print("========================================================")
        print(textwrap.dedent("""
            DESCRIPTION:
                ChROpt is a general-purpose optimization framework for PyChrono simulations.
                It searches for the parameter configuration that best optimizes a
                user-defined objective function using a pluggable search algorithm.

                Unlike ChRBayesOpt, the search algorithm is not fixed — the user
                can select any available Ray Tune search algorithm.

                For a simpler interface with Bayesian optimization hardcoded,
                see ChRBayesOpt.

            --------------------------------------------------------
            CONSTRUCTOR ARGUMENTS:
            --------------------------------------------------------

                simulate_fn (Callable) [REQUIRED]
                    A PyChrono simulation function.
                    INPUT:   config (dict) — parameter names and their sampled values.
                    RETURNS: output (dict) — simulation output values.
                    NOTE:    no visualization (e.g. Irrlicht, VSG) should take place.

                objective_fn (Callable) [REQUIRED]
                    A user-defined function that scores each simulation run.
                    INPUT:   output (dict) — return value of simulate_fn.
                    RETURNS: score (float) — scalar value to minimize or maximize.
                    Example: lambda output: output["settling_time"]

                param_sample_space (dict[str, ChR_Distr]) [REQUIRED]
                    Parameters to tune and their search distributions.
                    Keys   : parameter names (must match keys expected by simulate_fn).
                    Values : ChR_Distr distributions defining the search space.
                    NOTE:    run ChROpt.ChR_Distr.info() for available distributions.

                mode (str) [REQUIRED]
                    Whether to minimize or maximize the objective.
                    Options: "min" or "max"

                total_trials (int) [OPTIONAL, default: 10]
                    Total number of simulation trials to run.

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
                    The optimization runs immediately on construction.
                    No further configuration is possible.

                    Example:
                        opt = ChROpt(
                            simulate_fn        = my_sim,
                            objective_fn       = lambda output: output["settling_time"],
                            param_sample_space = {
                                "k": ChROpt.ChR_Distr.uniform(0, 10),
                                "m": ChROpt.ChR_Distr.loguniform(1e-2, 1e2),
                            },
                            mode = "min"
                        )

                MODE 2 — FLAG_auto_run=False
                    Construction stops after validation and config report.
                    The user can then configure optional settings via the
                    setters listed below before manually building and running.

                    Example:
                        opt = ChROpt(
                            simulate_fn        = my_sim,
                            objective_fn       = lambda output: output["settling_time"],
                            param_sample_space = {...},
                            mode               = "min",
                            FLAG_auto_run      = False
                        )
                        opt.set_search_alg(ChROpt.ChR_SearchAlg.OPTUNA)
                        opt.set_search_alg_config({"n_startup_trials": 10})
                        opt.set_max_concurrent_trials(2)
                        opt.set_resources_per_trial(cpu=4, gpu=0)
                        opt.set_FLAG_log_to_file(True)
                        opt._build_backend()
                        opt.run()

            --------------------------------------------------------
            OPTIONAL SETTERS (MODE 2 only):
            --------------------------------------------------------

                set_search_alg(alg: ChR_SearchAlg)
                    Set the search algorithm. Default: ChR_SearchAlg.BAYESOPT
                    NOTE: run ChROpt.ChR_SearchAlg.info() for available options.

                set_search_alg_config(cfg: dict)
                    Pass extra kwargs to the search algorithm constructor.
                    See Ray Tune documentation for available options.

                set_max_concurrent_trials(n: int)
                    Set the maximum number of trials running simultaneously.
                    Default: 4

                set_resources_per_trial(cpu: int, gpu: int)
                    Set the CPU/GPU resources allocated per trial.
                    Default: cpu=1, gpu=0

                set_FLAG_log_to_file(flag: bool)
                    Toggle whether Ray output is redirected to a file. Default: False
        """))
        print("========================================================")

    ChR_Distr = ChR_Distr
    ChR_SearchAlg = ChR_SearchAlg

    def __init__(
                self,
                simulate_fn: Callable,
                objective_fn: Callable,
                param_sample_space: dict[str, ChR_Distr],
                mode: str,
                total_trials: int = 10,
                FLAG_log_to_file: bool = False,
                FLAG_auto_run: bool = True) -> None:

        #1. mandatory parameters
        self.simulate_fn = simulate_fn
        self.objective_fn = objective_fn
        self.param_sample_space = param_sample_space
        self.mode = mode
        self.total_trials = total_trials

        #2. optional parameters
        self.max_concurrent_trials = 4
        self.resources_per_trial = {"cpu": 1, "gpu": 0}
        self.search_alg = ChR_SearchAlg.BAYESOPT
        self.search_alg_config = {}
        self.FLAG_log_to_file = FLAG_log_to_file

        self._validate_inputs()
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
        print("run ChROpt.info() for information on how to use this workflow.")
        print("************************************************************")

        #1. simulate_fn check
        if not callable(self.simulate_fn):
            raise ValueError("simulate_fn must be callable")

        #2. objective_fn check
        if not callable(self.objective_fn):
            raise ValueError("objective_fn must be callable")

        #3. parameter configuration checks
        if not isinstance(self.param_sample_space, dict):
            raise TypeError("param_sample_space must be a dict")

        if not all(isinstance(k, str) for k in self.param_sample_space.keys()):
            raise TypeError("param_sample_space keys must all be strings")

        if not all(getattr(v, "FLAG_is_chr_distr", False) for v in self.param_sample_space.values()):
            raise TypeError("param_sample_space values must all be ChR_Distr distributions. Use ChROpt.ChR_Distr.info() for available options and additional info.")

        #4. mode check
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")

    def _report_config(self) -> None:

        print("************************************************************")
        print("========================================================")
        print("ChROpt (\"ChronoRay Optimization Workflow\") configuration:")
        print("========================================================")
        print(f"  1. simulate_fn        : {self.simulate_fn.__name__}")
        print(f"  2. objective_fn       : {self.objective_fn.__name__}")
        print(f"  3. param_sample_space :")

        for name, distr in self.param_sample_space.items():
            print(f"       {name} : {ChR_Distr._format_distr(distr)}")

        print(f"  4. mode               : {self.mode}")
        print(f"  5. total_trials       : {self.total_trials}")
        print(f"  6. search_algorithm   : {self.search_alg.name}")
        print(f"  7. FLAG_log_to_file        : {self.FLAG_log_to_file}")

        if self.search_alg == ChR_SearchAlg.BAYESOPT:
            print(f"     NOTE: default search algorithm in use.")
            print(f"     -> run ChROpt.ChR_SearchAlg.info() for available search algorithms and additional info.")

        print("************************************************************")

    #<3 METHODS IN USER INTERFACE
    def set_search_alg(self, search_alg: ChR_SearchAlg) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set search_alg after (auto-)run has started")
        self.search_alg = search_alg

    def set_search_alg_config(self, search_alg_config: dict) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set search_alg_config after (auto-)run has started")
        self.search_alg_config = search_alg_config

    def set_max_concurrent_trials(self, max_concurrent_trials: int) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set max_concurrent_trials after (auto-)run has started")
        self.max_concurrent_trials = max_concurrent_trials

    def set_resources_per_trial(self, cpu: int, gpu: int) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set resources_per_trial after (auto-)run has started")
        self.resources_per_trial = {"cpu": cpu, "gpu": gpu}

    def set_FLAG_log_to_file(self, flag: bool) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set FLAG_log_to_file after (auto-)run has started")
        self.FLAG_log_to_file = flag

    def _build_backend(self) -> ChR_ChronoRay:

        if self.backend is not None and self.FLAG_auto_run:
            raise ValueError("Cannot rebuild since backend has already been built and (auto-)run has started.")

        return ChR_ChronoRay(
            simulate_fn=self.simulate_fn,
            objective_fn=self.objective_fn,
            param_space=self.param_sample_space,
            resources_per_trial=self.resources_per_trial,
            num_trials=self.total_trials,
            max_concurrent_trials=self.max_concurrent_trials,
            mode=self.mode,
            search_algorithm=self.search_alg,
            search_kwargs=self.search_alg_config,
        )

    def run(self) -> None:
        if self.backend is None:
            raise ValueError("backend has not been built yet. Call _build_backend() first.")

        self.FLAG_auto_run = True
        self.backend.run(FLAG_log_to_file=self.FLAG_log_to_file)