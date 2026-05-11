from enum import Enum
from collections.abc import Callable
import textwrap

from ChronoRay.ChR_ChronoRay import ChR_ChronoRay
from ChronoRay.ChR_Config import ChR_Distr, ChR_SearchAlg

class ChRParamEst:

    @staticmethod
    def info() -> None:
        print("========================================================")
        print("ChRParamEst - ChronoRay (PyChrono + Ray) Parameter Estimation Workflow")
        print("========================================================")
        print(textwrap.dedent("""
            DESCRIPTION:
                ChRParamEst is a parameter estimation framework for PyChrono simulations.
                It automatically tunes simulation parameters to match a set of target
                output values using Ray Tune as the search backend.

            --------------------------------------------------------
            CONSTRUCTOR ARGUMENTS:
            --------------------------------------------------------

                simulate_fn (Callable) [REQUIRED]
                    A PyChrono simulation function.
                    INPUT:   config (dict) — parameter names and their sampled values.
                    RETURNS: output (dict) — simulation output values.
                    NOTE:    output keys must match target_sim_outputs keys exactly.
                    NOTE:    no visualization (e.g. Irrlicht, VSG) should take place.

                est_rule (ChRParamEst.EstRule) [REQUIRED]
                    The estimator rule used to score each simulation run.
                    Available options:
                        ChRParamEst.EstRule.LS    — least squares
                        ChRParamEst.EstRule.MLE   — maximum likelihood      
                        ChRParamEst.EstRule.MAP   — maximum a posteriori    
                        ChRParamEst.EstRule.BAYES — bayesian               

                param_sample_space (dict[str, ChR_Distr]) [REQUIRED]
                    Parameters to tune and their search distributions.
                    Keys   : parameter names (must match keys expected by simulate_fn).
                    Values : ChR_Distr distributions defining the search space.
                    NOTE:    run ChRParamEst.ChR_Distr.info() for available distributions.

                target_sim_outputs (dict[str, float]) [REQUIRED]
                    Target output values to compare against simulation outputs.
                    Keys   : output names (must match keys returned by simulate_fn).
                    Values : target float values.

                est_config (dict) [OPTIONAL, default: None]
                    Additional configuration for the estimator rule.
                    NOTE:    not yet in use — reserved for future estimator rules.

                total_trials (int) [OPTIONAL, default: 10]
                    Total number of simulation trials to run.

                FLAG_auto_run (bool) [OPTIONAL, default: True]
                    Controls the operating mode. See OPERATING MODES below.

            --------------------------------------------------------
            OPERATING MODES:
            --------------------------------------------------------

                MODE 1 — FLAG_auto_run=True (default)
                    The estimation runs immediately on construction.
                    No further configuration is possible.

                    Example:
                        est = ChRParamEst(
                            simulate_fn        = my_sim,
                            est_rule           = ChRParamEst.EstRule.LS,
                            param_sample_space = {
                                "k": ChRParamEst.ChR_Distr.uniform(0, 10),
                                "m": ChRParamEst.ChR_Distr.loguniform(1e-2, 1e2),
                            },
                            target_sim_outputs = {
                                "output_1": 5.0,
                                "output_2": 3.2,
                            }
                        )

                MODE 2 — FLAG_auto_run=False
                    Construction stops after validation and config report.
                    The user can then configure optional settings via the
                    setters listed below before manually building and running.

                    Example:
                        est = ChRParamEst(
                            simulate_fn        = my_sim,
                            est_rule           = ChRParamEst.EstRule.LS,
                            param_sample_space = {...},
                            target_sim_outputs = {...},
                            FLAG_auto_run      = False
                        )
                        est.set_search_alg(ChR_SearchAlg.OPTUNA)
                        est.set_max_concurrent_trials(2)
                        est.set_resources_per_trial(cpu=4, gpu=0)
                        est.set_search_alg_config({"n_startup_trials": 10})
                        est._build_chrono_ray()
                        est.run()

            --------------------------------------------------------
            OPTIONAL SETTERS (MODE 2 only):
            --------------------------------------------------------

                set_search_alg(alg: ChR_SearchAlg)
                    Set the search algorithm. Default: ChR_SearchAlg.BAYESOPT

                set_search_alg_config(cfg: dict)
                    Pass extra kwargs to the search algorithm constructor.
                    See Ray Tune documentation for available options.

                set_max_concurrent_trials(n: int)
                    Set the maximum number of trials running simultaneously.
                    Default: 4

                set_resources_per_trial(cpu: int, gpu: int)
                    Set the CPU/GPU resources allocated per trial.
                    Default: cpu=1, gpu=0
        """))
        print("========================================================")

    ChR_Distr = ChR_Distr
    ChR_SearchAlg = ChR_SearchAlg

    class EstRule(Enum):
        LS = "least squares"
        MLE = "maximum likelihood"
        MAP = "maximum a posteriori"
        BAYES = "bayesian"

    def __init__(
                self,
                simulate_fn: Callable,
                est_rule: EstRule,
                param_sample_space: dict[str, ChR_Distr],
                target_sim_outputs: dict[str, float],
                est_config: dict = None, 
                total_trials: int = 10,
                FLAG_auto_run: bool = True) -> None:

        #1. manditory parameters 
        self.simulate_fn = simulate_fn
        self.est_rule = est_rule
        self.param_sample_space = param_sample_space
        self.target_sim_outputs = target_sim_outputs
        self.est_config = est_config
        self.total_trials = total_trials

        #2. optional parameters 
        self.max_concurrent_trials = 4 
        self.resources_per_trial = {"cpu": 1, "gpu": 0}
        self.search_alg = ChR_SearchAlg.BAYESOPT
        self.search_alg_config = {}

        self._validate_inputs()
        self._report_config()

        #3. immediately run if FLAG_auto_run is True
        self.FLAG_auto_run = FLAG_auto_run
        self.chrono_ray = None
        if FLAG_auto_run:
            self.chrono_ray = self._build_chrono_ray()
            self.run() 
        

    def _validate_inputs(self) -> None:

        print("************************************************************")
        print("Validating inputs...")
        print("run ChRParamEst.info() for information on how to use this estimator.")
        print("************************************************************")
        
        #1. simulate_fn check
        if not callable(self.simulate_fn):
            raise ValueError("simulate_fn must be callable")

        #2. est_rule check
        if not isinstance(self.est_rule, ChRParamEst.EstRule):
            raise ValueError("est_rule must be a valid EstRule")

        #3. parameter configuration checks
        if not isinstance(self.param_sample_space, dict):
            raise TypeError("param_sample_space must be a dict")

        if not all(isinstance(k, str) for k in self.param_sample_space.keys()):
            raise TypeError("param_sample_space keys must all be strings")

        if not all(getattr(v, "FLAG_is_chr_distr", False) for v in self.param_sample_space.values()):
            raise TypeError("param_sample_space values must all be ChR_Distr distributions. Use ChRParamEst.ChR_Distr.info() for available options and additional info.")

        #4. simulation target outputs check 
        if not isinstance(self.target_sim_outputs, dict):
            raise TypeError("target_sim_outputs must be a dict")

        if not all(isinstance(k, str) for k in self.target_sim_outputs.keys()):
            raise TypeError("target_sim_outputs keys must all be strings")

        if not all(isinstance(v, float) for v in self.target_sim_outputs.values()):
            raise TypeError("target_sim_outputs values must all be floats")

        #5. estimator configuration check 
        if not isinstance(self.est_config, dict):
            #TODO more thorough check after all estimator rules have been implemented 
            pass 
        

    def _report_config(self) -> None:

        print("************************************************************")
        print("========================================================")
        print("ChRParamEst (\"ChronoRay Parameter Estimation Workflow\") configuration:")
        print("========================================================")
        print(f"  1. simulate_fn        : {self.simulate_fn.__name__}")
        print(f"  2. est_rule           : {self.est_rule.value}")
        print(f"  3. param_sample_space :")

        for name, distr in self.param_sample_space.items():
            print(f"       {name} : {ChR_Distr._format_distr(distr)}")

        print(f"  4. target_sim_outputs : {self.target_sim_outputs}")
        print(f"  5. est_config         : {self.est_config}")
        print(f"  6. search_algorithm   : {self.search_alg.name}")

        if self.search_alg == ChR_SearchAlg.BAYESOPT:
            print(f"     NOTE: default search algorithm in use.")
            print(f"     -> run ChRParamEst.ChR_SearchAlg.info() for available search algorithms and additional info.")

        print("************************************************************")

    def _get_est_fn(self) -> tuple[Callable, str]:
        target = self.target_sim_outputs

        if self.est_rule == ChRParamEst.EstRule.LS:
            def est_fn(sim_output: dict) -> float:
                missing = [k for k in target.keys() if k not in sim_output]
                if missing:
                    raise KeyError(
                        f"simulate_fn output is missing key(s): {missing}. "
                        f"Expected keys: {list(target.keys())}"
                        f"NOTE: keys must be identical in name (including case) and order."
                    )
                return sum(
                    (sim_output[k] - target[k]) ** 2
                    for k in target.keys()
                )
            return est_fn, "min"

        else:
            raise NotImplementedError(f"{self.est_rule.value} has not been implemented yet")



    #<3 METHODS IN USER INTERFACE 
    def set_max_concurrent_trials(self, max_concurrent_trials: int) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set max_concurrent_trials after (auto-)run has started")

        self.max_concurrent_trials = max_concurrent_trials

    def set_resources_per_trial(self, cpu: int, gpu: int) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set resources_per_trial after (auto-)run has started")

        self.resources_per_trial = {"cpu": cpu, "gpu": gpu}
    
    def set_search_alg(self, search_alg: ChR_SearchAlg) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set search_alg after (auto-)run has started")

        self.search_alg = search_alg

    def set_search_alg_config(self, search_alg_config: dict) -> None:
        if self.FLAG_auto_run:
            raise ValueError("Cannot set search_alg_config after (auto-)run has started")

        self.search_alg_config = search_alg_config

    def _build_chrono_ray(self) -> ChR_ChronoRay:

        if self.chrono_ray is not None and self.FLAG_auto_run:
            raise ValueError("Cannot rebuild since chrono_ray has already been built and (auto-)run has started.")

        est_fn, mode = self._get_est_fn()

        return ChR_ChronoRay(
            simulate_fn=self.simulate_fn,
            objective_fn=est_fn,
            param_space=self.param_sample_space,
            resources_per_trial=self.resources_per_trial,
            num_trials=self.total_trials,
            max_concurrent_trials=self.max_concurrent_trials,
            mode=mode,
            search_algorithm=self.search_alg,
            search_kwargs=self.search_alg_config,
        )

    def run(self) -> None:
        if self.chrono_ray is None:
            raise ValueError("chrono_ray has not been built yet. Call _build_chrono_ray() first.")

        self.FLAG_auto_run = True

        self.chrono_ray.run()