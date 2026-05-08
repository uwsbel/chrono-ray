"""
ChronoRay
=========
A Ray Tune wrapper for PyChrono (or any) simulation-based optimisation.

Key features
------------
- Pluggable search algorithm  (ChronoRaySearchAlgorithm enum)
- Pluggable trial scheduler   (ChronoRayTrialScheduler  enum)
- Optional multi-objective linear combination
- Automatic ConcurrencyLimiter for gradient-based / sequential searchers
"""

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.air import session
from ChronoRaySearchAlgorithms import ChronoRaySearchAlgorithms
from ChronoRayTrialSchedulers import ChronoRayTrialSchedulers


# Searchers that are inherently sequential – benefit from ConcurrencyLimiter
_SEQUENTIAL_SEARCHERS = {
    ChronoRaySearchAlgorithms.BAYESOPT,
    ChronoRaySearchAlgorithms.HYPEROPT,
    ChronoRaySearchAlgorithms.BOHB,
    ChronoRaySearchAlgorithms.HEBO,
    ChronoRaySearchAlgorithms.ZOOPT,
}

# Searchers whose constructors accept metric / mode
_METRIC_MODE_SEARCHERS = {
    ChronoRaySearchAlgorithms.BAYESOPT,
    ChronoRaySearchAlgorithms.OPTUNA,
    ChronoRaySearchAlgorithms.HYPEROPT,
    ChronoRaySearchAlgorithms.BOHB,
    ChronoRaySearchAlgorithms.AX,
    ChronoRaySearchAlgorithms.HEBO,
    ChronoRaySearchAlgorithms.NEVERGRAD,
    ChronoRaySearchAlgorithms.ZOOPT,
}


# =============================================================================
# <3 CHRONORAY CLASS
# =============================================================================

class ChronoRay:
    """
    ================================
    CONSTRUCTOR PARAMETERS
    ================================
    
    1. param_space : dict
        DESCRIPTION: 
        Simulation parameters and accompanying distributions for tuning, 
        e.g. {"k": ChronoRayDistributions.uniform(10, 100), "c": ChronoRayDistributions.loguniform(0.1, 10.0)}.

    2. simulate_fn : callable (i.e function)
        DESCRIPTION: 
        PyChrono code utilizing input to configure the simulation. 

        ASSUMPTIONS: 
        - No visualization (e.g Irrlicht, VSG) takes place. 

        INPUT(S):   config (dict) [configuration of the simulation parameters].
        RETURNS:    raw_output (any type) [raw output of the to be used by the evaluate_fn].

    3. evaluate_fn : callable (i.e function)
        DESCRIPTION: 
        User defined implementation of the objective function used for optimization.

        ASSUMPTIONS: 
        - The input is the return value of simulate_fn(). 
        - The output is a scalar float or a dictionary of keys of type string and values of type float.
            - See FLAG_multi_objective for more information on return type.
        
        INPUT(S):   raw_output (any type) [raw output of the simulate_fn() function].
        RETURNS:    value (float or dict) [value of the objective function].

    5. FLAG_multi_objective : bool
        DESCRIPTION: 
        Flag indicating whether the objective function to be calculated by the ChronoRay 
        instance is multi-objective or not. 
        True  -> evaluate_fn must return a dict; objectives are linearly combined using user defined weights.
        False -> evaluate_fn returns a plain scalar, or a dict containing a key corresponding to the `metric` parameter.

        ASSUMPTIONS: 
        - The default value is False.

    6. objectives_weights : list[float]
        DESCRIPTION: 
        [Multi-objective mode only]
        List of weights for each objective. 
        Weight for each objective.  combined = sum(w_i * value_i).

        ASSUMPTIONS: 
        - It must have the same length as the `objectives` parameter.

    7. objectives : list[str]
        DESCRIPTION: 
        [Multi-objective mode only]
        Keys in evaluate_fn's output dict to combine.
        Example: ["max_stress", "total_mass", "settling_time"]

        ASSUMPTIONS: 
        - It must have the same length as the `objectives_weights` parameter.

    8. resources_per_trial : dict [OPTIONAL]
        DESCRIPTION: 
        Dictionary specifying the resources required for each trial.
        e.g. {"cpu": 4, "gpu": 0}

        ASSUMPTIONS: 
        - If not specified, the default value of {"cpu": 1, "gpu": 0} will be used.
        - It must be a dictionary with keys of type string and values of type int.

    9. num_samples : int
        DESCRIPTION: 
        Total number of trials to run.

        ASSUMPTIONS: 
        - It must be greater than 0.
        - The default value is 10.

    10. max_concurrent_trials : int
        DESCRIPTION: 
        Maximum number of trials allowed to run simultaneously.
        For sequential searchers (BayesOpt, HyperOpt, BOHB, HEBO, ZOOpt)
        lower values (2-4) are strongly recommended so the optimizer can
        learn from completed trials before launching new ones.
        Random / Grid / Optuna / Ax / Nevergrad tolerate higher values.

        ASSUMPTIONS: 
        - It must be greater than 0.
        - The default value is 4.

    11. metric : str
        DESCRIPTION: 
        Name of the scalar reported to Ray as the optimisation target.
        In multi-objective mode this is also the name of the combined score.

        ASSUMPTIONS: 
        - The default value is "objective".

    12. mode : str
        DESCRIPTION: 
        Mode of the optimization.

        ASSUMPTIONS: 
        - It must be either "min" or "max".
        - The default value is "min".

    13. search_algorithm : ChronoRaySearchAlgorithm
        DESCRIPTION: 
        Which search algorithm to use.  

        ASSUMPTIONS: 
        - It must be a ChronoRaySearchAlgorithm enum member.
        - The default value is ChronoRaySearchAlgorithm.BAYESOPT.

    14. trial_scheduler : ChronoRayTrialScheduler
        DESCRIPTION: 
        Which trial scheduler to use. 
        The trial scheduler is used to determine the order in which the trials are run.

        ASSUMPTIONS: 
        - It must be a ChronoRayTrialScheduler enum member.
        - The default value is ChronoRayTrialScheduler.NONE.

    15. search_algorithm_kwargs : dict [OPTIONAL]
        DESCRIPTION: 
        Extra keyword arguments forwarded verbatim to the search algorithm constructor. 
        metric and mode are injected automatically where required.
        See Ray Tune documentation for more information on the available options.

        ASSUMPTIONS: 
        - The default value is None.

    16. trial_scheduler_kwargs : dict [OPTIONAL]
        DESCRIPTION: 
        Extra keyword arguments forwarded verbatim to the trial scheduler constructor.
        metric and mode are injected automatically where required.
        See Ray Tune documentation for more information on the available options.

        ASSUMPTIONS: 
        - The default value is None.
    """


    def __init__(
        self,
        simulate_fn,
        evaluate_fn,
        param_space,
        resources_per_trial=None,
        num_samples=10,
        max_concurrent_trials=4,
        metric="objective",
        mode="min",
        search_algorithm: ChronoRaySearchAlgorithm = ChronoRaySearchAlgorithms.BAYESOPT,
        search_kwargs: dict | None = None,
        trial_scheduler: ChronoRayTrialScheduler = ChronoRayTrialSchedulers.NONE,
        scheduler_kwargs: dict | None = None,
        objectives=None,
        objectives_weights=None,
        FLAG_multi_objective=False,
    ):
        self.simulate_fn           = simulate_fn
        self.evaluate_fn           = evaluate_fn
        self.param_space           = param_space
        self.resources_per_trial   = resources_per_trial or {"cpu": 1, "gpu": 0}
        self.num_samples           = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        self.metric                = metric
        self.mode                  = mode
        self.search_algorithm      = search_algorithm
        self.search_kwargs         = search_kwargs or {}
        self.trial_scheduler       = trial_scheduler
        self.scheduler_kwargs      = scheduler_kwargs or {}
        self.objectives            = objectives or []
        self.objectives_weights    = objectives_weights or []
        self.FLAG_multi_objective  = FLAG_multi_objective

        self._validate_inputs()
        self._report_config()

    def _validate_inputs(self):
        """
        DESCRIPTION: 
        Validates the inputs of the ChronoRay instance.

        ASSUMPTIONS: 
        - None 

        INPUT(S):   None
        RETURNS:    None
        THROWS:     ValueError / TypeError if inputs are not valid
        """
        if not callable(self.simulate_fn):
            raise ValueError("simulate_fn must be callable")
        if not callable(self.evaluate_fn):
            raise ValueError("evaluate_fn must be callable")
        if not isinstance(self.param_space, dict):
            raise ValueError("param_space must be of type dict")
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")

        if not isinstance(self.search_algorithm, SearchAlgorithm):
            raise TypeError(
                f"search_algorithm must be a SearchAlgorithm enum member, "
                f"got {type(self.search_algorithm).__name__!r}. "
                f"Valid choices: {[e.name for e in SearchAlgorithm]}"
            )
        if not isinstance(self.trial_scheduler, TrialScheduler):
            raise TypeError(
                f"trial_scheduler must be a TrialScheduler enum member, "
                f"got {type(self.trial_scheduler).__name__!r}. "
                f"Valid choices: {[e.name for e in TrialScheduler]}"
            )

        # BOHB searcher <-> BOHB scheduler must be paired together
        bohb_searcher = self.search_algorithm == SearchAlgorithm.BOHB
        bohb_sched    = self.trial_scheduler   == TrialScheduler.BOHB_SCHED
        if bohb_sched and not bohb_searcher:
            raise ValueError(
                "TrialScheduler.BOHB_SCHED requires SearchAlgorithm.BOHB. "
                f"You selected SearchAlgorithm.{self.search_algorithm.name}."
            )
        if bohb_searcher and not bohb_sched:
            raise ValueError(
                "SearchAlgorithm.BOHB requires TrialScheduler.BOHB_SCHED. "
                f"You selected TrialScheduler.{self.trial_scheduler.name}."
            )

        # PBT / PB2 require specific kwarg keys
        if self.trial_scheduler == TrialScheduler.PBT:
            if "hyperparam_mutations" not in self.scheduler_kwargs:
                raise ValueError(
                    "TrialScheduler.PBT requires 'hyperparam_mutations' in scheduler_kwargs. "
                    "Example: scheduler_kwargs={'hyperparam_mutations': {'lr': tune.loguniform(1e-4, 1e-1)}}"
                )
        if self.trial_scheduler == TrialScheduler.PB2:
            if "hyperparam_bounds" not in self.scheduler_kwargs:
                raise ValueError(
                    "TrialScheduler.PB2 requires 'hyperparam_bounds' in scheduler_kwargs. "
                    "Example: scheduler_kwargs={'hyperparam_bounds': {'lr': [1e-4, 1e-1]}}"
                )

        # Multi-objective checks
        if self.FLAG_multi_objective:
            if not isinstance(self.objectives, list) or len(self.objectives) == 0:
                raise ValueError(
                    "FLAG_multi_objective is True but 'objectives' is empty or not a list. "
                    "Provide a non-empty list of metric key strings."
                )
            if not isinstance(self.objectives_weights, list) or len(self.objectives_weights) == 0:
                raise ValueError(
                    "FLAG_multi_objective is True but 'objectives_weights' is empty or not a list. "
                    "Provide a non-empty list of numeric weights."
                )
            if len(self.objectives) != len(self.objectives_weights):
                raise ValueError(
                    f"Length mismatch: 'objectives' has {len(self.objectives)} entries "
                    f"but 'objectives_weights' has {len(self.objectives_weights)} entries. "
                    "They must be the same length."
                )
            for i, w in enumerate(self.objectives_weights):
                if not isinstance(w, (int, float)):
                    raise ValueError(
                        f"objectives_weights[{i}] = {w!r} is not numeric. "
                        "All weights must be int or float."
                    )

    
    def _report_config(self):
        """
        DESCRIPTION: 
        Reports the configuration of the ChronoRay instance to the console.

        ASSUMPTIONS: 
        - None  

        INPUT(S):   None
        RETURNS:    None
        THROWS:     None
        """
        W = 58
        print("=" * W)
        print("ChronoRay configuration")
        print("=" * W)
        print(f"  simulate_fn           : {self.simulate_fn.__name__}")
        print(f"  evaluate_fn           : {self.evaluate_fn.__name__}")
        print(f"  param_space keys      : {list(self.param_space.keys())}")
        print(f"  resources_per_trial   : {self.resources_per_trial}")
        print(f"  num_samples           : {self.num_samples}")
        print(f"  max_concurrent_trials : {self.max_concurrent_trials}")
        print(f"  metric (Ray target)   : {self.metric}")
        print(f"  mode                  : {self.mode}")
        print(f"  search_algorithm      : {self.search_algorithm.name}")
        if self.search_kwargs:
            print(f"  search_kwargs         : {self.search_kwargs}")
        print(f"  trial_scheduler       : {self.trial_scheduler.name}")
        if self.scheduler_kwargs:
            print(f"  scheduler_kwargs      : {self.scheduler_kwargs}")
        if self.search_algorithm in _SEQUENTIAL_SEARCHERS:
            print(
                f"  [note] {self.search_algorithm.name} is a sequential searcher — "
                f"ConcurrencyLimiter({self.max_concurrent_trials}) applied automatically"
            )
        print(f"  FLAG_multi_objective  : {self.FLAG_multi_objective}")
        if self.FLAG_multi_objective:
            print("  --- Multi-objective breakdown ---")
            total_w = sum(self.objectives_weights)
            for obj, w in zip(self.objectives, self.objectives_weights):
                pct = 100.0 * w / total_w if total_w != 0 else 0.0
                print(f"    {obj:<30s}  weight={w:+.4f}  ({pct:.1f}%)")
            print(f"  combined score reported as '{self.metric}'")
        print("=" * W)

    
    def _build_search_alg(self):
        """
        DESCRIPTION: 
        Instantiates the correct Ray Tune searcher from the enum value.
        - metric / mode are injected automatically where the constructor needs them.
        - Sequential searchers are wrapped in ConcurrencyLimiter automatically.
        - Returns None for GRID (Ray handles grid search internally).

        ASSUMPTIONS: 
        - None 

        INPUT(S):   None
        RETURNS:    searcher (Ray Tune searcher object) or None if GRID
        THROWS:     NotImplementedError if unhandled SearchAlgorithm is provided
        """
        alg = self.search_algorithm
        kw  = dict(self.search_kwargs) 

        if alg in _METRIC_MODE_SEARCHERS:
            kw.setdefault("metric", self.metric)
            kw.setdefault("mode",   self.mode)

        if alg == SearchAlgorithm.RANDOM:
            from ray.tune.search.basic_variant import BasicVariantGenerator
            searcher = BasicVariantGenerator(**kw)

        elif alg == SearchAlgorithm.GRID:
            return None   # Ray Tune handles grid search without a searcher object

        elif alg == SearchAlgorithm.BAYESOPT:
            from ray.tune.search.bayesopt import BayesOptSearch
            searcher = BayesOptSearch(**kw)

        elif alg == SearchAlgorithm.OPTUNA:
            from ray.tune.search.optuna import OptunaSearch
            searcher = OptunaSearch(**kw)

        elif alg == SearchAlgorithm.HYPEROPT:
            from ray.tune.search.hyperopt import HyperOptSearch
            searcher = HyperOptSearch(**kw)

        elif alg == SearchAlgorithm.BOHB:
            from ray.tune.search.bohb import TuneBOHB
            searcher = TuneBOHB(**kw)

        elif alg == SearchAlgorithm.AX:
            from ray.tune.search.ax import AxSearch
            searcher = AxSearch(**kw)

        elif alg == SearchAlgorithm.HEBO:
            from ray.tune.search.hebo import HEBOSearch
            searcher = HEBOSearch(**kw)

        elif alg == SearchAlgorithm.NEVERGRAD:
            from ray.tune.search.nevergrad import NevergradSearch
            searcher = NevergradSearch(**kw)

        elif alg == SearchAlgorithm.ZOOPT:
            from ray.tune.search.zoopt import ZOOptSearch
            searcher = ZOOptSearch(**kw)

        else:
            raise NotImplementedError(f"Unhandled SearchAlgorithm: {alg}")

        if alg in _SEQUENTIAL_SEARCHERS:
            searcher = ConcurrencyLimiter(
                searcher, max_concurrent=self.max_concurrent_trials
            )

        return searcher


    def _build_scheduler(self):
        """
        DESCRIPTION: 
        Instantiates the correct Ray Tune scheduler from the enum value.
        metric / mode are injected automatically.
        Returns None for TrialScheduler.NONE.

        ASSUMPTIONS: 
        - None 

        INPUT(S):   None
        RETURNS:    scheduler (Ray Tune scheduler object) or None if NONE
        THROWS:     NotImplementedError if unhandled TrialScheduler is provided
        """
        sched = self.trial_scheduler
        kw    = dict(self.scheduler_kwargs)
        kw.setdefault("metric", self.metric)
        kw.setdefault("mode",   self.mode)

        if sched == TrialScheduler.NONE:
            return None

        elif sched == TrialScheduler.ASHA:
            from ray.tune.schedulers.async_hyperband import ASHAScheduler
            return ASHAScheduler(**kw)

        elif sched == TrialScheduler.HYPERBAND:
            from ray.tune.schedulers.hyperband import HyperBandScheduler
            return HyperBandScheduler(**kw)

        elif sched == TrialScheduler.BOHB_SCHED:
            from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
            return HyperBandForBOHB(**kw)

        elif sched == TrialScheduler.PBT:
            from ray.tune.schedulers.pbt import PopulationBasedTraining
            return PopulationBasedTraining(**kw)

        elif sched == TrialScheduler.PB2:
            from ray.tune.schedulers.pb2 import PB2
            return PB2(**kw)

        elif sched == TrialScheduler.MEDIAN_STOPPING:
            from ray.tune.schedulers.median_stopping_rule import MedianStoppingRule
            return MedianStoppingRule(**kw)

        else:
            raise NotImplementedError(f"Unhandled TrialScheduler: {sched}")

    def _combine_objectives(self, result_dict: dict) -> float:
        """
        DESCRIPTION: 
        Computes the linear combination of sub-objectives from a result dictionary.

        ASSUMPTIONS: 
        - None 

        INPUT(S):   result_dict (dict) [result dictionary from the evaluate_fn].
        RETURNS:    combined (float) [linear combination of sub-objectives].
        """
        missing = [obj for obj in self.objectives if obj not in result_dict]
        if missing:
            raise KeyError(
                f"evaluate_fn result dict is missing objective key(s): {missing}. "
                f"Dict contained: {list(result_dict.keys())}"
            )
        return sum(
            w * result_dict[obj]
            for obj, w in zip(self.objectives, self.objectives_weights)
        )

    def _ray_objective(self, config):
        """
        DESCRIPTION: 
        Runs the simulation, evaluates the objective function, and reports the metric to Ray.

        ASSUMPTIONS: 
        - None 

        INPUT(S):   config (dict) [configuration of the simulation parameters].
        RETURNS:    None
        THROWS:     TypeError / KeyError if inputs are not valid
        """
        raw_output = self.simulate_fn(config)
        value      = self.evaluate_fn(raw_output)

        if self.FLAG_multi_objective:
            if not isinstance(value, dict):
                raise TypeError(
                    f"FLAG_multi_objective is True so evaluate_fn must return a dict, "
                    f"but got {type(value).__name__!r}. "
                    f"Expected keys: {self.objectives}"
                )
            combined = self._combine_objectives(value)
            report   = {self.metric: combined}
            report.update(value)   # raw sub-objectives logged alongside combined score

        else:
            if isinstance(value, dict):
                if self.metric not in value:
                    raise KeyError(
                        f"evaluate_fn returned a dict but FLAG_multi_objective is False. "
                        f"The dict must contain key '{self.metric}', "
                        f"but only found: {list(value.keys())}. "
                        f"Either set FLAG_multi_objective=True or return a plain scalar."
                    )
                report = value   # pass everything through for logging
            else:
                report = {self.metric: value}

        session.report(report)

    def run(self):
        """
        DESCRIPTION: 
        Entry point – configure and execute the Ray Tune optimisation.

        ASSUMPTIONS: 
        - None 

        INPUT(S):   None
        RETURNS:    ray.tune.ResultGrid
        THROWS:     None
        """
        if not ray.is_initialized():
            ray.init()

        search_alg = self._build_search_alg()   # None for GRID
        scheduler  = self._build_scheduler()     # None for NONE

        trainable = tune.with_resources(
            self._ray_objective,
            resources=self.resources_per_trial,
        )

        tune_config_kwargs = {"num_samples": self.num_samples}
        if search_alg is not None:
            tune_config_kwargs["search_alg"] = search_alg
        if scheduler is not None:
            tune_config_kwargs["scheduler"] = scheduler

        tuner = tune.Tuner(
            trainable,
            param_space=self.param_space,
            tune_config=tune.TuneConfig(**tune_config_kwargs),
        )

        return tuner.fit()


   