"""
ChronoRay
=========
A Ray Tune wrapper for PyChrono (or any) simulation-based optimisation.

Key features
------------
- Pluggable search algorithm  (ChronoRaySearchAlgorithm enum)
- Automatic ConcurrencyLimiter for gradient-based / sequential searchers
"""

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.air import session
from ChronoRaySearchAlgorithms import ChronoRaySearchAlgorithms


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

class ChR_ChronoRay:
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
        - The output is a scalar float.
        
        INPUT(S):   raw_output (any type) [raw output of the simulate_fn() function].
        RETURNS:    value (float) [value of the objective function].

    4. resources_per_trial : dict [OPTIONAL]
        DESCRIPTION: 
        Dictionary specifying the resources required for each trial.
        e.g. {"cpu": 4, "gpu": 0}

        ASSUMPTIONS: 
        - If not specified, the default value of {"cpu": 1, "gpu": 0} will be used.
        - It must be a dictionary with keys of type string and values of type int.

    5. num_trials : int
        DESCRIPTION: 
        Total number of trials to run.

        ASSUMPTIONS: 
        - It must be greater than 0.
        - The default value is 10.

    6. max_concurrent_trials : int
        DESCRIPTION: 
        Maximum number of trials allowed to run simultaneously.
        For sequential searchers (BayesOpt, HyperOpt, BOHB, HEBO, ZOOpt)
        lower values (2-4) are strongly recommended so the optimizer can
        learn from completed trials before launching new ones.
        Random / Grid / Optuna / Ax / Nevergrad tolerate higher values.

        ASSUMPTIONS: 
        - It must be greater than 0.
        - The default value is 4.

    7. metric : str
        DESCRIPTION: 
        Name of the scalar reported to Ray as the optimisation target.

        ASSUMPTIONS: 
        - The default value is "objective".

    8. mode : str
        DESCRIPTION: 
        Mode of the optimization.

        ASSUMPTIONS: 
        - It must be either "min" or "max".
        - The default value is "min".

    9. search_algorithm : ChronoRaySearchAlgorithm
        DESCRIPTION: 
        Which search algorithm to use.  

        ASSUMPTIONS: 
        - It must be a ChronoRaySearchAlgorithm enum member.
        - The default value is ChronoRaySearchAlgorithm.BAYESOPT.

    10. search_algorithm_kwargs : dict [OPTIONAL]
        DESCRIPTION: 
        Extra keyword arguments forwarded verbatim to the search algorithm constructor. 
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
        num_trials=10,
        max_concurrent_trials=4,
        metric="objective",
        mode="min",
        search_algorithm: ChronoRaySearchAlgorithm = ChronoRaySearchAlgorithms.BAYESOPT,
        search_kwargs: dict | None = None,
    ):
        self.simulate_fn           = simulate_fn
        self.evaluate_fn           = evaluate_fn
        self.param_space           = param_space
        self.resources_per_trial   = resources_per_trial or {"cpu": 1, "gpu": 0}
        self.num_trials            = num_trials
        self.max_concurrent_trials = max_concurrent_trials
        self.metric                = metric
        self.mode                  = mode
        self.search_algorithm      = search_algorithm
        self.search_kwargs         = search_kwargs or {}

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
        print(f"  num_trials            : {self.num_trials}")
        print(f"  max_concurrent_trials : {self.max_concurrent_trials}")
        print(f"  metric (Ray target)   : {self.metric}")
        print(f"  mode                  : {self.mode}")
        print(f"  search_algorithm      : {self.search_algorithm.name}")
        if self.search_kwargs:
            print(f"  search_kwargs         : {self.search_kwargs}")
        if self.search_algorithm in _SEQUENTIAL_SEARCHERS:
            print(
                f"  [note] {self.search_algorithm.name} is a sequential searcher — "
                f"ConcurrencyLimiter({self.max_concurrent_trials}) applied automatically"
            )
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

    def _ray_objective(self, config):
        """
        DESCRIPTION: 
        Runs the simulation, evaluates the objective function, and reports the metric to Ray.

        ASSUMPTIONS: 
        - None 

        INPUT(S):   config (dict) [configuration of the simulation parameters].
        RETURNS:    None
        THROWS:     TypeError if evaluate_fn does not return a scalar float
        """
        raw_output = self.simulate_fn(config)
        value      = self.evaluate_fn(raw_output)

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"evaluate_fn must return a scalar float, but got {type(value).__name__!r}."
            )

        session.report({self.metric: value})

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

        trainable = tune.with_resources(
            self._ray_objective,
            resources=self.resources_per_trial,
        )

        tune_config_kwargs = {"num_trials": self.num_trials}
        if search_alg is not None:
            tune_config_kwargs["search_alg"] = search_alg

        tuner = tune.Tuner(
            trainable,
            param_space=self.param_space,
            tune_config=tune.TuneConfig(**tune_config_kwargs),
        )

        return tuner.fit()