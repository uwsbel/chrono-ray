"""
ChronoRay
=========
A Ray Tune wrapper for PyChrono (or any) simulation-based optimisation.

Key features
------------
- Pluggable search algorithm  (ChronoRayChR_SearchAlg enum)
- Automatic ConcurrencyLimiter for gradient-based / sequential searchers
"""

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.air import session
from ChronoRay.ChR_Config import ChR_SearchAlg



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
        RETURNS:    raw_output (any type) [raw output of the to be used by the objective_fn].

    3. objective_fn : callable (i.e function)
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

    9. search_algorithm : ChronoRayChR_SearchAlg
        DESCRIPTION: 
        Which search algorithm to use.  

        ASSUMPTIONS: 
        - It must be a ChronoRayChR_SearchAlg enum member.
        - The default value is ChronoRayChR_SearchAlg.BAYESOPT.

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
        objective_fn,
        param_space,
        resources_per_trial,
        num_trials,
        max_concurrent_trials,
        mode, 
        search_algorithm,
        search_kwargs,
    ):
        self.simulate_fn           = simulate_fn
        self.objective_fn          = objective_fn
        self.param_space           = param_space
        self.resources_per_trial   = resources_per_trial or {"cpu": 1, "gpu": 0}
        self.num_trials            = num_trials
        self.max_concurrent_trials = max_concurrent_trials
        self.mode                  = mode
        self.search_algorithm      = search_algorithm
        self.search_kwargs         = search_kwargs or {}
        self.metric                = "objective"

   

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
        THROWS:     NotImplementedError if unhandled ChR_SearchAlg is provided
        """
        alg = self.search_algorithm
        kw  = dict(self.search_kwargs)

        if alg.metric_mode:
            kw.setdefault("metric", self.metric)
            kw.setdefault("mode",   self.mode)

        if alg == ChR_SearchAlg.RANDOM:
            from ray.tune.search.basic_variant import BasicVariantGenerator
            searcher = BasicVariantGenerator(**kw)

        elif alg == ChR_SearchAlg.GRID:
            return None

        elif alg == ChR_SearchAlg.BAYESOPT:
            from ray.tune.search.bayesopt import BayesOptSearch
            searcher = BayesOptSearch(**kw)

        elif alg == ChR_SearchAlg.OPTUNA:
            from ray.tune.search.optuna import OptunaSearch
            searcher = OptunaSearch(**kw)

        elif alg == ChR_SearchAlg.HYPEROPT:
            from ray.tune.search.hyperopt import HyperOptSearch
            searcher = HyperOptSearch(**kw)

        elif alg == ChR_SearchAlg.BOHB:
            from ray.tune.search.bohb import TuneBOHB
            searcher = TuneBOHB(**kw)

        elif alg == ChR_SearchAlg.AX:
            from ray.tune.search.ax import AxSearch
            searcher = AxSearch(**kw)

        elif alg == ChR_SearchAlg.HEBO:
            from ray.tune.search.hebo import HEBOSearch
            searcher = HEBOSearch(**kw)

        elif alg == ChR_SearchAlg.NEVERGRAD:
            from ray.tune.search.nevergrad import NevergradSearch
            searcher = NevergradSearch(**kw)

        elif alg == ChR_SearchAlg.ZOOPT:
            from ray.tune.search.zoopt import ZOOptSearch
            searcher = ZOOptSearch(**kw)

        else:
            raise NotImplementedError(f"Unhandled ChR_SearchAlg: {alg}")

        if alg.sequential:
            searcher = ConcurrencyLimiter(
                searcher, max_concurrent=self.max_concurrent_trials
            )

        return searcher

    def _trial(self, config):
        """
        DESCRIPTION: 
        Runs the simulation, evaluates the objective function, and reports the metric to Ray.

        ASSUMPTIONS: 
        - None 

        INPUT(S):   config (dict) [configuration of the simulation parameters].
        RETURNS:    None
        THROWS:     TypeError if objective_fn does not return a scalar float
        """
        raw_output = self.simulate_fn(config)
        value      = self.objective_fn(raw_output)

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"objective_fn must return a scalar float, but got {type(value).__name__!r}."
            )

        session.report({self.metric: value})

    def run(self):
    """
    DESCRIPTION: 
    Entry point - configure and execute the Ray Tune optimisation.

    ASSUMPTIONS: 
    - None 

    INPUT(S):   None
    RETURNS:    ray.tune.ResultGrid
    THROWS:     None
    """
        import os
        import logging
        from datetime import datetime

        # suppress Ray console output
        os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
        logging.getLogger("ray").setLevel(logging.WARNING)
        logging.getLogger("ray.tune").setLevel(logging.WARNING)

        # redirect Ray logs to timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(os.getcwd(), f"chr_ray_log_{timestamp}.txt")
        logging.basicConfig(filename=log_path, level=logging.WARNING)

        if not ray.is_initialized():
            ray.init(logging_level=logging.WARNING, log_to_driver=False)

        built_search_alg = self._build_search_alg()   # None for GRID

        trainable = tune.with_resources(
            self._trial,
            resources=self.resources_per_trial,
        )

        tune_config_kwargs = {"num_samples": self.num_trials}
        if built_search_alg is not None:
            tune_config_kwargs["search_alg"] = built_search_alg

        tuner = tune.Tuner(
            trainable,
            param_space=self.param_space,
            tune_config=tune.TuneConfig(**tune_config_kwargs),
        )

        return tuner.fit()