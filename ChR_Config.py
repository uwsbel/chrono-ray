import ray 
from ray import tune
from typing import Sequence, Callable, Iterable, Any
from enum import Enum
import textwrap

#********************************************************************
# DIRECTORY 
# 1. [SEARCH ALGORITHMS] ChR_SearchAlg (Enum) - search algorithms
# 2. [DISTRIBUTIONS] ChR_Distr (Class w/ static methods) - statistical distributions
#********************************************************************

#================================================================================
# 1. SEARCH ALGORITHMS
#================================================================================
"""
Searcher compatibility quick-reference
---------------------------------------
Searcher          | needs metric/mode | supports concurrent | typical use
------------------|-------------------|---------------------|-------------------------------
RANDOM            | no                | yes (unlimited)     | baseline / large spaces
GRID              | no                | yes (unlimited)     | small discrete spaces
BAYESOPT          | yes               | limited (1-8)       | cheap fn, small param count
OPTUNA            | yes               | yes                 | general purpose, most popular
HYPEROPT          | yes               | limited             | tree-structured spaces
BOHB              | yes               | limited             | needs paired BOHB scheduler
AX                | yes               | yes                 | expensive fn, constraints
HEBO              | yes               | limited             | high-dim Bayesian
NEVERGRAD         | yes               | yes                 | black-box, gradient-free
ZOOPT             | yes               | limited             | noisy / discrete spaces
"""

class ChR_SearchAlg(Enum):
    """
    Ray Tune search algorithms.

    Pass as `search_algorithm=ChronoRaySearchAlgorithm.OPTUNA` (etc.) to ChronoRay.

    RANDOM      - uniform random sampling (no extra dependency)
    GRID        - exhaustive grid search  (no extra dependency)
    BAYESOPT    - Gaussian-process Bayesian optimisation  [pip: bayesian-optimization]
    OPTUNA      - Optuna TPE / CMA-ES / others            [pip: optuna]
    HYPEROPT    - Tree-structured Parzen Estimators        [pip: hyperopt]
    BOHB        - Bayesian Optimisation + HyperBand        [pip: hpbandster ConfigSpace]
                  Must be paired with TrialScheduler.BOHB_SCHED
    AX          - Adaptive Experimentation Platform        [pip: ax-platform]
    HEBO        - Heteroscedastic Evolutionary BO          [pip: hebo]
    NEVERGRAD   - Gradient-free optimisation toolbox       [pip: nevergrad]
    ZOOPT       - Zeroth-order optimisation                [pip: zoopt]
    """
    def __new__(cls, sequential, metric_mode):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__) + 1
        obj.sequential  = sequential
        obj.metric_mode = metric_mode
        return obj

    @staticmethod
    def info() -> None:
        print("========================================================")
        print("Available ChR_SearchAlg search algorithms:")
        print("========================================================")
        print(f"  {'Algorithm':<12} {'Sequential':<12} {'Needs metric/mode':<20} Notes")
        print(f"  {'-'*12} {'-'*12} {'-'*20} {'-'*30}")
        for alg in ChR_SearchAlg:
            seq  = "yes" if alg.sequential  else "no"
            mm   = "yes" if alg.metric_mode else "no"
            print(f"  {alg.name:<12} {seq:<12} {mm:<20}")
        print(textwrap.dedent("""
            Notes:
                RANDOM    - no extra dependency
                GRID      - no extra dependency
                BAYESOPT  - pip install bayesian-optimization
                OPTUNA    - pip install optuna
                HYPEROPT  - pip install hyperopt
                BOHB      - pip install hpbandster ConfigSpace
                AX        - pip install ax-platform
                HEBO      - pip install hebo
                NEVERGRAD - pip install nevergrad
                ZOOPT     - pip install zoopt
        """))
        print("========================================================")

    RANDOM    = (False, False)
    GRID      = (False, False)
    BAYESOPT  = (True,  True)
    OPTUNA    = (False, True)
    HYPEROPT  = (True,  True)
    BOHB      = (True,  True)
    AX        = (False, True)
    HEBO      = (True,  True)
    NEVERGRAD = (False, True)
    ZOOPT     = (True,  True)

#================================================================================
# 2. DISTRIBUTIONS
#================================================================================   

"""
A thin, self-contained bridge to every Ray Tune sampling distribution.

All methods are static — call them directly on the class, no instantiation needed.

Quick-reference
---------------
Distribution    | Signature                         | Use when...
----------------|-----------------------------------|------------------------------
uniform         | (lower, upper)                    | continuous, linear scale
loguniform      | (lower, upper, base=10)           | continuous, log scale
randn           | (mean, sd)                        | continuous, Gaussian prior
randint         | (lower, upper)                    | integer, uniform
lograndint      | (lower, upper, base=10)           | integer, log scale
choice          | (categories)                      | discrete / categorical list
grid_search     | (values)                          | exhaustive grid over a list
quniform        | (lower, upper, q)                 | continuous rounded to nearest q
qloguniform     | (lower, upper, q, base=10)        | log-scale rounded to nearest q
qrandn          | (mean, sd, q)                     | Gaussian rounded to nearest q
qrandint        | (lower, upper, q=1)               | integer rounded to nearest q
qlograndint     | (lower, upper, q, base=10)        | log-scale integer rounded to q
sample_from     | (func: config -> value)           | custom / dependent sampling

Example
-------
    param_space = {
        "spring_k"  : ChronoRayDistributions.loguniform(1.0, 1000.0),
        "damper_c"  : ChronoRayDistributions.uniform(0.1, 50.0),
        "mass"      : ChronoRayDistributions.choice([0.5, 1.0, 2.0, 5.0]),
        "num_links" : ChronoRayDistributions.randint(1, 10),
    }
"""


class ChR_Distr:

    FLAG_is_chr_distr = True


    @staticmethod
    def uniform(lower: float, upper: float):
        """
        Sample uniformly between `lower` and `upper` on a linear scale.

        Example: ParamDist.uniform(0.0, 100.0)
            -> draws values like 12.4, 67.8, 3.1 ...
        Good for: stiffness, damping, mass, geometry parameters with no
                  strong preference for any sub-range.
        """
        d = tune.uniform(lower, upper)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def loguniform(lower: float, upper: float, base: float = 10):
        """
        Sample uniformly between `lower` and `upper` on a log scale.
        Draws more samples near the lower end of the range.

        Example: ParamDist.loguniform(1e-4, 1e-1)
            -> draws values like 0.0003, 0.012, 0.07 ...
        Good for: parameters spanning multiple orders of magnitude,
                  e.g. spring constants from 1 to 10000 N/m.

        base : log base (default 10; use math.e for natural log spacing)
        """
        d = tune.loguniform(lower, upper, base)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def randn(mean: float, sd: float):
        """
        Sample from a Gaussian (normal) distribution N(mean, sd).

        Example: ParamDist.randn(5.0, 1.0)
            -> draws values clustered around 5.0 with std-dev 1.0
        Good for: parameters where you have a strong prior expectation
                  of the correct value and want to search around it.
        Note: unbounded — can produce values outside any intended range.
              Use sample_from with clipping if hard bounds are required.
        """
        d = tune.randn(mean, sd)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def randint(lower: int, upper: int):
        """
        Sample a uniformly random integer in [lower, upper). upper is exclusive.

        Example: ParamDist.randint(1, 8)
            -> draws values like 1, 3, 5, 7 ...
        Good for: number of links, solver iterations, polynomial degree.
        """
        d = tune.randint(lower, upper)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def lograndint(lower: int, upper: int, base: float = 10):
        """
        Sample a random integer in [lower, upper) on a log scale. upper is exclusive.

        Example: ParamDist.lograndint(1, 1000)
            -> draws integers denser near 1, sparser near 1000
        Good for: integer parameters spanning orders of magnitude,
                  e.g. number of solver sub-steps from 1 to 1000.

        base : log base (default 10)
        """
        d = tune.lograndint(lower, upper, base)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def choice(categories: Sequence):
        """
        Sample uniformly from a fixed list of options.
        Options can be strings, numbers, or any Python objects.

        Example: ParamDist.choice(["euler", "runge_kutta", "implicit"])
                 ParamDist.choice([0.5, 1.0, 2.0, 5.0])
        Good for: solver type, contact model, material preset,
                  any discrete named option.
        """
        d = tune.choice(categories)
        d.FLAG_is_chr_distr = True
        return d
    
    class _ChRDistrDict(dict): 
        pass 
    
    @staticmethod
    def grid_search(values: Iterable):
        """
        Exhaustively evaluate every value in the list.
        Combined with other grid_search params it forms a full Cartesian grid.

        Example: ParamDist.grid_search([0.01, 0.1, 1.0, 10.0])
            -> Ray runs one trial per value (4 trials for this param alone)
        Good for: a small known set of values you want to evaluate completely.
        Note: best paired with SearchAlgorithm.GRID.
        """
        d = ChR_Distr._ChRDistrDict(tune.grid_search(values))
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def quniform(lower: float, upper: float, q: float):
        """
        Sample uniformly in [lower, upper], then round to the nearest multiple of q.

        Example: ParamDist.quniform(0.0, 1.0, 0.1)
            -> draws values like 0.0, 0.1, 0.3, 0.7, 1.0
        Good for: step sizes or ratios that must be multiples of a fixed increment.
        """
        d = tune.quniform(lower, upper, q)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def qloguniform(lower: float, upper: float, q: float, base: float = 10):
        """
        Sample log-uniformly in [lower, upper], then round to the nearest multiple of q.

        Example: ParamDist.qloguniform(1e-4, 1e-1, 1e-4)
            -> log-spaced floats snapped to 4-decimal-place increments
        Good for: tolerances or rates needing a discrete grid on a log scale.

        base : log base (default 10)
        """
        d = tune.qloguniform(lower, upper, q, base)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def qrandn(mean: float, sd: float, q: float):
        """
        Sample from N(mean, sd), then round to the nearest multiple of q.

        Example: ParamDist.qrandn(5.0, 1.0, 0.5)
            -> Gaussian values snapped to 0.5 increments: 4.0, 4.5, 5.0, 5.5 ...
        Good for: Gaussian-prior parameters that must land on a discrete grid.
        """
        d = tune.qrandn(mean, sd, q)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def qrandint(lower: int, upper: int, q: int = 1):
        """
        Sample a random integer in [lower, upper), then round to the nearest multiple of q.

        Example: ParamDist.qrandint(0, 100, 10)
            -> draws values like 0, 10, 20, 50, 90 ...
        Good for: batch size, hidden units, anything that must be a
                  multiple of some base value.
        """
        d = tune.qrandint(lower, upper, q)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def qlograndint(lower: int, upper: int, q: int, base: float = 10):
        """
        Sample a log-scale integer in [lower, upper), then round to the nearest multiple of q.

        Example: ParamDist.qlograndint(1, 1000, 10)
            -> log-spaced integers snapped to multiples of 10
        Good for: large integer ranges on a log grid.

        base : log base (default 10)
        """
        d = tune.qlograndint(lower, upper, q, base)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def sample_from(func: Callable[[dict], Any]):
        """
        Define a custom sampling function that can depend on other sampled parameters.
        `func` receives the full config dict sampled so far and returns a value.

        Example — damping always <= half the spring constant:
            ParamDist.sample_from(lambda cfg: cfg["spring_k"] * 0.5 * random.random())

        Example — clipped Gaussian:
            ParamDist.sample_from(lambda cfg: max(0.0, min(10.0, random.gauss(5.0, 1.0))))

        Good for: physically constrained relationships between parameters,
                  or any distribution not covered by the built-ins above.
        """
        d = tune.sample_from(func)
        d.FLAG_is_chr_distr = True
        return d

    @staticmethod
    def _format_distr(d) -> str:
        if isinstance(d, ChR_Distr._ChRDistrDict):
            return f"grid_search(values={list(d.get('grid_search', []))})"
        
        attrs = {k: v for k, v in vars(d).items() if k not in ("sampler", "FLAG_is_chr_distr")}
        
        if hasattr(d, "sampler"):
            distr_name = type(d.sampler).__name__.strip("_")
        else:
            distr_name = type(d).__name__
        
        args_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
        return f"{distr_name}({args_str})"

    @staticmethod
    def info() -> None:
        import inspect
        print("========================================================")
        print("Available ChR_Distr sampling distributions:")
        print("========================================================")
        for name, method in inspect.getmembers(ChR_Distr, predicate=callable):
            if not name.startswith("_"):
                doc = inspect.getdoc(method)
                print(f"\n  {name}")
                print(f"    {doc}")
        print("========================================================")