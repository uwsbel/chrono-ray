from enum import Enum, auto

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

    RANDOM      – uniform random sampling (no extra dependency)
    GRID        – exhaustive grid search  (no extra dependency)
    BAYESOPT    – Gaussian-process Bayesian optimisation  [pip: bayesian-optimization]
    OPTUNA      – Optuna TPE / CMA-ES / others            [pip: optuna]
    HYPEROPT    – Tree-structured Parzen Estimators        [pip: hyperopt]
    BOHB        – Bayesian Optimisation + HyperBand        [pip: hpbandster ConfigSpace]
                  Must be paired with TrialScheduler.BOHB_SCHED
    AX          – Adaptive Experimentation Platform        [pip: ax-platform]
    HEBO        – Heteroscedastic Evolutionary BO          [pip: hebo]
    NEVERGRAD   – Gradient-free optimisation toolbox       [pip: nevergrad]
    ZOOPT       – Zeroth-order optimisation                [pip: zoopt]
    """
    RANDOM    = auto()
    GRID      = auto()
    BAYESOPT  = auto()
    OPTUNA    = auto()
    HYPEROPT  = auto()
    BOHB      = auto()
    AX        = auto()
    HEBO      = auto()
    NEVERGRAD = auto()
    ZOOPT     = auto()