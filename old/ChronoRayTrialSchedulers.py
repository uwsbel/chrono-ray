from enum import Enum, auto

"""
Scheduler compatibility quick-reference
-----------------------------------------
Scheduler         | prunes trials | works with         | note
------------------|---------------|--------------------|----------------------------
NONE              | no            | all searchers      | run every trial to completion
ASHA              | yes           | most searchers     | best default pruner
HYPERBAND         | yes           | most searchers     | classic bandit
BOHB_SCHED        | yes           | BOHB searcher only | must be paired with BOHB
PBT               | no (mutates)  | stateful trainables| population-based training
PB2               | no (mutates)  | stateful trainables| PBT variant using GP
MEDIAN_STOPPING   | yes           | most searchers     | simple, conservative

Incompatible pairs (enforced at runtime)
-----------------------------------------
- BOHB_SCHED requires SearchAlgorithm.BOHB  (and vice-versa)
- PBT requires  `hyperparam_mutations` in scheduler_kwargs
- PB2 requires  `hyperparam_bounds`    in scheduler_kwargs

"""

class ChronoRayTrialSchedulers(Enum):
    """
    Ray Tune trial schedulers (early-stopping / population strategies).

    Pass as `trial_scheduler=TrialScheduler.ASHA` (etc.) to ChronoRay.

    NONE            – no scheduler; every trial runs to completion (default)
    ASHA            – Async Successive Halving; best all-round pruner
    HYPERBAND       – classic synchronous HyperBand
    BOHB_SCHED      – HyperBand variant for BOHB; MUST use with SearchAlgorithm.BOHB
    PBT             – Population Based Training; evolves hyperparams mid-run
                      Requires `hyperparam_mutations` in scheduler_kwargs
    PB2             – PBT variant backed by a Gaussian process
                      Requires `hyperparam_bounds` in scheduler_kwargs
    MEDIAN_STOPPING – stops trials performing below the running median
    """
    NONE            = auto()
    ASHA            = auto()
    HYPERBAND       = auto()
    BOHB_SCHED      = auto()
    PBT             = auto()
    PB2             = auto()
    MEDIAN_STOPPING = auto()