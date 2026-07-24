import math
import numpy as np
from ray.tune.search import Searcher
from collections.abc import Callable
import textwrap

from ChronoRay.ChR_ChronoRay import ChR_ChronoRay
from ChronoRay.ChR_Config import ChR_Distr, ChR_SearchAlg

#crash detector: a protected sim returns the +inf crash sentinel on failure, and
#is_crash() recognises it so a failed trial can be turned into a rejection.
#NOTE: fix this import path to wherever crash_safe.py lives in the package.
from crash_safe import is_crash


class PriorMetropolisSearch(Searcher):

    # ==================================================================
    # __init__ : set up the chain's memory
    # the only new wrinkle is that the prior (param_space) may arrive
    # later via set_search_properties, so we leave a slot for it
    # ==================================================================
    def __init__(self, param_space=None, metric="output", sigma=1.0, burn_in_frac=0.0, seed=0, **kw):
        mode = "min"
        super().__init__(metric=metric, mode=mode, **kw)

        #1. parameter space = the priors for the parameters to be tuned 
        self.param_space = param_space

        #2. chain's current position and its log-likelihood
        self.current_x = None
        self.current_loglik = None

        #3. scratchpad for proposals that are in flight (trial_id -> theta)
        self.pending = {}

        #4. the chain itself (burn-in prefix approach). in the end, it will be the posterior 
        self.samples = []

        #5. A private, seeded RNG (acceptance in the MCMC sampling is a coin flip, kept reproducible) 
        self.rng = np.random.default_rng(seed)

        #6. noise model for negative log-likelihood calculation 
        self.sigma = sigma

        #7. fraction of the earliest samples to discard as burn-in 
        self.burn_in_frac = burn_in_frac

    # ==================================================================
    # set_search_properties : Tune hands us the param_space (the priors)
    # through here, so we capture it if it wasn't passed in directly
    # ==================================================================
    def set_search_properties(self, metric, mode, config, **spec):

        #1. capture the prior space if we don't already have it 
        if self.param_space is None:
            self.param_space = config

        #2. tell Tune we took ownership of the search space 
        return True

    # ------------------------------------------------------------------
    # helper: draw ONE theta straight from the prior
    # Every Domain (tune.uniform, tune.randn, ...) knows how to .sample()
    # itself; plain constants are passed through untouched
    # ------------------------------------------------------------------
    def _sample_prior(self):

        #note: compact dict generation
        return {key: (distribution.sample() if hasattr(distribution, "sample") else distribution) for key, distribution in self.param_space.items()}

    # ==================================================================
    # suggest : the ASK half. Propose a candidate -- here the candidate
    # IS a fresh draw from the prior (that's the independence sampler)
    # ==================================================================
    def suggest(self, trial_id):

        #1. propose new theta by sampling from the prior distr 
        x_prop = self._sample_prior()

        #2. log the proposal in the scratchpad 
        self.pending[trial_id] = x_prop

        #3. return the proposal to Tune to actually run the simulator on it
        return x_prop

    # ==================================================================
    # on_trial_complete : the TELL half. The simulator finished, so the
    # Metropolis accept/reject happens here -- now just a likelihood ratio
    # ==================================================================
    def on_trial_complete(self, trial_id, result=None, error=False):

        #1. recover theta value for trial 
        x_prop = self.pending.pop(trial_id, None)

        #1a. unknown trial_id -> nothing to score, nothing to record 
        if x_prop is None:
            return

        #2. a FAILED trial (sim crash sentinel, Ray error, or missing metric) is a
        #   plain REJECT: stay put and re-record the current state. a crash is the
        #   ABSENCE of information about this point, NOT evidence it fits badly --
        #   so we never run it through the likelihood (it can't be accepted, and it
        #   can't distort the posterior with a fake misfit).
        failed = (
            error
            or result is None
            or self.metric not in result
            or is_crash(result[self.metric])
        )

        if failed:
            #2a. a reject only means something once we HAVE a current state. if the
            #    very first step failed there's nothing to fall back to, so we skip
            #    it and wait for a usable first sample.
            if self.current_x is not None:
                self.samples.append(self.current_x)
            return

        #3. turn the user's sim-to-real data mismatch  into a log-likelihood
        #user reports misfit (e.g. sum of squared errors), lower = better fit
        #under Gaussian noise: loglik = -misfit / (2 * sigma^2)
        misfit_new = result[self.metric]
        loglik_new = -misfit_new / (2 * self.sigma ** 2)

        #APPENDING TO THE CHAIN: 

        #4a. if this is first step of the chain, there nothing to compare to, so just adopt this point
        if self.current_x is None:
            self.current_x, self.current_loglik = x_prop, loglik_new

        #4b. accept/reject
        else:
            log_alpha = loglik_new - self.current_loglik
            if math.log(self.rng.uniform()) < log_alpha:
                self.current_x, self.current_loglik = x_prop, loglik_new
            # else: reject -> stay put, current_x unchanged

        #5. append the sample to the chain (building the posterior)
        self.samples.append(self.current_x)

    # ==================================================================
    # get_posterior : return the chain with the burn-in prefix discarded
    # (the early "travel" samples before the chain settled)
    # ==================================================================
    def get_posterior(self):

        #1. how many of the earliest samples to drop 
        start = int(self.burn_in_frac * len(self.samples))

        #2. return everything after the burn-in prefix 
        return self.samples[start:]


class ChRBayesCali:

    @staticmethod
    def info() -> None:
        print("========================================================")
        print("ChRBayesCali - ChronoRay (PyChrono + Ray) Bayesian Calibration Workflow")
        print("========================================================")
        print(textwrap.dedent("""
            DESCRIPTION:
                ChRBayesCali calibrates a PyChrono simulation against observed data.
                Unlike ChRBayesOpt, it does NOT return a single best parameter set --
                it returns a POSTERIOR DISTRIBUTION over the parameters: the values that
                make the simulation match the data, together with their uncertainty.

                The search backend is an MCMC sampler (prior-proposal Metropolis), not
                an optimizer. Each trial = one simulator run = one step in the chain.

                Two ideas distinguish this from ChRBayesOpt:
                  - param_prior_space defines PRIOR distributions, not just a search range.
                    The sampler draws its candidate parameters straight from these priors.
                  - there IS observed data to match. Your objective_fn scores how far a
                    simulation is from that data (returning a misfit, lower = better).

            --------------------------------------------------------
            CONSTRUCTOR ARGUMENTS:
            --------------------------------------------------------

                simulate_fn (Callable) [REQUIRED]
                    A PyChrono simulation function.
                    INPUT:   config (dict) -- parameter names and their sampled values.
                    RETURNS: output (dict) -- simulation output values.
                    NOTE:    no visualization (e.g. Irrlicht, VSG) should take place.

                objective_fn (Callable) [REQUIRED]
                    Scores how far a single simulation run is from your observed data.
                    INPUT:   output (dict) -- return value of simulate_fn.
                    RETURNS: misfit (float) -- lower = better fit.
                    IMPORTANT: report DATA MISFIT ONLY. Do not add any prior/penalty term.
                               The prior is already applied by drawing candidates from
                               param_prior_space; adding it here would double-count it.
                    NOTE:    close over your observed data inside the function.
                    Example: observed = {"settling_time": 1.42}
                             objective_fn = lambda output: (
                                 (output["settling_time"] - observed["settling_time"]) ** 2
                             )

                param_prior_space (dict[str, ChR_Distr]) [REQUIRED]
                    Parameters to calibrate and their PRIOR distributions.
                    Keys   : parameter names (must match keys expected by simulate_fn).
                    Values : ChR_Distr distributions -- treated as priors.
                    NOTE:    run ChRBayesCali.ChR_Distr.info() for available distributions.

                sigma (float) [OPTIONAL, default: 1.0]
                    Assumed noise / scatter between simulation and data. It converts the
                    misfit into a likelihood:   loglik = -misfit / (2 * sigma**2).
                    Larger sigma -> more tolerant of mismatch (wider posterior).
                    Set it to reflect your measurement noise. It can later be promoted to
                    a parameter and inferred too, but a fixed value is fine to start.

                total_samples (int) [OPTIONAL, default: 2000]
                    Number of MCMC steps (= number of simulator runs). Mapping a full
                    posterior needs many more samples than optimization needs trials.

                burn_in_frac (float) [OPTIONAL, default: 0.3]
                    Fraction of the earliest samples to discard. The chain starts at an
                    arbitrary point and needs time to settle into the region the data
                    supports; those early "travel" samples would bias the result.

                FLAG_log_to_file (bool) [OPTIONAL, default: False]
                    If True, Ray's console output is redirected to a timestamped
                    .txt file in the current working directory. User prints are also
                    redirected to .txt file. Default keeps all output in the console.

                FLAG_auto_run (bool) [OPTIONAL, default: True]
                    Controls the operating mode. See OPERATING MODES below.

                NOTE ON CONCURRENCY:
                    A single MCMC chain is strictly sequential -- each step depends on the
                    previous one's result -- so trials run ONE AT A TIME. There is no
                    max_concurrent_trials knob here; it is fixed to 1 internally.
                    (Parallelism would require multiple chains, which this class does not
                    yet expose.)

            --------------------------------------------------------
            OPERATING MODES:
            --------------------------------------------------------

                MODE 1 -- FLAG_auto_run=True (default)
                    The calibration runs immediately on construction.
                    No further configuration is possible.

                    Example:
                        cal = ChRBayesCali(
                            simulate_fn  = my_sim,
                            objective_fn = lambda output: (
                                (output["settling_time"] - 1.42) ** 2
                            ),
                            param_prior_space = {
                                "k": ChRBayesCali.ChR_Distr.uniform(0, 10),
                                "m": ChRBayesCali.ChR_Distr.loguniform(1e-2, 1e2),
                            },
                            sigma = 0.05,
                        )

                MODE 2 -- FLAG_auto_run=False
                    Construction stops after validation and config report.
                    The user can then configure optional settings via the
                    setters listed below before manually building and running.

                    Example:
                        cal = ChRBayesCali(
                            simulate_fn       = my_sim,
                            objective_fn      = my_objective,
                            param_prior_space = {...},
                            sigma             = 0.05,
                            FLAG_auto_run     = False
                        )
                        cal.set_resources_per_trial(cpu=4, gpu=0)
                        cal.set_FLAG_log_to_file(True)
                        cal._build_backend()
                        cal.run()

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

    def __init__(
                self,
                simulate_fn: Callable,
                objective_fn: Callable,
                param_prior_space: dict[str, ChR_Distr],
                sigma: float = 1.0,
                total_samples: int = 2000,
                burn_in_frac: float = 0.3,
                FLAG_log_to_file: bool = False,
                FLAG_auto_run: bool = True) -> None:

        #1. mandatory parameters
        self.simulate_fn = simulate_fn
        self.objective_fn = objective_fn
        self.param_prior_space = param_prior_space

        #2. calibration settings
        self.sigma = sigma
        self.total_samples = total_samples
        self.burn_in_frac = burn_in_frac

        #3. single-chain MCMC is sequential -> exactly one trial at a time
        self.max_concurrent_trials = 1

        #4. optional parameters
        self.resources_per_trial = {"cpu": 1, "gpu": 1}
        self.FLAG_log_to_file = FLAG_log_to_file

        self._validate_inputs()
        self._report_config()

        #5. immediately run if FLAG_auto_run is True
        self.FLAG_auto_run = FLAG_auto_run
        self.backend = None
        if FLAG_auto_run:
            self.backend = self._build_backend()
            self.run()

    def _validate_inputs(self) -> None:

        print("************************************************************")
        print("Validating inputs...")
        print("run ChRBayesCali.info() for information on how to use this workflow.")
        print("************************************************************")

        #1. simulate_fn check
        if not callable(self.simulate_fn):
            raise ValueError("simulate_fn must be callable")

        #2. objective_fn check
        if not callable(self.objective_fn):
            raise ValueError("objective_fn must be callable")

        #3. parameter configuration checks
        if not isinstance(self.param_prior_space, dict):
            raise TypeError("param_prior_space must be a dict")

        if not all(isinstance(k, str) for k in self.param_prior_space.keys()):
            raise TypeError("param_prior_space keys must all be strings")

        if not all(getattr(v, "FLAG_is_chr_distr", False) for v in self.param_prior_space.values()):
            raise TypeError("param_prior_space values must all be ChR_Distr distributions. Use ChRBayesCali.ChR_Distr.info() for available options and additional info.")

        #4. sigma check
        if not isinstance(self.sigma, (int, float)) or self.sigma <= 0:
            raise ValueError(f"sigma must be a positive number, got {self.sigma!r}")

        #5. total_samples check
        if not isinstance(self.total_samples, int) or self.total_samples <= 0:
            raise ValueError(f"total_samples must be a positive integer, got {self.total_samples!r}")

        #6. burn_in_frac check
        if not isinstance(self.burn_in_frac, (int, float)) or not (0.0 <= self.burn_in_frac < 1.0):
            raise ValueError(f"burn_in_frac must be in [0.0, 1.0), got {self.burn_in_frac!r}")

    def _report_config(self) -> None:

        print("************************************************************")
        print("========================================================")
        print("ChRBayesCali (\"ChronoRay Bayesian Calibration Workflow\") configuration:")
        print("========================================================")
        print(f"  1. simulate_fn        : {self.simulate_fn.__name__}")
        print(f"  2. objective_fn       : {self.objective_fn.__name__}")
        print(f"  3. param_prior_space  :")

        for name, distr in self.param_prior_space.items():
            print(f"       {name} : {ChR_Distr._format_distr(distr)}")

        print(f"  4. sigma              : {self.sigma}")
        print(f"  5. total_samples      : {self.total_samples}")
        print(f"  6. burn_in_frac       : {self.burn_in_frac}")
        print(f"  7. search_algorithm   : METROPOLIS (fixed)")
        print(f"  8. max_concurrent     : 1 (single chain, sequential)")
        print(f"  9. FLAG_log_to_file        : {self.FLAG_log_to_file}")
        print("************************************************************")

    #<3 METHODS IN USER INTERFACE
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
            param_space=self.param_prior_space,
            resources_per_trial=self.resources_per_trial,
            num_trials=self.total_samples,
            max_concurrent_trials=self.max_concurrent_trials,
            mode="min",                              # always minimize misfit / NLL
            search_algorithm=ChR_SearchAlg.METROPOLIS,
            search_kwargs={
                "sigma": self.sigma,                 # misfit -> likelihood conversion
                "burn_in_frac": self.burn_in_frac,   # discard the chain's early travel
            },
        )

    def run(self) -> None:
        if self.backend is None:
            raise ValueError("backend has not been built yet. Call _build_backend() first.")

        self.FLAG_auto_run = True
        self.backend.run(FLAG_log_to_file=self.FLAG_log_to_file)

    def get_posterior(self):
        if self.backend is None:
            raise ValueError("backend has not been built/run yet. Call run() first.")
        return self.backend.get_searcher().get_posterior()