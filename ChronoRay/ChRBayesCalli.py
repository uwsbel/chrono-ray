import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import textwrap
from typing import Any, Callable, Iterable, Sequence

#********************************************************************
# DIRECTORY 
# 1. [BAYESIAN CALIBRATION ENGINE] ChRBayesCalli (Class) - Bayesian calibration engine
# 2. [DISTRIBUTIONS] PyMC_Distr_template (Class w/ static methods) - statistical distributions
#********************************************************************

#================================================================================
# 1. BAYESIAN CALIBRATION ENGINE
#================================================================================

class ChRBayesCalli:

    @staticmethod
    def info() -> None:
        print("========================================================")
        print("ChRBayesCalli - ChronoRay Bayesian Calibration Workflow")
        print("========================================================")
        print(textwrap.dedent("""
            DESCRIPTION:
                ChRBayesCalli is a Bayesian calibration framework for simulation models.
                It estimates unknown simulation parameters by comparing simulated outputs
                against observed data using PyMC's simulator-based Bayesian inference tools.

                Unlike ChRBayesOpt, the goal is not to directly minimize or maximize a
                user-defined objective function. Instead, the goal is to infer plausible
                parameter values that could have generated the observed data.

                This is useful when you have experimental or reference data and want to
                calibrate uncertain model parameters such as stiffness, damping, mass,
                friction, geometry, or material properties.

            --------------------------------------------------------
            CONSTRUCTOR ARGUMENTS:
            --------------------------------------------------------

                simulate_fn (Callable) [REQUIRED]
                    A simulation function compatible with PyMC Simulator.
                    INPUT:   rng, parameter values, size
                    RETURNS: simulated data/output array.
                    NOTE:    the function should not perform visualization
                            (e.g. Irrlicht, VSG) during inference.

                    Expected form:
                        def simulate_fn(rng, param_1, param_2, ..., size):
                            ...
                            return simulated_data

                param_space (dict[str, ChR_Distr]) [REQUIRED]
                    Parameters to calibrate and their prior distributions.
                    Keys   : parameter names.
                    Values : ChR_Distr distributions defining Bayesian priors.
                    NOTE:    run ChRBayesCalli.ChR_Distr.info() for available distributions.

                    Example:
                        param_space = {
                            "spring_k": ChRBayesCalli.ChR_Distr.loguniform(1.0, 1000.0),
                            "damper_c": ChRBayesCalli.ChR_Distr.uniform(0.1, 50.0),
                            "mass"    : ChRBayesCalli.ChR_Distr.truncatednormal(
                                            1.0, 0.5, lower=0.0, upper=5.0
                                        ),
                        }

                data (array-like) [OPTIONAL, default: None]
                    Observed data to calibrate against.
                    If FLAG_auto_run=True, data must be provided in the constructor.
                    If FLAG_auto_run=False, data may be passed later to run(data=...).

                epsilon (float) [OPTIONAL, default: 1.0]
                    Tolerance used by PyMC Simulator when comparing simulated data
                    to observed data.
                    Smaller values enforce closer agreement but may make sampling harder.
                    Must be greater than zero.

                sum_stat (str or Callable) [OPTIONAL, default: "sort"]
                    Summary statistic used to compare simulated and observed data.

                    Built-in options:
                        "identity"
                        "sort"
                        "mean"
                        "median"

                    A custom callable may also be provided.

                FLAG_auto_run (bool) [OPTIONAL, default: True]
                    Controls the operating mode. See OPERATING MODES below.

            --------------------------------------------------------
            OPERATING MODES:
            --------------------------------------------------------

                MODE 1 — FLAG_auto_run=True (default)
                    The calibration runs immediately on construction.
                    Data must be supplied in the constructor.
                    No further configuration is possible before sampling starts.

                    Example:
                        calli = ChRBayesCalli(
                            simulate_fn = my_simulator,
                            param_space = {
                                "k": ChRBayesCalli.ChR_Distr.uniform(0, 10),
                                "c": ChRBayesCalli.ChR_Distr.halfnormal(1.0),
                            },
                            data = observed_data,
                        )

                    After running, results are stored on:
                        calli.model
                        calli.params
                        calli.simulator
                        calli.idata

                MODE 2 — FLAG_auto_run=False
                    Construction stops after validation and config report.
                    The user can then manually run calibration later.

                    Example:
                        calli = ChRBayesCalli(
                            simulate_fn   = my_simulator,
                            param_space   = {
                                "k": ChRBayesCalli.ChR_Distr.uniform(0, 10),
                                "c": ChRBayesCalli.ChR_Distr.halfnormal(1.0),
                            },
                            FLAG_auto_run = False,
                        )

                        calli.run(data=observed_data)

                    After running, results are stored on:
                        calli.model
                        calli.params
                        calli.simulator
                        calli.idata

            --------------------------------------------------------
            RESULTS:
            --------------------------------------------------------

                model
                    The PyMC model object created by the engine.

                params
                    Dictionary mapping parameter names to PyMC random variables.

                simulator
                    The PyMC Simulator node used to compare simulations against data.

                idata
                    ArviZ InferenceData object containing posterior samples and
                    posterior predictive samples.

                    Typical usage:
                        az.summary(calli.idata)
                        az.plot_trace(calli.idata)
                        az.plot_posterior(calli.idata)

            --------------------------------------------------------
            IMPORTANT NOTES:
            --------------------------------------------------------

                Parameter order matters.
                    PyMC Simulator receives parameters as tuple(params.values()).
                    Therefore, the insertion order of param_space should match the
                    argument order expected by simulate_fn.

                    Example:
                        def simulate_fn(rng, k, c, size):
                            ...

                        param_space = {
                            "k": ChRBayesCalli.ChR_Distr.uniform(0, 10),
                            "c": ChRBayesCalli.ChR_Distr.halfnormal(1.0),
                        }

                Some distribution helpers use parser hints.
                    Helpers such as loguniform, qloguniform, choice, grid_search,
                    and sample_from require special handling in the model parser.
                    Direct PyMC distributions such as uniform, normal, halfnormal,
                    beta, gamma, exponential, and lognormal are simpler.

                Simulator functions should be lightweight.
                    Avoid rendering, GUI windows, file-heavy side effects, or slow
                    visualization inside simulate_fn during calibration.
        """))
        print("========================================================")

    #================================================================================
    #1. CONSTRUCTOR
    #================================================================================

    def __init__(self, simulate_fn: Callable, param_space: dict, data = None, epsilon = 1.0, sum_stat = "sort", FLAG_auto_run = True):

        #1. mandatory parameters
        self.simulate_fn = simulate_fn
        self.param_space = param_space

        #2. optional parameters 
        self.epsilon = epsilon
        self.sum_stat = sum_stat
        self.data = data
        self.FLAG_auto_run = FLAG_auto_run

        #3. var to save model info for inspection after running 
        self.model = None
        self.params = None 
        self.simulator = None
        self.idata = None

        #4. validate and report configuration
        self._validate_inputs()
        self._report_config()

        #5. immediately run if FLAG_auto_run is True
        if self.FLAG_auto_run and data is not None:
            self.run() 
        elif self.FLAG_auto_run and data is None:
            raise ValueError("Data is required to auto run the engine.")

    #================================================================================
    #2. VALIDATE AND REPORT CONFIGURATION
    #================================================================================
        
    def _validate_inputs(self) -> None:

        print("************************************************************")
        print("Validating inputs...")
        print("run ChRBayesCalli.info() for information on how to use this workflow.")
        print("************************************************************")

        #1. simulate_fn check
        if not callable(self.simulate_fn):
            raise ValueError("simulate_fn must be callable")

        #2. param_space check
        if not isinstance(self.param_space, dict):
            raise TypeError("param_space must be a dict")

        if len(self.param_space) == 0:
            raise ValueError("param_space cannot be empty")

        if not all(isinstance(k, str) for k in self.param_space.keys()):
            raise TypeError("param_space keys must all be strings")

        #3. distribution template flag check
        if not all(
            isinstance(v, dict) and v.get("FLAG_is_pymc_distr_template", False)
            for v in self.param_space.values()
        ):
            raise TypeError(
                "param_space values must all be ChR_Distr distributions. "
                "Use ChRBayesCalli.ChR_Distr.info() for available options and additional info."
            )

        #4. minimum distribution dict structure check
        for curr_var_name, curr_distr_info in self.param_space.items():

            if "kind" not in curr_distr_info:
                raise ValueError(f"param_space[{curr_var_name!r}] is missing required key 'kind'")

            if "dist" not in curr_distr_info:
                raise ValueError(f"param_space[{curr_var_name!r}] is missing required key 'dist'")

            if "kwargs" not in curr_distr_info:
                raise ValueError(f"param_space[{curr_var_name!r}] is missing required key 'kwargs'")

            if not isinstance(curr_distr_info["kind"], str):
                raise TypeError(f"param_space[{curr_var_name!r}]['kind'] must be a string")

            if not isinstance(curr_distr_info["dist"], str):
                raise TypeError(f"param_space[{curr_var_name!r}]['dist'] must be a string")

            if not isinstance(curr_distr_info["kwargs"], dict):
                raise TypeError(f"param_space[{curr_var_name!r}]['kwargs'] must be a dict")

            if not hasattr(pm, curr_distr_info["dist"]) and curr_distr_info["dist"] != "Deterministic":
                raise ValueError(
                    f"PyMC does not have distribution {curr_distr_info['dist']!r} "
                    f"for parameter {curr_var_name!r}"
                )

            #Optional hint checks
            if "log_scale" in curr_distr_info and not isinstance(curr_distr_info["log_scale"], bool):
                raise TypeError(f"param_space[{curr_var_name!r}]['log_scale'] must be a bool")

            if "to_int" in curr_distr_info and not isinstance(curr_distr_info["to_int"], bool):
                raise TypeError(f"param_space[{curr_var_name!r}]['to_int'] must be a bool")

            if "base" in curr_distr_info:
                if not isinstance(curr_distr_info["base"], (int, float)):
                    raise TypeError(f"param_space[{curr_var_name!r}]['base'] must be numeric")
                if curr_distr_info["base"] <= 0 or curr_distr_info["base"] == 1:
                    raise ValueError(f"param_space[{curr_var_name!r}]['base'] must be > 0 and != 1")

            if "round_to" in curr_distr_info:
                if not isinstance(curr_distr_info["round_to"], (int, float)):
                    raise TypeError(f"param_space[{curr_var_name!r}]['round_to'] must be numeric")
                if curr_distr_info["round_to"] <= 0:
                    raise ValueError(f"param_space[{curr_var_name!r}]['round_to'] must be > 0")

            if "categories" in curr_distr_info:
                if not isinstance(curr_distr_info["categories"], list):
                    raise TypeError(f"param_space[{curr_var_name!r}]['categories'] must be a list")
                if len(curr_distr_info["categories"]) == 0:
                    raise ValueError(f"param_space[{curr_var_name!r}]['categories'] cannot be empty")

            if "func" in curr_distr_info and not callable(curr_distr_info["func"]):
                raise TypeError(f"param_space[{curr_var_name!r}]['func'] must be callable")

        #5. epsilon check
        if not isinstance(self.epsilon, (int, float)):
            raise TypeError("epsilon must be an int or float")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")

        #6. sum_stat check
        valid_sum_stats = ("identity", "sort", "mean", "median")

        if not (isinstance(self.sum_stat, str) or callable(self.sum_stat)):
            raise TypeError("sum_stat must be a string or callable")

        if isinstance(self.sum_stat, str) and self.sum_stat not in valid_sum_stats:
            raise ValueError(f"sum_stat must be one of {valid_sum_stats}, got {self.sum_stat!r}")

        #7. data check
        if self.data is not None:
            try:
                len(self.data)
            except TypeError:
                raise TypeError("data must be array-like")

            if len(self.data) == 0:
                raise ValueError("data cannot be empty")

        #8. FLAG_auto_run check
        if not isinstance(self.FLAG_auto_run, bool):
            raise TypeError("FLAG_auto_run must be a bool")

    def _report_config(self):
        print("************************************************************")
        print("========================================================")
        print("ChRBayesCalli (\"Bayesian Callibration Engine for ChronoRay\") configuration:")
        print("========================================================")
        print(f"simulate_fn: {self.simulate_fn}")
        print(f"param_space:")
        for curr_var_name, curr_distr_info in self.param_space.items():
            print(f"    {curr_var_name}: {curr_distr_info['dist']} with kwargs: {curr_distr_info['kwargs']}")
        if self.data is not None:
            print(f"data: Properly initialized in constructor")
        else: 
            print(f"data: not yet initialized, must pass data to run() [DO NOT AUTO RUN THIS CONFIG]")
        print(f"epsilon: {self.epsilon}")
        print(f"sum_stat: {self.sum_stat}")
        print(f"FLAG_auto_run: {self.FLAG_auto_run}")
        print("************************************************************")

    #================================================================================
    #3. RUN THE ENGINE
    #================================================================================

    def _build_param(self, curr_var_name: str, curr_distr_info: dict, params: dict):

        #1. unpack distribution info
        kind = curr_distr_info["kind"]
        dist_name = curr_distr_info["dist"]
        kwargs = curr_distr_info["kwargs"]

        #2. handle dependent / deterministic parameters
        if kind == "sample_from":

            if "func" not in curr_distr_info:
                raise ValueError(f"sample_from distribution for {curr_var_name!r} requires 'func'")

            return pm.Deterministic(
                curr_var_name,
                curr_distr_info["func"](params)
            )

        #3. handle categorical distributions
        if kind in ("choice", "grid_search"):

            if "categories" not in curr_distr_info:
                raise ValueError(f"{kind} distribution for {curr_var_name!r} requires 'categories'")

            categories = curr_distr_info["categories"]
            n_categories = len(categories)

            return pm.Categorical(
                curr_var_name,
                p=np.ones(n_categories) / n_categories
            )

        #4. handle log-scale distributions
        if curr_distr_info.get("log_scale", False):

            lower = kwargs["lower"]
            upper = kwargs["upper"]
            base = curr_distr_info.get("base", 10)

            log_lower = np.log(lower) / np.log(base)
            log_upper = np.log(upper) / np.log(base)

            raw_param = pm.Uniform(
                f"{curr_var_name}_log",
                lower=log_lower,
                upper=log_upper
            )

            param_value = base ** raw_param

            if "round_to" in curr_distr_info:
                q = curr_distr_info["round_to"]
                param_value = pm.math.round(param_value / q) * q

            if curr_distr_info.get("to_int", False):
                param_value = pm.math.round(param_value)

            return pm.Deterministic(
                curr_var_name,
                param_value
            )

        #5. handle standard PyMC distributions
        dist_fn = getattr(pm, dist_name)

        param_value = dist_fn(
            curr_var_name,
            **kwargs
        )

        #6. handle quantized non-log distributions
        if "round_to" in curr_distr_info:

            q = curr_distr_info["round_to"]

            param_value = pm.Deterministic(
                curr_var_name,
                pm.math.round(param_value / q) * q
            )

        return param_value

    def run(self, data=None):

        #0. ensure data is properly set 
        if self.data is None and data is None:
            raise ValueError("Data is required to run the engine.")
        elif self.data is None and data is not None:
            self.data = data

        #1. build the model
        model = pm.Model()
        params = {}

        with model:

            for curr_var_name, curr_distr_info in self.param_space.items():

                params[curr_var_name] = self._build_param(
                    curr_var_name,
                    curr_distr_info,
                    params
                )

            s = pm.Simulator(
                "s",
                self.simulate_fn,
                params=tuple(params.values()),
                sum_stat=self.sum_stat,
                epsilon=self.epsilon,
                observed=self.data
            )

            idata = pm.sample_smc()

            posterior_predictive = pm.sample_posterior_predictive(idata)

            if hasattr(idata, "extend"):
                idata.extend(posterior_predictive)
            else:
                idata["posterior_predictive"] = posterior_predictive["posterior_predictive"]
                idata["observed_data"] = posterior_predictive["observed_data"]

        #2. store model info for inspection after running
        self.model = model
        self.params = params
        self.simulator = s
        self.idata = idata

        #3. return inference data
        return idata
    

def _distr_template(kind: str, dist: str, kwargs: dict, **hints) -> dict:
    """Assemble a standard distribution-description dict."""
    return {
        "FLAG_is_pymc_distr_template": True,
        "kind": kind,
        "dist": dist,
        "kwargs": kwargs,
        **hints,
    }       

#================================================================================
# 2. DISTRIBUTIONS
#================================================================================

"""
A thin, self-contained bridge to PyMC prior distributions for Bayesian calibration.

All methods are static — call them directly through ChRBayesCalli.ChR_Distr,
with no instantiation needed.

Each method returns a plain dictionary describing a prior distribution.
These dictionaries do NOT create PyMC variables immediately. They are parsed
later by ChRBayesCalli inside a `with pm.Model():` block.

Every distribution dict has the shape:
    {
        "FLAG_is_pymc_distr_template": True,
        "kind": <str>,      # user-facing prior name, e.g. "uniform"
        "dist": <str>,      # PyMC distribution name, e.g. "Uniform"
        "kwargs": {...},    # keyword arguments for the PyMC distribution
        ...                 # optional parser hints
    }

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
grid_search     | (values)                          | uniform prior over listed values
quniform        | (lower, upper, q)                 | continuous rounded to nearest q
qloguniform     | (lower, upper, q, base=10)        | log-scale rounded to nearest q
qrandn          | (mean, sd, q)                     | Gaussian rounded to nearest q
qrandint        | (lower, upper, q=1)               | integer rounded to nearest q
qlograndint     | (lower, upper, q, base=10)        | log-scale integer rounded to q
sample_from     | (func: params -> expression)      | dependent / deterministic prior

Extra Bayesian calibration priors
---------------------------------
normal          | (mu, sigma)                       | conventional Normal prior
halfnormal      | (sigma)                           | positive scale / magnitude
truncatednormal | (mu, sigma, lower, upper)         | bounded Gaussian prior
lognormal       | (mu, sigma)                       | positive, right-skewed prior
beta            | (alpha, beta)                     | fractions / probabilities
gamma           | (alpha, beta)                     | positive continuous values
halfcauchy      | (beta)                            | weakly informative scale prior
exponential     | (lam)                             | positive decay / waiting values

Parser hints
------------
Some methods add extra keys that the ChRBayesCalli parser must handle:

    log_scale=True     build the prior in log space
    to_int=True        convert sampled value to integer
    round_to=q         round sampled value to nearest multiple of q
    categories=[...]   map categorical index back to listed values
    func=...           build a deterministic/dependent expression

Example
-------
    param_space = {
        "spring_k"  : ChRBayesCalli.ChR_Distr.loguniform(1.0, 1000.0),
        "damper_c"  : ChRBayesCalli.ChR_Distr.uniform(0.1, 50.0),
        "mass"      : ChRBayesCalli.ChR_Distr.truncatednormal(
                          1.0, 0.5, lower=0.0, upper=5.0
                      ),
        "noise_sd"  : ChRBayesCalli.ChR_Distr.halfnormal(1.0),
    }
"""

class PyMC_Distr_template:

    FLAG_is_pymc_distr_template = True

    # ------------------------------------------------------------------ #
    # Direct analogs of the Ray Tune distributions
    # ------------------------------------------------------------------ #

    @staticmethod
    def uniform(lower: float, upper: float) -> dict:
        """
        Uniform prior on [lower, upper] (linear scale).
        Good for: stiffness, damping, mass, geometry with no preferred sub-range.
        """
        return _distr_template("uniform", "Uniform", {"lower": lower, "upper": upper})

    @staticmethod
    def loguniform(lower: float, upper: float, base: float = 10) -> dict:
        """
        Log-uniform prior on [lower, upper]: flat in log space.
        PyMC has no native log-uniform; the hint `log_scale=True` tells the
        parser to build it (typically as exp of a Uniform over the log range).
        `base` is kept for API parity but does not change a continuous range.
        Good for: parameters spanning several orders of magnitude.
        """
        return _distr_template(
            "loguniform", "Uniform",
            {"lower": lower, "upper": upper},
            log_scale=True, base=base,
        )

    @staticmethod
    def randn(mean: float, sd: float) -> dict:
        """
        Gaussian prior N(mean, sd). Unbounded; use `truncatednormal` for bounds.
        Good for: parameters with a strong prior expectation to search around.
        """
        return _distr_template("randn", "Normal", {"mu": mean, "sigma": sd})

    @staticmethod
    def randint(lower: int, upper: int) -> dict:
        """
        Uniform integer prior on [lower, upper) (upper EXCLUSIVE, matching Ray).
        PyMC's DiscreteUniform is inclusive, so kwargs use upper - 1.
        Good for: number of links, solver iterations, polynomial degree.
        """
        return _distr_template("randint", "DiscreteUniform", {"lower": lower, "upper": upper - 1})

    @staticmethod
    def lograndint(lower: int, upper: int, base: float = 10) -> dict:
        """
        Log-scaled integer prior on [lower, upper). Hints `log_scale=True` and
        `to_int=True` tell the parser to build a log-uniform and round to int.
        Good for: integer parameters spanning orders of magnitude.
        """
        return _distr_template(
            "lograndint", "Uniform",
            {"lower": lower, "upper": upper},
            log_scale=True, base=base, to_int=True,
        )

    @staticmethod
    def choice(categories: Sequence) -> dict:
        """
        Uniform prior over a fixed list of options (Categorical).
        The `categories` key holds the list; the parser maps the sampled index
        back to the value (numeric) or leaves it as an index (non-numeric).
        Good for: solver type, contact model, material preset, discrete options.
        """
        return _distr_template("choice", "Categorical", {}, categories=list(categories))

    @staticmethod
    def grid_search(values: Iterable) -> dict:
        """
        NOTE: grid search is an optimization concept with no Bayesian-prior
        equivalent. The closest analog is a uniform prior over exactly these
        values (same as `choice`); it does NOT cause exhaustive evaluation.
        Good for: a small set of candidate values weighted equally a priori.
        """
        return _distr_template("grid_search", "Categorical", {}, categories=list(values))

    # ------------------------------------------------------------------ #
    # Quantized variants
    # ------------------------------------------------------------------ #

    @staticmethod
    def quniform(lower: float, upper: float, q: float) -> dict:
        """
        Uniform on [lower, upper], rounded to the nearest multiple of q
        (hint `round_to=q`).
        Good for: step sizes or ratios that must be multiples of a fixed value.
        """
        return _distr_template(
            "quniform", "Uniform",
            {"lower": lower, "upper": upper},
            round_to=q,
        )

    @staticmethod
    def qloguniform(lower: float, upper: float, q: float, base: float = 10) -> dict:
        """
        Log-uniform on [lower, upper], rounded to the nearest multiple of q.
        Good for: tolerances or rates needing a discrete grid on a log scale.
        """
        return _distr_template(
            "qloguniform", "Uniform",
            {"lower": lower, "upper": upper},
            log_scale=True, base=base, round_to=q,
        )

    @staticmethod
    def qrandn(mean: float, sd: float, q: float) -> dict:
        """
        Normal N(mean, sd), rounded to the nearest multiple of q.
        Good for: Gaussian-prior parameters that must land on a discrete grid.
        """
        return _distr_template("qrandn", "Normal", {"mu": mean, "sigma": sd}, round_to=q)

    @staticmethod
    def qrandint(lower: int, upper: int, q: int = 1) -> dict:
        """
        Integer on [lower, upper) (upper exclusive), rounded to nearest multiple
        of q.
        Good for: counts that must be multiples of a base value.
        """
        return _distr_template(
            "qrandint", "DiscreteUniform",
            {"lower": lower, "upper": upper - 1},
            round_to=q,
        )

    @staticmethod
    def qlograndint(lower: int, upper: int, q: int, base: float = 10) -> dict:
        """
        Log-scaled integer on [lower, upper), rounded to nearest multiple of q.
        Good for: large integer ranges on a log grid.
        """
        return _distr_template(
            "qlograndint", "Uniform",
            {"lower": lower, "upper": upper},
            log_scale=True, base=base, to_int=True, round_to=q,
        )

    # ------------------------------------------------------------------ #
    # Dependent / custom prior
    # ------------------------------------------------------------------ #

    @staticmethod
    def sample_from(func: Callable[[dict], Any]) -> dict:
        """
        Dependent prior defined in terms of other already-built parameters.
        Unlike Ray's sample_from (which draws its own random numbers), `func`
        here should return a PyTensor expression built from other params, since
        a Bayesian prior's randomness must come from the prior variables.
        The callable is stored under `func` for the parser.

        Example: PyMC_Distr_template.sample_from(lambda b: b["spring_k"] * 0.5)
        """
        return _distr_template("sample_from", "Deterministic", {}, func=func)

    # ------------------------------------------------------------------ #
    # Extra priors that are staples of Bayesian calibration
    # ------------------------------------------------------------------ #

    @staticmethod
    def normal(mu: float, sigma: float) -> dict:
        """Normal N(mu, sigma). Alias of `randn` with conventional naming."""
        return _distr_template("normal", "Normal", {"mu": mu, "sigma": sigma})

    @staticmethod
    def halfnormal(sigma: float) -> dict:
        """
        HalfNormal(sigma): a Normal folded to be positive.
        Good for: standard deviations, noise scales, positive magnitudes.
        """
        return _distr_template("halfnormal", "HalfNormal", {"sigma": sigma})

    @staticmethod
    def truncatednormal(mu: float, sigma: float,
                        lower: float = None, upper: float = None) -> dict:
        """
        TruncatedNormal: a Gaussian prior with hard bounds. The principled fix
        for `randn` when a parameter must stay within limits.
        Good for: a best-guess value with uncertainty that is also bounded.
        """
        return _distr_template(
            "truncatednormal", "TruncatedNormal",
            {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper},
        )

    @staticmethod
    def lognormal(mu: float, sigma: float) -> dict:
        """
        LogNormal(mu, sigma): positive, right-skewed (mu/sigma are on log scale).
        Good for: positive multiplicative quantities (rates, conductivities).
        """
        return _distr_template("lognormal", "LogNormal", {"mu": mu, "sigma": sigma})

    @staticmethod
    def beta(alpha: float, beta: float) -> dict:
        """
        Beta(alpha, beta): support on [0, 1].
        Good for: fractions, efficiencies, probabilities, mixing ratios.
        """
        return _distr_template("beta", "Beta", {"alpha": alpha, "beta": beta})

    @staticmethod
    def gamma(alpha: float, beta: float) -> dict:
        """
        Gamma(alpha, beta): positive (`beta` is the rate).
        Good for: positive quantities like waiting times or positive scales.
        """
        return _distr_template("gamma", "Gamma", {"alpha": alpha, "beta": beta})

    @staticmethod
    def halfcauchy(beta: float) -> dict:
        """
        HalfCauchy(beta): positive, heavy-tailed.
        Good for: weakly-informative priors on scale parameters.
        """
        return _distr_template("halfcauchy", "HalfCauchy", {"beta": beta})

    @staticmethod
    def exponential(lam: float) -> dict:
        """
        Exponential(lam): positive, rate `lam` (mean = 1 / lam).
        Good for: positive quantities with a simple one-parameter decay prior.
        """
        return _distr_template("exponential", "Exponential", {"lam": lam})

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def info() -> None:
        import inspect
        print("==================================================================================")
        print("Available ChR_Distr (for Bayesian Callibration Enginer) sampling distributionss:")
        print("=================================================================================")
        for name, method in inspect.getmembers(PyMC_Distr_template, predicate=callable):
            if not name.startswith("_") and name != "info":
                doc = inspect.getdoc(method)
                print(f"\n  {name}")
                print(f"    {doc}")
        print("=================================================================================")



# Attach distribution helpers to main user-facing class

ChRBayesCalli.uniform = staticmethod(PyMC_Distr_template.uniform)
ChRBayesCalli.loguniform = staticmethod(PyMC_Distr_template.loguniform)
ChRBayesCalli.randn = staticmethod(PyMC_Distr_template.randn)
ChRBayesCalli.randint = staticmethod(PyMC_Distr_template.randint)
ChRBayesCalli.lograndint = staticmethod(PyMC_Distr_template.lograndint)
ChRBayesCalli.choice = staticmethod(PyMC_Distr_template.choice)
ChRBayesCalli.grid_search = staticmethod(PyMC_Distr_template.grid_search)

ChRBayesCalli.quniform = staticmethod(PyMC_Distr_template.quniform)
ChRBayesCalli.qloguniform = staticmethod(PyMC_Distr_template.qloguniform)
ChRBayesCalli.qrandn = staticmethod(PyMC_Distr_template.qrandn)
ChRBayesCalli.qrandint = staticmethod(PyMC_Distr_template.qrandint)
ChRBayesCalli.qlograndint = staticmethod(PyMC_Distr_template.qlograndint)

ChRBayesCalli.sample_from = staticmethod(PyMC_Distr_template.sample_from)

ChRBayesCalli.normal = staticmethod(PyMC_Distr_template.normal)
ChRBayesCalli.halfnormal = staticmethod(PyMC_Distr_template.halfnormal)
ChRBayesCalli.truncatednormal = staticmethod(PyMC_Distr_template.truncatednormal)
ChRBayesCalli.lognormal = staticmethod(PyMC_Distr_template.lognormal)
ChRBayesCalli.beta = staticmethod(PyMC_Distr_template.beta)
ChRBayesCalli.gamma = staticmethod(PyMC_Distr_template.gamma)
ChRBayesCalli.halfcauchy = staticmethod(PyMC_Distr_template.halfcauchy)
ChRBayesCalli.exponential = staticmethod(PyMC_Distr_template.exponential)

ChRBayesCalli.ChR_Distr = PyMC_Distr_template