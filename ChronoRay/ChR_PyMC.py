import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

"""
A thin, self-contained bridge to PyMC prior distributions, mirroring the
ChR_Distr_template (Ray Tune) interface as closely as PyMC allows.

Each method returns a plain dict describing one PyMC prior. The dict carries
everything needed to build the real PyMC variable later (the PyMC distribution
class name plus its keyword arguments), but does NOT build it. Parsing the dicts
into actual variables inside a `with pm.Model():` block is handled elsewhere.

Every dict has the shape:
    {
        "FLAG_is_pymc_distr_template": True,
        "kind": <str>,      # the ChR-style name used here, e.g. "randn"
        "dist": <str>,      # PyMC distribution class, e.g. "Normal"
        "kwargs": {...},    # keyword args for that distribution
        ...                 # extra hints for non-trivial cases (see below)
    }

Quick-reference (Ray name -> PyMC prior)
----------------------------------------
Method          | Signature                          | PyMC dist
----------------|------------------------------------|----------------------------
uniform         | (lower, upper)                     | Uniform
loguniform      | (lower, upper, base=10)            | (log-uniform, see hint)
randn           | (mean, sd)                         | Normal
randint         | (lower, upper)                     | DiscreteUniform [lower, upper)
lograndint      | (lower, upper, base=10)            | (log-int, see hint)
choice          | (categories)                       | Categorical
grid_search     | (values)                           | Categorical over values*
quniform        | (lower, upper, q)                  | Uniform + round-to-q hint
qloguniform     | (lower, upper, q, base=10)         | log-uniform + round-to-q hint
qrandn          | (mean, sd, q)                      | Normal + round-to-q hint
qrandint        | (lower, upper, q=1)                | DiscreteUniform + round-to-q hint
qlograndint     | (lower, upper, q, base=10)         | log-int + round-to-q hint
sample_from     | (func)                             | dependent prior*

Extra priors common in Bayesian calibration (no Ray equivalent)
---------------------------------------------------------------
normal          | (mu, sigma)                        | Normal (alias of randn)
halfnormal      | (sigma)                            | HalfNormal
truncatednormal | (mu, sigma, lower, upper)          | TruncatedNormal
lognormal       | (mu, sigma)                        | LogNormal
beta            | (alpha, beta)                      | Beta
gamma           | (alpha, beta)                      | Gamma
halfcauchy      | (beta)                             | HalfCauchy
exponential     | (lam)                              | Exponential

* grid_search and sample_from have no literal Bayesian equivalent; see their
  docstrings. Hints (log_scale, round_to, to_int, categories, func) live as
  extra keys on the dict for the later parser to act on.

Example
-------
    param_space = {
        "spring_k" : PyMC_Distr_template.loguniform(1.0, 1000.0),
        "damper_c" : PyMC_Distr_template.uniform(0.1, 50.0),
        "mass"     : PyMC_Distr_template.truncatednormal(1.0, 0.5, lower=0.0, upper=5.0),
        "num_links": PyMC_Distr_template.randint(1, 10),
    }
    # param_space["damper_c"] ==
    #   {"FLAG_is_pymc_distr_template": True, "kind": "uniform", "dist": "Uniform",
    #    "kwargs": {"lower": 0.1, "upper": 50.0}}
"""

from typing import Any, Callable, Iterable, Sequence


def _distr_template(kind: str, dist: str, kwargs: dict, **hints) -> dict:
    """Assemble a standard distribution-description dict."""
    return {
        "FLAG_is_pymc_distr_template": True,
        "kind": kind,
        "dist": dist,
        "kwargs": kwargs,
        **hints,
    }


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
        print("========================================================")
        print("Available PyMC_Distr_template priors:")
        print("========================================================")
        for name, method in inspect.getmembers(PyMC_Distr_template, predicate=callable):
            if not name.startswith("_") and name != "info":
                doc = inspect.getdoc(method)
                print(f"\n  {name}")
                print(f"    {doc}")
        print("========================================================")