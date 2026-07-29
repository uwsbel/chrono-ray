"""
PyMC-backed Bayesian calibration workflow for ChronoRay.

The public API intentionally preserves the existing usage pattern:

    param_prior_space = {
        "mu_s": ChRBayesCali.ChR_Distr.uniform(0.5, 0.8),
        "young_modulus": ChRBayesCali.ChR_Distr.loguniform(5e6, 8e6),
    }

    cal = ChRBayesCali(
        simulate_fn=simulate_fn,
        objective_fn=objective_fn,
        param_prior_space=param_prior_space,
        sigma=1.0,
        total_samples=2000,
        burn_in_frac=0.3,
    )

    posterior = cal.get_posterior()

Internally, PyMC owns the priors, Metropolis sampler, chain state, tuning,
acceptance/rejection, and posterior storage. The user's simulator and objective
are exposed to PyMC through a black-box PyTensor Op.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
import textwrap
from collections.abc import Callable, Iterator
from typing import Any, Literal
import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op

from ChronoRay.ChRCrashProtection import is_crash


PriorKind = Literal["uniform", "loguniform", "normal", "lognormal"]


@dataclass(frozen=True)
class _ChRPrior:
    """
    Deferred prior specification.

    ChR_Distr methods return these lightweight objects outside a PyMC model.
    ChRBayesCali later converts them into named PyMC variables inside the
    correct ``with pm.Model():`` context.
    """

    kind: PriorKind
    parameters: dict[str, float]
    FLAG_is_chr_distr: bool = True

    def build(self, name: str):
        """Create the corresponding named PyMC variable."""
        if not isinstance(name, str) or not name:
            raise ValueError("Prior name must be a non-empty string")

        if self.kind == "uniform":
            return pm.Uniform(
                name,
                lower=self.parameters["lower"],
                upper=self.parameters["upper"],
            )

        if self.kind == "normal":
            return pm.Normal(
                name,
                mu=self.parameters["mu"],
                sigma=self.parameters["sigma"],
            )

        if self.kind == "lognormal":
            return pm.LogNormal(
                name,
                mu=self.parameters["mu"],
                sigma=self.parameters["sigma"],
            )

        if self.kind == "loguniform":
            lower = self.parameters["lower"]
            upper = self.parameters["upper"]

            # PyMC does not need a custom log-uniform implementation. Sampling
            # uniformly in log(x), then exponentiating, gives p(x) proportional
            # to 1/x over [lower, upper].
            log_rv = pm.Uniform(
                f"__chr_log_{name}",
                lower=math.log(lower),
                upper=math.log(upper),
            )
            return pm.Deterministic(name, pt.exp(log_rv))

        raise ValueError(f"Unsupported prior kind: {self.kind!r}")


class ChR_Distr:
    """ChronoRay's stable, user-facing prior-distribution factory."""

    @staticmethod
    def uniform(lower: float, upper: float) -> _ChRPrior:
        lower, upper = ChR_Distr._validate_bounds(
            lower,
            upper,
            positive=False,
            label="uniform",
        )
        return _ChRPrior(
            kind="uniform",
            parameters={"lower": lower, "upper": upper},
        )

    @staticmethod
    def loguniform(lower: float, upper: float) -> _ChRPrior:
        lower, upper = ChR_Distr._validate_bounds(
            lower,
            upper,
            positive=True,
            label="loguniform",
        )
        return _ChRPrior(
            kind="loguniform",
            parameters={"lower": lower, "upper": upper},
        )

    @staticmethod
    def normal(mu: float, sigma: float) -> _ChRPrior:
        mu = ChR_Distr._finite_float(mu, "mu")
        sigma = ChR_Distr._positive_float(sigma, "sigma")
        return _ChRPrior(
            kind="normal",
            parameters={"mu": mu, "sigma": sigma},
        )

    @staticmethod
    def lognormal(mu: float, sigma: float) -> _ChRPrior:
        mu = ChR_Distr._finite_float(mu, "mu")
        sigma = ChR_Distr._positive_float(sigma, "sigma")
        return _ChRPrior(
            kind="lognormal",
            parameters={"mu": mu, "sigma": sigma},
        )

    @staticmethod
    def info() -> None:
        print(
            textwrap.dedent(
                """
                ChR_Distr prior factories
                -------------------------
                uniform(lower, upper)
                    Uniform prior on the stated interval.

                loguniform(lower, upper)
                    Log-uniform prior on a strictly positive interval.

                normal(mu, sigma)
                    Normal prior with mean mu and standard deviation sigma.

                lognormal(mu, sigma)
                    Log-normal prior where log(parameter) ~ Normal(mu, sigma).

                Example
                -------
                priors = {
                    "mu_s": ChRBayesCali.ChR_Distr.uniform(0.5, 0.8),
                    "E": ChRBayesCali.ChR_Distr.loguniform(5e6, 8e6),
                }
                """
            ).strip()
        )

    @staticmethod
    def _format_distr(prior: _ChRPrior) -> str:
        args = ", ".join(
            f"{key}={value:g}" for key, value in prior.parameters.items()
        )
        return f"{prior.kind}({args})"

    @staticmethod
    def _finite_float(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool")

        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric, got {value!r}") from exc

        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite, got {value!r}")

        return result

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        result = ChR_Distr._finite_float(value, name)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}")
        return result

    @staticmethod
    def _validate_bounds(
        lower: Any,
        upper: Any,
        *,
        positive: bool,
        label: str,
    ) -> tuple[float, float]:
        lower_f = ChR_Distr._finite_float(lower, "lower")
        upper_f = ChR_Distr._finite_float(upper, "upper")

        if positive and lower_f <= 0.0:
            raise ValueError(
                f"{label} requires a strictly positive lower bound; "
                f"got {lower!r}"
            )

        if lower_f >= upper_f:
            raise ValueError(
                f"{label} requires lower < upper; "
                f"got lower={lower_f!r}, upper={upper_f!r}"
            )

        return lower_f, upper_f


class _SimulationLogLikelihoodOp(Op):
    """
    PyTensor bridge for an arbitrary Python simulation and objective function.

    Inputs are scalar parameter values in a stable order. The output is one
    scalar log-likelihood:

        log_likelihood = -misfit / (2 * sigma**2)

    A crash-protected Bayesian-calibration workflow must use mode="min":
    +inf is recognized as the crash sentinel and mapped to -inf log likelihood.
    NaN and -inf are treated as invalid objective outputs rather than crashes.

    The Op deliberately exposes no gradient. ChRBayesCali therefore selects
    PyMC's established Metropolis step method rather than NUTS.
    """

    __props__ = ("parameter_names", "sigma")

    def __init__(
        self,
        *,
        parameter_names: tuple[str, ...],
        simulate_fn: Callable[[dict[str, float]], Any],
        objective_fn: Callable[[Any], float],
        sigma: float,
    ) -> None:
        self.parameter_names = parameter_names
        self.simulate_fn = simulate_fn
        self.objective_fn = objective_fn
        self.sigma = float(sigma)

    def make_node(self, *theta) -> Apply:
        if len(theta) != len(self.parameter_names):
            raise ValueError(
                f"Expected {len(self.parameter_names)} parameters, "
                f"received {len(theta)}"
            )

        inputs = [pt.as_tensor_variable(value) for value in theta]

        for value in inputs:
            if value.ndim != 0:
                raise TypeError(
                    "ChRBayesCali currently supports scalar calibration "
                    "parameters only"
                )

        return Apply(self, inputs, [pt.dscalar()])

    def perform(self, node, inputs, outputs) -> None:
        del node

        try:
            config = {
                name: float(value)
                for name, value in zip(
                    self.parameter_names,
                    inputs,
                    strict=True,
                )
            }

            simulation_output = self.simulate_fn(config)
            misfit = float(self.objective_fn(simulation_output))

            # Bayesian calibration receives a data misfit, so lower is better.
            # ChRCrashProtection must therefore be configured with mode="min",
            # whose crash sentinel is +inf. Only that specific infinity means
            # "simulation failed" and should become a normal MCMC rejection.
            if is_crash(misfit, mode="min"):
                log_likelihood = -np.inf
            elif not np.isfinite(misfit):
                raise ValueError(
                    "objective_fn returned an invalid non-finite misfit: "
                    f"{misfit!r}. For Bayesian calibration, protected crashes "
                    "must use mode='min' so the sentinel is +inf."
                )
            else:
                log_likelihood = -misfit / (2.0 * self.sigma**2)

        except BaseException as exc:
            # A failed simulator call represents an inadmissible proposal for
            # the sampler. Do not terminate the full calibration run.
            print(
                "[ChRBayesCali] simulation/objective failure; "
                f"proposal rejected. Error: {exc}",
                file=sys.stderr,
                flush=True,
            )
            log_likelihood = -np.inf

        outputs[0][0] = np.asarray(log_likelihood, dtype=np.float64)

    def infer_shape(self, fgraph, node, input_shapes):
        del fgraph, node, input_shapes
        return [()]


class ChRBayesCali:
    """
    Bayesian calibration façade backed by PyMC.

    The public constructor intentionally mirrors the former Ray/Tune-backed
    implementation. ``simulate_fn`` receives a parameter dictionary and
    ``objective_fn`` returns a scalar data misfit. PyMC supplies the priors,
    Metropolis implementation, tuning, chain storage, and posterior trace.
    """

    ChR_Distr = ChR_Distr

    @staticmethod
    def info() -> None:
        print("========================================================")
        print("ChRBayesCali - PyMC-backed Bayesian Calibration Workflow")
        print("========================================================")
        print(
            textwrap.dedent(
                """
                DESCRIPTION
                    Calibrates an arbitrary simulation against observed data and
                    returns a posterior distribution over its parameters.

                    The public workflow remains:

                        config -> simulate_fn -> output -> objective_fn -> misfit

                    Internally, PyMC evaluates:

                        log posterior = log prior - misfit / (2 * sigma**2)

                    PyMC's Metropolis implementation owns proposal generation,
                    tuning, acceptance/rejection, chain state, and sample storage.

                REQUIRED ARGUMENTS
                    simulate_fn(config)
                        Runs one simulation. ``config`` is a dictionary mapping
                        parameter names to scalar numerical values.

                    objective_fn(output)
                        Compares simulation output with observations and returns
                        one finite scalar data misfit. Lower is better. Do not add
                        prior penalties here.

                    param_prior_space
                        Dictionary of parameter names to
                        ``ChRBayesCali.ChR_Distr`` prior specifications.

                OPTIONAL ARGUMENTS
                    sigma=1.0
                        Converts data misfit to log likelihood using
                        -misfit / (2*sigma**2).

                    total_samples=2000
                        Total per-chain sampling budget, including tuning.

                    burn_in_frac=0.3
                        Fraction of total_samples used for PyMC tuning and then
                        discarded.

                    chains=1
                        Number of independent PyMC chains.

                    cores=1
                        Number of processes used for chains. Keep at 1 when the
                        simulator is not safe to execute concurrently.

                    random_seed=0
                        Sampling seed.

                    FLAG_log_to_file=False
                        Redirect console output to a timestamped text file.

                    FLAG_auto_run=True
                        Build and run immediately during construction.

                RETURN VALUE
                    get_posterior() returns:

                        list[dict[str, float]]

                    This preserves compatibility with:

                        pd.DataFrame(cal.get_posterior())
                """
            )
        )
        print("========================================================")

    def __init__(
        self,
        simulate_fn: Callable,
        objective_fn: Callable,
        param_prior_space: dict[str, _ChRPrior],
        sigma: float = 1.0,
        total_samples: int = 2000,
        burn_in_frac: float = 0.3,
        FLAG_log_to_file: bool = False,
        FLAG_auto_run: bool = True,
        *,
        chains: int = 1,
        cores: int = 1,
        random_seed: int | None = 0,
        progressbar: bool = True,
    ) -> None:
        self.simulate_fn = simulate_fn
        self.objective_fn = objective_fn
        self.param_prior_space = param_prior_space

        self.sigma = sigma
        self.total_samples = total_samples
        self.burn_in_frac = burn_in_frac
        self.chains = chains
        self.cores = cores
        self.random_seed = random_seed
        self.progressbar = progressbar

        self.FLAG_log_to_file = FLAG_log_to_file
        self.FLAG_auto_run = FLAG_auto_run

        #TODO probably important when we scale up to cluster 
        self.resources_per_trial = {"cpu": 1, "gpu": 0}

        self.model: pm.Model | None = None          #PyMC engine 
        self.idata: Any | None = None               #the ArviZ obj (ret by PyMC sampling) that holds all the important info and such
        self.log_file_path: Path | None = None

        self._parameter_names = tuple(param_prior_space.keys())
        self._parameter_variables: dict[str, Any] = {}
        self._loglike_op: _SimulationLogLikelihoodOp | None = None
        self._has_run = False

        self._validate_inputs()
        self._report_config()

        if FLAG_auto_run:
            self._build_model()
            self.run()

    def _validate_inputs(self) -> None:
        if not callable(self.simulate_fn):
            raise TypeError("simulate_fn must be callable")

        if not callable(self.objective_fn):
            raise TypeError("objective_fn must be callable")

        if not isinstance(self.param_prior_space, dict):
            raise TypeError("param_prior_space must be a dict")

        if not self.param_prior_space:
            raise ValueError("param_prior_space cannot be empty")

        if not all(
            isinstance(name, str) and name
            for name in self.param_prior_space
        ):
            raise TypeError(
                "param_prior_space keys must be non-empty strings"
            )

        if not all(
            isinstance(prior, _ChRPrior)
            and prior.FLAG_is_chr_distr
            for prior in self.param_prior_space.values()
        ):
            raise TypeError(
                "param_prior_space values must be created with "
                "ChRBayesCali.ChR_Distr"
            )

        self.sigma = ChR_Distr._positive_float(self.sigma, "sigma")

        if (
            isinstance(self.total_samples, bool)
            or not isinstance(self.total_samples, int)
            or self.total_samples <= 0
        ):
            raise ValueError(
                "total_samples must be a positive integer, "
                f"got {self.total_samples!r}"
            )

        if (
            isinstance(self.burn_in_frac, bool)
            or not isinstance(self.burn_in_frac, (int, float))
            or not 0.0 <= float(self.burn_in_frac) < 1.0
        ):
            raise ValueError(
                "burn_in_frac must be in [0.0, 1.0), "
                f"got {self.burn_in_frac!r}"
            )

        self.burn_in_frac = float(self.burn_in_frac)

        if (
            isinstance(self.chains, bool)
            or not isinstance(self.chains, int)
            or self.chains <= 0
        ):
            raise ValueError(
                f"chains must be a positive integer, got {self.chains!r}"
            )

        if (
            isinstance(self.cores, bool)
            or not isinstance(self.cores, int)
            or self.cores <= 0
        ):
            raise ValueError(
                f"cores must be a positive integer, got {self.cores!r}"
            )

        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
        ):
            raise TypeError(
                "random_seed must be an integer or None, "
                f"got {self.random_seed!r}"
            )

        if not isinstance(self.FLAG_log_to_file, bool):
            raise TypeError("FLAG_log_to_file must be bool")

        if not isinstance(self.FLAG_auto_run, bool):
            raise TypeError("FLAG_auto_run must be bool")

        if not isinstance(self.progressbar, bool):
            raise TypeError("progressbar must be bool")

        if self._draws < 1:
            raise ValueError(
                "burn_in_frac leaves no retained posterior draws. "
                "Increase total_samples or reduce burn_in_frac."
            )

    @property
    def _tune(self) -> int:
        return int(self.total_samples * self.burn_in_frac)

    @property
    def _draws(self) -> int:
        return self.total_samples - self._tune

    def _report_config(self) -> None:
        print("************************************************************")
        print('ChRBayesCali ("PyMC Bayesian Calibration Workflow")')
        print("************************************************************")
        print(f"  simulate_fn       : {self._callable_name(self.simulate_fn)}")
        print(f"  objective_fn      : {self._callable_name(self.objective_fn)}")
        print("  param_prior_space :")

        for name, prior in self.param_prior_space.items():
            print(f"    {name}: {ChR_Distr._format_distr(prior)}")

        print(f"  sigma             : {self.sigma}")
        print(f"  total_samples     : {self.total_samples}")
        print(f"  tune/burn-in      : {self._tune}")
        print(f"  retained draws    : {self._draws}")
        print(f"  chains            : {self.chains}")
        print(f"  cores             : {self.cores}")
        print("  sampler           : PyMC Metropolis")
        print(f"  FLAG_log_to_file  : {self.FLAG_log_to_file}")
        print("************************************************************")

    @staticmethod
    def _callable_name(fn: Callable) -> str:
        return getattr(fn, "__name__", fn.__class__.__name__)

    def set_resources_per_trial(self, cpu: int, gpu: int) -> None:
        """
        Compatibility method retained from the Ray implementation.

        These values are recorded but are not used by PyMC. Use ``chains`` and
        ``cores`` to control chain-level execution.
        """
        if self._has_run:
            raise ValueError(
                "Cannot change resources after sampling has started"
            )

        if (
            isinstance(cpu, bool)
            or not isinstance(cpu, int)
            or cpu <= 0
        ):
            raise ValueError("cpu must be a positive integer")

        if (
            isinstance(gpu, bool)
            or not isinstance(gpu, int)
            or gpu < 0
        ):
            raise ValueError("gpu must be a non-negative integer")

        self.resources_per_trial = {"cpu": cpu, "gpu": gpu}

        warnings.warn(
            "set_resources_per_trial() is retained only for API "
            "compatibility. The PyMC model does not allocate Ray trials.",
            RuntimeWarning,
            stacklevel=2,
        )

    def set_FLAG_log_to_file(self, flag: bool) -> None:
        if self._has_run:
            raise ValueError(
                "Cannot change FLAG_log_to_file after sampling has started"
            )

        if not isinstance(flag, bool):
            raise TypeError("flag must be bool")

        self.FLAG_log_to_file = flag

    def _build_model(self) -> pm.Model:
        """
        Build and retain the PyMC model.

        Calling this method repeatedly before sampling is safe; the existing
        model is returned.
        """
        if self.model is not None:
            return self.model

        #0. set up the wrapper around the black box simulation (deterministic mapping from params to output)
        self._loglike_op = _SimulationLogLikelihoodOp(
            parameter_names=self._parameter_names,
            simulate_fn=self.simulate_fn,
            objective_fn=self.objective_fn,
            sigma=self.sigma,
        )

        with pm.Model() as model:
            #1. priors 
            self._parameter_variables = {
                name: prior.build(name)
                for name, prior in self.param_prior_space.items()
            }

            #2. turn the black box simulation into a symbolic expression PyMC can work with 
            ordered_variables = [
                self._parameter_variables[name]
                for name in self._parameter_names
            ]


            log_likelihood = self._loglike_op(*ordered_variables)

            #3. log likelihood 
            pm.Potential("chr_simulation_loglikelihood", log_likelihood)

        self.model = model
        return model

    def run(self) -> None:
        if self._has_run:
            raise ValueError(
                "Calibration has already run. Construct a new "
                "ChRBayesCali instance for a fresh posterior."
            )

        if self.model is None:
            self._build_model()

        assert self.model is not None

        self.FLAG_auto_run = True
        self._has_run = True

        with self._output_context():
            print(
                "[ChRBayesCali] Starting PyMC calibration: "
                f"{self.chains} chain(s), "
                f"{self._tune} tune step(s), "
                f"{self._draws} retained draw(s) per chain.",
                flush=True,
            )

            with self.model:
                # The simulation likelihood is an opaque Python operation with
                # no derivative. Explicit Metropolis selection prevents PyMC
                # from attempting gradient-based NUTS.
                step = pm.Metropolis()

                self.idata = pm.sample(
                    draws=self._draws,
                    tune=self._tune,
                    chains=self.chains,
                    cores=min(self.cores, self.chains),
                    step=step,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    discard_tuned_samples=True,
                    progressbar=self.progressbar,
                    compute_convergence_checks=self.chains > 1,
                    var_names=list(self._parameter_names),
                )

            print(
                "[ChRBayesCali] Calibration complete. "
                f"Retained posterior samples: "
                f"{self._draws * self.chains}",
                flush=True,
            )

    @contextmanager
    def _output_context(self) -> Iterator[None]:
        if not self.FLAG_log_to_file:
            yield
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = (
            Path.cwd() / f"ChRBayesCali_{timestamp}.txt"
        )

        with self.log_file_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        ) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                yield

        print(
            f"[ChRBayesCali] Log written to {self.log_file_path}",
            flush=True,
        )

    def get_posterior(self) -> list[dict[str, float]]:
        """
        Return flattened posterior draws in the legacy ChronoRay format.

        Output order is chain-major, then draw-major.
        """
        if self.idata is None:
            raise ValueError(
                "Calibration has not run. Call run() first."
            )

        posterior_group = self.idata.posterior
        flattened: dict[str, np.ndarray] = {}

        for name in self._parameter_names:
            try:
                values = np.asarray(posterior_group[name].values)
            except Exception as exc:
                raise KeyError(
                    f"Posterior does not contain public parameter {name!r}"
                ) from exc

            flattened[name] = values.reshape(-1)

        sample_counts = {
            values.size for values in flattened.values()
        }

        if len(sample_counts) != 1:
            raise RuntimeError(
                "Posterior variables have inconsistent sample counts"
            )

        sample_count = sample_counts.pop()

        return [
            {
                name: float(flattened[name][index])
                for name in self._parameter_names
            }
            for index in range(sample_count)
        ]

    def get_inference_data(self):
        """Return PyMC's complete posterior result object."""
        if self.idata is None:
            raise ValueError(
                "Calibration has not run. Call run() first."
            )
        return self.idata

    def get_model(self) -> pm.Model:
        """Return the constructed PyMC model."""
        if self.model is None:
            self._build_model()

        assert self.model is not None
        return self.model