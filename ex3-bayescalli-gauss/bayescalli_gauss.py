import numpy as np
import arviz as az
import matplotlib.pyplot as plt

from ChronoRay import ChRBayesCalli

#1. gen observed data

rng = np.random.default_rng(123)

data = rng.normal(
    loc=0.0,
    scale=1.0,
    size=1000
)


#2. simulation function

def normal_sim(rng, a, b, size=1000):
    return rng.normal(
        loc=a,
        scale=b,
        size=size
    )


#3. define prior 

param_space = {
    "a": ChRBayesCalli.normal(
        mu=0,
        sigma=5
    ),
    "b": ChRBayesCalli.halfnormal(
        sigma=1
    ),
}


#4. run bayes calli 

calli = ChRBayesCalli(
    simulate_fn=normal_sim,
    param_space=param_space,
    data=data,
    epsilon=1.0,
    sum_stat="sort",
    FLAG_auto_run=True
)


#5. inspect results

idata = calli.idata

print(az.summary(idata, kind="stats"))

az.plot_trace(idata)
plt.show()

posterior = idata["posterior"]

a_samples = posterior["a"].values.flatten()
b_samples = posterior["b"].values.flatten()

plt.figure()
plt.hist(a_samples, bins=50, density=True, alpha=0.7)
plt.axvline(0.0, linestyle="--", label="true a = 0")
plt.title("Posterior distribution of a")
plt.xlabel("a")
plt.ylabel("density")
plt.legend()
plt.show()

plt.figure()
plt.hist(b_samples, bins=50, density=True, alpha=0.7)
plt.axvline(1.0, linestyle="--", label="true b = 1")
plt.title("Posterior distribution of b")
plt.xlabel("b")
plt.ylabel("density")
plt.legend()
plt.show()