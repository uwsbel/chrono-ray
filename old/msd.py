import ray
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air import session
from ChronoRay import ChronoRay


# ============================================================
# Simulation Function (PyChrono)
# ============================================================
def simulate(params):
    import pychrono as chrono
    import math

    MASS = 1.0
    SPRING_K = params["k"]
    DAMPER_C = params["c"]
    X0 = 0.1
    V0 = 0.0

    TIME_STEP = 0.001
    TIME_END = 5.0

    def analytical(t, m, k, c, x0, v0):
        omega_n = math.sqrt(k / m)
        zeta = c / (2 * math.sqrt(m * k))

        if zeta < 1.0:
            omega_d = omega_n * math.sqrt(1 - zeta**2)
            A = x0
            B = (v0 + zeta * omega_n * x0) / omega_d
            return math.exp(-zeta * omega_n * t) * (
                A * math.cos(omega_d * t) + B * math.sin(omega_d * t)
            )
        else:
            return 0.0

    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))

    ground = chrono.ChBody()
    ground.SetFixed(True)
    system.Add(ground)

    mass_body = chrono.ChBody()
    mass_body.SetMass(MASS)
    mass_body.SetPos(chrono.ChVector3d(X0, 0, 0))
    mass_body.SetPosDt(chrono.ChVector3d(V0, 0, 0))
    system.Add(mass_body)

    spring = chrono.ChLinkTSDA()
    spring.Initialize(
        ground,
        mass_body,
        False,
        chrono.ChVector3d(0, 0, 0),
        chrono.ChVector3d(X0, 0, 0),
    )
    spring.SetSpringCoefficient(SPRING_K)
    spring.SetDampingCoefficient(DAMPER_C)
    spring.SetRestLength(0.0)
    system.Add(spring)

    t = 0.0
    times = []
    sim_positions = []
    exact_positions = []

    while t <= TIME_END:
        x_sim = mass_body.GetPos().x
        x_exact = analytical(t, MASS, SPRING_K, DAMPER_C, X0, V0)

        times.append(t)
        sim_positions.append(x_sim)
        exact_positions.append(x_exact)

        system.DoStepDynamics(TIME_STEP)
        t += TIME_STEP

    return {
        "time": times,
        "x_sim": sim_positions,
        "x_exact": exact_positions,
    }


# ============================================================
# Evaluation Function
# ============================================================
def evaluate(data):
    import numpy as np

    x_sim = np.array(data["x_sim"])
    x_exact = np.array(data["x_exact"])

    error = np.abs(x_sim - x_exact)

    return float(np.mean(error)),


# ============================================================
# Parameter Space
# ============================================================
param_space = {
    "k": tune.uniform(10.0, 100.0),
    "c": tune.uniform(0.1, 10.0),
}


# ============================================================
# Run Optimization
# ============================================================
if __name__ == "__main__":
    runner = ChronoRay(
        simulate_fn=simulate,
        evaluate_fn=evaluate,
        param_space=param_space,
        resources_per_trial={"cpu": 2},
        num_samples=20,
    )

    results = runner.run()

    best = results.get_best_result(metric="objective", mode="min")
    print("\nBest parameters found:")
    print(best.config)
    print("Best objective:", best.metrics["objective"])

