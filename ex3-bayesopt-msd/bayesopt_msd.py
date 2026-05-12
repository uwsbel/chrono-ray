import numpy as np
import pychrono as chrono

MASS = 3.0


def analytical(t, m, k, c, x0, v0):
    omega_n = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))

    if zeta < 1.0:
        omega_d = omega_n * np.sqrt(1.0 - zeta**2)
        A = x0
        B = (v0 + zeta * omega_n * x0) / omega_d
        return np.exp(-zeta * omega_n * t) * (
            A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
        )

    elif np.isclose(zeta, 1.0):
        A = x0
        B = v0 + omega_n * x0
        return (A + B * t) * np.exp(-omega_n * t)

    else:
        r1 = -omega_n * (zeta - np.sqrt(zeta**2 - 1.0))
        r2 = -omega_n * (zeta + np.sqrt(zeta**2 - 1.0))

        A = (v0 - r2 * x0) / (r1 - r2)
        B = x0 - A

        return A * np.exp(r1 * t) + B * np.exp(r2 * t)


def simulate_fn(config):
    k = config["k"]
    c = config["c"]

    x0 = 0.1
    v0 = 0.0

    time_step = 0.001
    time_end = 5.0

    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))

    ground = chrono.ChBody()
    ground.SetFixed(True)
    system.Add(ground)

    mass_body = chrono.ChBody()
    mass_body.SetMass(MASS)
    mass_body.SetPos(chrono.ChVector3d(x0, 0, 0))
    mass_body.SetPosDt(chrono.ChVector3d(v0, 0, 0))
    system.Add(mass_body)

    spring = chrono.ChLinkTSDA()
    spring.Initialize(
        ground,
        mass_body,
        False,
        chrono.ChVector3d(0, 0, 0),
        chrono.ChVector3d(x0, 0, 0),
    )
    spring.SetSpringCoefficient(k)
    spring.SetDampingCoefficient(c)
    spring.SetRestLength(0.0)
    system.Add(spring)

    times = []
    x_sim = []
    x_exact = []

    t = 0.0
    while t <= time_end:
        x = mass_body.GetPos().x

        times.append(t)
        x_sim.append(x)
        x_exact.append(analytical(t, MASS, k, c, x0, v0))

        system.DoStepDynamics(time_step)
        t += time_step

    return {
        "time": np.array(times),
        "x_sim": np.array(x_sim),
        "x_exact": np.array(x_exact),
        "params": {
            "m": MASS,
            "k": k,
            "c": c,
            "x0": x0,
            "v0": v0,
            "time_step": time_step,
            "time_end": time_end,
        },
    }