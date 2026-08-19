import numpy as np
import pychrono as chrono
import os
import csv
import pandas as pd 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ChronoRay import ChRBayesCali

######################################################################
##== HELPERS NEEDED FOR PYCHRONO SIMULATION, NOT CHRONO::RAY ==##
######################################################################

#NOTE: also put the sim params here
SIM_DT = 1.0e-3   #[s]
TOTAL_SIM_TIME = 10.0   #[s]

#discard the beginning of the simulation from the calibration metric.
#helps prevent startup transients from dominating the comparison.
DATA_START_TIME = 0.5

#record every N integration steps.
SAMPLE_EVERY = 10

#prescribed crank speed [rad/s]
CRANK_SPEED = np.pi

#geometry [m]
CRANK_RADIUS = 0.4
CRANK_THICKNESS = 0.10

ROD_LENGTH = 1.5
ROD_WIDTH = 0.10
ROD_HEIGHT = 0.10

PISTON_RADIUS = 0.20
PISTON_LENGTH = 0.30

#fixed densities for bodies that are not being calibrated [kg/m^3]
FLOOR_DENSITY = 1000.0
CRANK_DENSITY = 1000.0
PISTON_DENSITY = 1000.0

#fixed spring stiffness.
PISTON_SPRING_K = 0.0

#NOTE: this is the true parameter set that will be used to generate the synthetic observations
TRUE_PARAMS = {
    "rod_density": 780.0,       # kg/m^3
    "damping_coeff": 250.0,     # N s/m
}

TORQUE_NOISE_STD = 5.0

OUTPUT_DIR = "chrbayescali_demo_results"
GROUND_TRUTH_FILE_NAME = "ground_truth.csv"

GROUND_TRUTH_DATA = None

def _vector_x(vec):
    x = vec.x
    return float(x() if callable(x) else x)


def _build_slider_crank(config):

    #0. pull the params from the config dict 
    rod_density = float(config["rod_density"])
    damping_coeff = float(config["damping_coeff"])

    #1. system and misc config 
    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, 0.0))

    crank_center = chrono.ChVector3d(-1.0, 0.5, 0.0)

    #2. ground body
    floor = chrono.ChBodyEasyBox(3.0, 1.0, 3.0, FLOOR_DENSITY)
    floor.SetPos(chrono.ChVector3d(0.0, -0.5, 0.0))
    floor.SetFixed(True)
    system.Add(floor)

    #3. crank
    crank = chrono.ChBodyEasyCylinder(chrono.ChAxis_Y, CRANK_RADIUS, CRANK_THICKNESS, CRANK_DENSITY)
    crank.SetPos(crank_center + chrono.ChVector3d(0.0, 0.0, -0.1))
    system.Add(crank)

    #4. connecting rod
    rod = chrono.ChBodyEasyBox(ROD_LENGTH, ROD_WIDTH, ROD_HEIGHT, rod_density)
    rod.SetPos(crank_center + chrono.ChVector3d(CRANK_RADIUS + ROD_LENGTH / 2.0, 0.0, 0.0))
    system.Add(rod)

    #5. piston
    piston = chrono.ChBodyEasyCylinder(chrono.ChAxis_X, PISTON_RADIUS, PISTON_LENGTH, PISTON_DENSITY)
    piston_initial_pos = crank_center + chrono.ChVector3d(CRANK_RADIUS + ROD_LENGTH, 0.0, 0.0)
    piston.SetPos(piston_initial_pos)
    system.Add(piston)

    #6. prescribed-speed crank motor
    motor = chrono.ChLinkMotorRotationSpeed()
    motor.Initialize(crank, floor, chrono.ChFrameD(crank_center))
    motor.SetSpeedFunction(chrono.ChFunctionConst(CRANK_SPEED))
    system.Add(motor)

    #7. crank-to-rod revolute joint
    joint_crank_rod = chrono.ChLinkLockRevolute()
    joint_crank_rod.Initialize(rod, crank, chrono.ChFrameD(crank_center + chrono.ChVector3d(CRANK_RADIUS, 0.0, 0.0)))
    system.Add(joint_crank_rod)

    #8. rod-to-piston revolute joint
    joint_rod_piston = chrono.ChLinkLockRevolute()
    joint_rod_piston.Initialize(piston, rod, chrono.ChFrameD(crank_center + chrono.ChVector3d(CRANK_RADIUS + ROD_LENGTH, 0.0, 0.0)))
    system.Add(joint_rod_piston)

    #9. piston-to-ground prismatic joint
    joint_piston_ground = chrono.ChLinkLockPrismatic()
    #the prismatic axis must align with the piston x direction.
    prism_frame = chrono.ChFrameD(piston_initial_pos, chrono.QuatFromAngleY(np.pi / 2.0))
    joint_piston_ground.Initialize(piston, floor, prism_frame)
    system.Add(joint_piston_ground)

    #10. piston viscous load
    piston_damper = chrono.ChLinkTSDA()
    #positions are specified in the absolute frame because local=False.
    ground_anchor = chrono.ChVector3d(-1.5, 0.5, 0.0)
    piston_anchor = piston_initial_pos

    piston_damper.Initialize(piston, floor, False, piston_anchor, ground_anchor)
    piston_damper.SetSpringCoefficient(PISTON_SPRING_K)
    piston_damper.SetDampingCoefficient(damping_coeff)

    system.Add(piston_damper)

    return system, crank, rod, piston, motor, piston_damper

def _generate_synthetic_observations(true_params=None, torque_noise_std=0.0, random_seed=42):

    csv_file = os.path.join(OUTPUT_DIR, GROUND_TRUTH_FILE_NAME)

    if true_params is None:
        true_params = TRUE_PARAMS

    reference = simulate_fn(true_params)

    observed_torque = np.asarray(reference["motor_torque"], dtype=float)

    if torque_noise_std > 0.0:
        rng = np.random.default_rng(random_seed)
        observed_torque += rng.normal(loc=0.0, scale=torque_noise_std, size=observed_torque.shape)

    reference["motor_torque"] = observed_torque.tolist()

    pd.DataFrame(reference).to_csv(csv_file, index=False)

######################################################################
##== CHRONO::RAY STEP 1: SIMULATION ==##
######################################################################

def simulate_fn(config):
    system, crank, rod, piston, motor, piston_damper = _build_slider_crank(config)

    time_history = []
    torque_history = []
    piston_x_history = []
    piston_vx_history = []
    damper_force_history = []

    step = 0

    while system.GetChTime() < TOTAL_SIM_TIME:
        system.DoStepDynamics(SIM_DT)
        time = system.GetChTime()

        if time >= DATA_START_TIME and step % SAMPLE_EVERY == 0:
            motor_torque = float(motor.GetMotorTorque())
            piston_pos = piston.GetPos()
            piston_vel = piston.GetPosDt()
            damper_force = float(piston_damper.GetForce())

            time_history.append(time)
            torque_history.append(motor_torque)
            piston_x_history.append(_vector_x(piston_pos))
            piston_vx_history.append(_vector_x(piston_vel))
            damper_force_history.append(damper_force)

        step += 1

    time_history = np.asarray(time_history, dtype=float)
    torque_history = np.asarray(torque_history, dtype=float)
    piston_x_history = np.asarray(piston_x_history, dtype=float)
    piston_vx_history = np.asarray(piston_vx_history, dtype=float)
    damper_force_history = np.asarray(damper_force_history, dtype=float)

    return {"time": time_history, "motor_torque": torque_history, "piston_x": piston_x_history, "piston_vx": piston_vx_history, "damper_force": damper_force_history}

######################################################################
##== CHRONO::RAY STEP 2: PARAMETER SPACE ==##
######################################################################
param_prior_space = {
    "rod_density": ChRBayesCali.ChR_Distr.uniform(500.0, 1100.0),
    "damping_coeff": ChRBayesCali.ChR_Distr.uniform(50.0, 500.0),
}

######################################################################
##== CHRONO::RAY STEP 3: OBJECTIVE FUNCTION ==##
######################################################################
def objective_fn(sim_output):
    ground_truth_torque = GROUND_TRUTH_DATA["motor_torque"].to_numpy()
    simulated_torque = np.asarray(sim_output["motor_torque"], dtype=float)

    if simulated_torque.shape != ground_truth_torque.shape:
        raise ValueError(f"Simulated and ground-truth torque histories have different shapes: {simulated_torque.shape} vs {ground_truth_torque.shape}")

    residual = simulated_torque - ground_truth_torque

    return float(np.sum(residual**2))

######################################################################
##== CHRONO::RAY STEP 4: SET UP WORKFLOW AND RUN ==##
######################################################################

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #0. generate the synthetic observation 
    _generate_synthetic_observations(torque_noise_std=TORQUE_NOISE_STD)

    GROUND_TRUTH_DATA = pd.read_csv(os.path.join(OUTPUT_DIR, GROUND_TRUTH_FILE_NAME))

    #1. run :) 
    chrbayescali = ChRBayesCali(
        simulate_fn=simulate_fn,
        objective_fn=objective_fn,
        param_prior_space=param_prior_space,
        sigma=TORQUE_NOISE_STD,
        total_samples=2000,
        burn_in_frac=0.30,
    )

    #2. plot the results
    posterior = chrbayescali.get_posterior()

    rod_density = [sample["rod_density"] for sample in posterior]
    damping_coeff = [sample["damping_coeff"] for sample in posterior]

    plt.figure()
    plt.plot(rod_density, damping_coeff, "o", markersize=2, alpha=0.25)
    plt.plot(TRUE_PARAMS["rod_density"], TRUE_PARAMS["damping_coeff"], "x", markersize=10)
    plt.xlabel("rod_density")
    plt.ylabel("damping_coeff")
    plt.title("Bayesian Calibration Results")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bayesian_calibration_results.png"), dpi=300)
    plt.close()