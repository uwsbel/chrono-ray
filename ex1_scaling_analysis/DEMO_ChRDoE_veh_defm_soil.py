# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2014 projectchrono.org
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =============================================================================
# Authors: Khailanii Slaton, Radu Serban
# =============================================================================
#
# Demonstration of a vehicle traversing SCM deformable terrain, adapted for
# execution through the Chrono::Ray design-of-experiments (DoE) workflow.
#
# The underlying PyChrono simulation models a HMMWV operating on SCM deformable
# terrain and logging tire-terrain forces. Selected SCM soil parameters are exposed 
# through a Chrono::Ray parameter space and sampled using a Sobol design. 
# Each sampled configuration is evaluated by the simulation callable `simulate_fn`
# and the results are logged to a CSV file.
#
# The number of GPUs used by the Chrono::Ray workflow may be specified from the
# command line with:
#
#     --num_gpu <N>
#
# For example:
#
#     python <script_name>.py --num_gpu 4
#
# If omitted, the workflow defaults to one GPU.
#
# The vehicle reference frame has Z up, X towards the front of the vehicle, and
# Y pointing to the left. All units are SI unless otherwise noted.
#
# =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import math as m
import argparse
import os 
import uuid
import csv

from ChronoRay import ChRDoE

######################################################################
##== HELPER CLASS NEEDED FOR PYCHRONO SIMULATION, NOT CHRONO::RAY ==##
######################################################################

class MyDriver (veh.ChDriver):

    def __init__(self, vehicle, delay):
        veh.ChDriver.__init__(self, vehicle)
        self.delay = delay

    def Synchronize(self, time):
        eff_time = time - self.delay

        if (eff_time < 0):
            return
            
        if (eff_time > 0.2):
            self.SetThrottle(0.7)
        else:
            self.SetThrottle(3.5 * eff_time)

        if (eff_time < 2):
            self.SetSteering(0.0)
        else:
            self.SetSteering(0.6 * m.sin(2.0 * m.pi * (eff_time - 2) / 6))

        self.SetBraking(0.0)

######################################################################
##== CHRONO::RAY STEP 1: SIMULATION ==##
######################################################################

#NOTE: variables shared across all simulations can be declared outside of the Python callable 
TERRAIN_LENGTH = 16.0   #terrain size in x dir
TERRAIN_WIDTH = 8.0     #terrain size in y dir
TERRAIN_DELTA = 0.05    #terrain (SCM) grid spacing

TOTAL_SIM_TIME = 5.0   #[s]
SIM_DT = 1e-3          #[s]
OUTPUT_DIR = "chrdoe_demo_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_fn(config):

    #0. pull the params from the config dict 
    config_kphi = config["kphi"]
    config_kc = config["kc"]
    config_n = config["n"]
    config_mohr_cohesive_limit = config["mohr_cohesive_limit"]
    config_mohr_friction_limit = config["mohr_friction_limit"]
    config_janosi_shear_coefficient = config["janosi_shear_coefficient"]
    config_elastic_stiffness = config["elastic_stiffness"]
    config_damping = config["damping"]

    ##STARTING THE SIMULATION##
    
    #1. create the vehicle
    hmmwv = veh.HMMWV_Full()
    hmmwv.SetContactMethod(chrono.ChContactMethod_SMC)
    hmmwv.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(-5, -2, 0.6), chrono.ChQuaterniond(1, 0, 0, 0)))
    hmmwv.SetEngineType(veh.EngineModelType_SHAFTS);
    hmmwv.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS);
    hmmwv.SetDriveType(veh.DrivelineTypeWV_AWD)
    hmmwv.SetTireType(veh.TireModelType_RIGID)
    hmmwv.Initialize()

    hmmwv.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    #2. create the driver 
    driver = MyDriver(hmmwv.GetVehicle(), 0.5)
    driver.Initialize()

    #3. create the terrain (SCM deformable terrain) using the params from the config dict 
    terrain = veh.SCMTerrain(hmmwv.GetSystem())
    terrain.SetSoilParameters(config_kphi, config_kc, config_n, config_mohr_cohesive_limit, config_mohr_friction_limit, config_janosi_shear_coefficient, config_elastic_stiffness, config_damping)

    #4. enable active domains feature (single domain around vehicle chassis) -- optional (Py)Chrono feature 
    terrain.AddActiveDomain(hmmwv.GetChassisBody(), chrono.ChVector3d(0, 0, 0), chrono.ChVector3d(5, 3, 1))

    #5. initialize the terrain, specifying the initial mesh grid
    terrain.Initialize(TERRAIN_LENGTH, TERRAIN_WIDTH, TERRAIN_DELTA);

    #6. setup logging 

    #get wheel/tire references (for logging) before entering the loop
    vehicle = hmmwv.GetVehicle()

    tire_FL = vehicle.GetAxle(0).m_wheels[veh.VehicleSide_LEFT].GetTire()
    tire_FR = vehicle.GetAxle(0).m_wheels[veh.VehicleSide_RIGHT].GetTire()
    tire_RL = vehicle.GetAxle(1).m_wheels[veh.VehicleSide_LEFT].GetTire()
    tire_RR = vehicle.GetAxle(1).m_wheels[veh.VehicleSide_RIGHT].GetTire()

    csv_file = os.path.join(OUTPUT_DIR, f"trial_{os.getpid()}_{uuid.uuid4().hex[:8]}.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

    writer.writerow([
        "time",
        "FL_Fx", "FL_Fy", "FL_Fz",
        "FR_Fx", "FR_Fy", "FR_Fz",
        "RL_Fx", "RL_Fy", "RL_Fz",
        "RR_Fx", "RR_Fy", "RR_Fz",
    ])

    #7. simulation loop
    time = 0.0

    while time < TOTAL_SIM_TIME:

        #7a. get driver inputs
        driver_inputs = driver.GetInputs()

        #7b. update modules
        driver.Synchronize(time)
        terrain.Synchronize(time)
        hmmwv.Synchronize(time, driver_inputs, terrain)

        #7c. report tire-terrain forces
        force_FL = tire_FL.ReportTireForce(terrain).force
        force_FR = tire_FR.ReportTireForce(terrain).force
        force_RL = tire_RL.ReportTireForce(terrain).force
        force_RR = tire_RR.ReportTireForce(terrain).force

        #7d. log current state
        writer.writerow([
            time,
            force_FL.x, force_FL.y, force_FL.z,
            force_FR.x, force_FR.y, force_FR.z,
            force_RL.x, force_RL.y, force_RL.z,
            force_RR.x, force_RR.y, force_RR.z,
        ])

        #7e. advance simulation for one timestep
        driver.Advance(SIM_DT)
        terrain.Advance(SIM_DT)
        hmmwv.Advance(SIM_DT)

        time += SIM_DT

    return 0
######################################################################
##== CHRONO::RAY STEP 2: PARAMETER SPACE ==##
######################################################################

param_space = {
    "kphi": ChRDoE.uniform(1e5, 5e6),                           #[Pa/m]
    "kc": ChRDoE.uniform(0, 1e5),                               #[Pa]
    "n": ChRDoE.uniform(0.7, 1.3),                              #[-]
    "mohr_cohesive_limit": ChRDoE.uniform(0, 20e3),             #[Pa]
    "mohr_friction_limit": ChRDoE.uniform(20, 40),              #[Degrees]
    "janosi_shear_coefficient": ChRDoE.uniform(0.005, 0.05),    #[m]
    "elastic_stiffness": ChRDoE.uniform(1e7, 2e8),              #[Pa/m]
    "damping": ChRDoE.uniform(1e3, 5e4)                         #[Pa*s/m]
}

######################################################################
##== CHRONO::RAY STEP 3: OBJECTIVE FUNCTION ==##
######################################################################

#NOTE: not needed since we will be using ChRDoE :) 

######################################################################
##== CHRONO::RAY STEP 4: SET UP WORKFLOW AND RUN ==##
######################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpu", type=int, default=1)
    args = parser.parse_args()


    doe = ChRDoE(
        simulate_fn = simulate_fn,
        param_space = param_space,
        sampling_design = ChRDoE.SamplingDesign.SOBOL,
        num_trials = 2**10,
        num_gpus = args.num_gpu,
        results_prefix = "chrdoe_demo_veh_defm_soil",
        FLAG_auto_run = True
    )