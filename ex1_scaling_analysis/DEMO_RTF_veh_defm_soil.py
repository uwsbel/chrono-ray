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
# Demonstration of a vehicle traversing SCM deformable terrain with wall-clock
# performance measurement and real-time factor (RTF) reporting.
#
# The simulation models a HMMWV operating on SCM deformable terrain using a
# prescribed driver input profile and fixed SCM soil parameters. The terrain
# active-domain feature is enabled around the vehicle chassis to limit SCM
# calculations to the region of interest.
#
# Wall-clock execution time is measured over the main simulation loop using
# Python's high-resolution performance timer. At the end of the simulation, the
# script reports:
#
#     - elapsed wall-clock time,
#     - elapsed simulated time, and
#     - real-time factor (RTF).
#
# In this implementation, RTF is defined as:
#
#     RTF = wall-clock time / simulated time
#
# Therefore, RTF < 1 indicates execution faster than real time, RTF = 1
# indicates real-time execution, and RTF > 1 indicates execution slower than
# real time.
#
# The vehicle reference frame has Z up, X towards the front of the vehicle, and
# Y pointing to the left. All units are SI unless otherwise noted.
#
# =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import math as m
import argparse
import time as wallclock 

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
##== RUN SIMULATION, WALL CLOCK TIMING AND RTF ==##
######################################################################

TERRAIN_LENGTH = 16.0   #terrain size in x dir
TERRAIN_WIDTH = 8.0     #terrain size in y dir
TERRAIN_DELTA = 0.05    #terrain (SCM) grid spacing

TOTAL_SIM_TIME = 5.0   #[s]
SIM_DT = 1e-3          #[s]



if __name__ == "__main__":

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
    terrain.SetSoilParameters(2e6,   # Bekker Kphi
                              0,     # Bekker Kc
                              1.1,   # Bekker n exponent
                              0,     # Mohr cohesive limit (Pa)
                              30,    # Mohr friction limit (degrees)
                              0.01,  # Janosi shear coefficient (m)
                              2e8,   # Elastic stiffness (Pa/m), before plastic yield
                              3e4    # Damping (Pa s/m), proportional to negative vertical speed (optional)
    )

    #4. enable active domains feature (single domain around vehicle chassis) -- optional (Py)Chrono feature 
    terrain.AddActiveDomain(hmmwv.GetChassisBody(), chrono.ChVector3d(0, 0, 0), chrono.ChVector3d(5, 3, 1))

    #5. initialize the terrain, specifying the initial mesh grid
    terrain.Initialize(TERRAIN_LENGTH, TERRAIN_WIDTH, TERRAIN_DELTA);

    #6. simulation loop
    time = 0.0
    wall_start = wallclock.perf_counter()   # start timer just before the loop

    while time < TOTAL_SIM_TIME:
        time += SIM_DT

        #6a-c. (driver / terrain / hmmwv synchronize + advance, unchanged)
        driver_inputs = driver.GetInputs()
        driver.Synchronize(time)
        terrain.Synchronize(time)
        hmmwv.Synchronize(time, driver_inputs, terrain)
        driver.Advance(SIM_DT)
        terrain.Advance(SIM_DT)
        hmmwv.Advance(SIM_DT)

    wall_elapsed = wallclock.perf_counter() - wall_start
    sim_elapsed  = time   # total simulated seconds

    rtf = wall_elapsed / sim_elapsed
    print(f"Wall-clock: {wall_elapsed:.3f} s")
    print(f"Simulated:  {sim_elapsed:.3f} s")
    print(f"RTF:        {rtf:.3f}  ({'slower' if rtf > 1 else 'faster'} than real-time)")





   