'''
DESIRED PARAMETERS: 
    l = 1.825 
    beta = 0.30077859136349705
    alpha2 = 1.0472
    fy = 150

TARGET SIM OUTPUTS:
    Peak Acceleration: 2412.9178028956803
    Energy Absorbed: 40.168956533686924
'''

from ChronoRay import ChRParamEst
from Lander import Lander 
import pychrono.core as chrono
import numpy as np
import time

#1. PARAMETER SAMPLE SPACE
param_sample_space = {
        "beta": ChRParamEst.ChR_Distr.uniform(0.261799, 1.0472),
        "alpha2": ChRParamEst.ChR_Distr.uniform(0.872665, 1.825),
        "fy": ChRParamEst.ChR_Distr.loguniform(100, 150)
                    }

#2. TARGET SIMULATION OUTPUT VALS
target_sim_outputs = {
    "peak_accel": 2412.9178028956803,
    "energy_absorbed": 40.168956533686924
                    }

#3. SIMULATION WRAPPED IN FUNCTION
def simulate_fn(config):

    lander_location = chrono.ChVector3d(0, 0, 4)
    ground_position = chrono.ChVector3d(0, 0, 0) #where the ground plane is located 
    ground_length = 10.0 #m  
    ground_width  = 10.0 #m
    ground_height = 0.25 #m
    hz = ground_height / 2

    #<3 SYSTEM 
    sys = chrono.ChSystemNSC()
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))

    #<3 LANDER
    l = Lander(sys)
    l.set_lander_params(**config)
    l.add_lander(lander_location)

    #<3 GROUND PLANE 
    GROUND_FAMILY = 0 #just safety measure 
    ground_body = chrono.ChBody()
    ground_body.SetMass(10)
    ground_body.SetFixed(True)
    ground_body.SetPos(ground_position + chrono.ChVector3d(0, 0, -hz))
    ground_body.EnableCollision(True)
    ground_body.AddCollisionShape(chrono.ChCollisionShapeBox(l.cmaterial, ground_width, ground_length, hz))
    ground_body.GetCollisionModel().SetFamily(GROUND_FAMILY)
    sys.AddBody(ground_body)
    ground_shape = chrono.ChVisualShapeBox(ground_width, ground_length, hz)
    ground_shape.SetColor(chrono.ChColor(0.2, 0.7, 0.3))
    ground_body.AddVisualShape(ground_shape)

    #<3 SIM LOOP 
    dt = 1e-3
    peak_accel = 0.0 

    while l.hit_ground() == False:
        sys.DoStepDynamics(dt)

        curr_accel = l.get_accel()
        if curr_accel > peak_accel:
            peak_accel = curr_accel
        
    e_absorbed = l.get_energy_absorbed()

    return {
        "peak_accel": peak_accel,
        "energy_absorbed": e_absorbed
    }

#4. CHRONO RAY PARAMETER ESTIMATION INSTANCE 
param_est = ChRParamEst(simulate_fn=simulate_fn, 
                        est_rule=ChRParamEst.EstRule.LS, 
                        param_sample_space=param_sample_space, 
                        target_sim_outputs=target_sim_outputs,
                        total_trials=50)