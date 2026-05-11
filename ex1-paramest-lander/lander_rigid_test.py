import pychrono.core as chrono
import pychrono.vsg3d as vsg3d
from Lander import Lander
import numpy as np
import time

#==============================================================
#==============================================================
#<3 SIMULATION OPTIONS (NOTE: please change to suit your needs!)
lander_location = chrono.ChVector3d(0, 0, 4)
ground_position = chrono.ChVector3d(0, 0, 0) #where the ground plane is located 
ground_length = 10.0 #m  
ground_width  = 10.0 #m
ground_height = 0.25 #m
hz = ground_height / 2
#==============================================================
#==============================================================

#<3 SYSTEM 
sys = chrono.ChSystemNSC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))

#==============================================================
#==============================================================
#<3 LANDER
l = Lander(sys)
user_rot = chrono.QuatFromAngleAxis(chrono.CH_PI/4, chrono.ChVector3d(0, 1, 0))
l.add_lander(lander_location)#, user_rot) 
#==============================================================
#==============================================================

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

#<3 VISUALIZATION
vis = vsg3d.ChVisualSystemVSG()
vis.AttachSystem(sys)
vis.SetWindowSize(1024, 768)
vis.SetWindowTitle("25-Body Lander Model With Custom Honeycomb Force Model on Rigid Terrain (RTF == 0.06)")
vis.AddCamera(chrono.ChVector3d(0, 8, 5), chrono.ChVector3d(0, 0, 0))
vis.SetGuiVisibility(False)

vis.Initialize()

#<3 SIM LOOP 
dt = 1e-3
peak_accel = 0.0 

while l.hit_ground() == False:
    vis.Render()
    sys.DoStepDynamics(dt)

    curr_accel = l.get_accel()
    if curr_accel > peak_accel:
        peak_accel = curr_accel
    
e_absorbed = l.get_energy_absorbed()
print(f"Peak Acceleration: {peak_accel}")

# cool_down = 1000

# for i in range(cool_down):
#     vis.Render()
#     sys.DoStepDynamics(dt)
#     # print(l.get_pos().z)

print(f"Energy Absorbed: {l.get_energy_absorbed()}")
print(f"Initial Energy: {l.get_energy_initial()}")