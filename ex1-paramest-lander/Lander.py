'''
DESCRIPTION: 
this file contains the instantiation and creation of lander pychrono mbs model 
with custom aluminium honeycomb force model, configured for rigid terrain. 

=====================================================
API
=====================================================
Lander(sys)
DESCRIPTION: constructor
    - sys: assumes system is of type NSC (non-smooth contact)
    - returns a Lander object

set_lander_params(**kwargs)
DESCRIPTION: setter for tunable lander params 
    - kwargs: dictionary of parameters to set
        note: only parameters in _ALLOWED_PARAMS can be set
    - returns None

add_lander(lander_bottom_pos = chrono.ChVector3d(0, 0, 0))
DESCRIPTION: adds the lander to the system (passed to the constructor)
    - lander_bottom_pos: position of the bottom-center of the lander base
    - returns None
'''

import pychrono as chrono
import math
from HoneycombForceFunctor import HoneycombForceFunctor 


class Lander: 

    #====================================================
    #<3 CONSTANTS 
    #====================================================

    #some large number that should be available system wide... 
    LANDER_FAMILY = 100 

    #NOTE: currently disabled 
    #lander attributes that can be set via setter function 
    # _ALLOWED_PARAMS = {  
    # #lander base parameters 
    # "mass",
    # "inertia",

    # #angles for strut construction 
    # "beta",
    # "alpha2",

    # #honeycomb 
    # "fy", 
    # }

    #====================================================
    #<3 CONSTRUCTOR 
    #====================================================

    #PARAMETERS:
    # sys: assumes system is of type NSC (non-smooth contact)
    def __init__(self, sys): 
        self.sys = sys
        self.sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        self.cmaterial = chrono.ChContactMaterialNSC() 

        #<3 PARAMETERS FOR THE LANDER 
        self.h1 = 0.9557                                    #h1: dist btwn ground and bottom of lander body 
        self.h2 = 2.0                                       #h2: height/length of lander body 
        self.h2_offset = 0.4554                             #h2_offset: attatchment point for A*-D* strut based on IM dimensions 
        self.r = 0.8405                                     #r: radius of the lander body 
        self.mass = 565                                     #mass of the lander body 
        self.inertia = chrono.ChVector3d(1.0, 1.0, 1.0)     #inertia of the lander body 
        self.com_offset = chrono.ChVector3d(-0.0220, -0.0830, 0.7909) 

        self.beta = 0.30077859136349705                     #beta: (abs) angle betweeen leg A/C and the center line of the strut 
        c = chrono.CH_PI / 180 
        #tilt from vertical is 39.4, but we tilt the leg from the horizontal 
        self.alpha2 = 60*c#50.59*c #(90-39.38713744471256)*c              #self.alpha2: angle in which leg B/E is oriented relative to center of lander body 
        self.num_struts = 6                                 #number of struts (legs) attatched to lander body
        
        #<3 [STRUT SPECIFIC PARAMS] LEG LENGTHS (that can be precomputed)
        self.l = 1.825#(self.h1 + 0.5*self.h2)/(math.sin(self.alpha2)) #l: length of center strut leg (fully extended) 

        self.e_leg_length = self.l*(9/10) 
        self.b_leg_length = self.l*(1/10) 

        #<3 [STRUT SPECIFIC PARAMS] LEG RADII 
        self.e_leg_radius = 0.05 
        self.b_leg_radius = self.e_leg_radius / 1.5
        self.a_leg_radius = 0.05 
        self.c_leg_radius = 0.05 

        #<3 [STRUT SPECIFIC PARAMS] LEG MASSES 
        self.e_leg_mass = 10 
        self.b_leg_mass = self.e_leg_mass / 1.5
        self.a_leg_mass = 10  
        self.c_leg_mass = 10  

        #<3 LEG INERTIAS 
        self.e_leg_inertia = chrono.ChVector3d(1.0, 1.0, 1.0)
        self.b_leg_inertia = chrono.ChVector3d(1.0, 1.0, 1.0)
        self.a_leg_inertia = chrono.ChVector3d(1.0, 1.0, 1.0)
        self.c_leg_inertia = chrono.ChVector3d(1.0, 1.0, 1.0)

        self.fy = 50 
        self.hcff_list = []
        self.honeycomb_list = []
        
        self.lander_bodies = []
        self.lander_radii = []
        self.lander_lengths = []
        self.footpad_bodies = [] 
        
        self.base_pos = chrono.ChVector3d(0, 0, 0)
        self.base_body = None 

        self.footpad_mesh_file = "./im_footpad.obj"
        self.footpad_points_file = "./im_footpad_points.csv"
        self.footpad_mass = 10
    #====================================================
    #<3 FUNC TO HELP WITH MODELING 
    #====================================================

    def add_marker(self,pos, color = "red"):
        marker = chrono.ChBody()
        marker.SetMass(0)
        marker.SetInertiaXX(chrono.ChVector3d(0, 0, 0))
        marker.SetPos(pos)
        marker.SetRot(chrono.ChQuaterniond(1, 0, 0, 0))
        marker.EnableCollision(False)
        marker.SetFixed(True)
        self.sys.AddBody(marker)
        marker_shape = chrono.ChVisualShapeSphere(0.08)
        if color == "red":
            marker_shape.SetColor(chrono.ChColor(1, 0, 0))
        elif color == "green":
            marker_shape.SetColor(chrono.ChColor(0, 1, 0))
        elif color == "blue":
            marker_shape.SetColor(chrono.ChColor(0, 0, 1))
        elif color == "yellow":
            marker_shape.SetColor(chrono.ChColor(1, 1, 0))
        elif color == "purple":
            marker_shape.SetColor(chrono.ChColor(1, 0, 1))
        marker.AddVisualShape(marker_shape)
        return marker

    #====================================================
    #<3 SETTER TO CONFIGURE LANDER PARAMETERS
    #====================================================

    def set_lander_params(self, **kwargs): 
        for name, value in kwargs.items(): 

            if name not in self._ALLOWED_PARAMS:
                raise ValueError(f"Unknown parameter: {name}")

            if value is not None:
                setattr(self, name, value)

                if (name == 'alpha2'):
                    self.l = (self.h1 + 0.5*self.h2)/(math.sin(self.alpha2))
                    self.e_leg_length = self.l*(3/4) 
                    self.b_leg_length = self.l/4 

    
    #====================================================
    #<3 LANDER STRUT CONSTRUCTION 
    #====================================================
    def make_leg_between(self, p0, p1, radius, mass, inertia, color):
        v = p1 - p0
        L = v.Length()

        if L < 1e-9:
            raise ValueError("Leg endpoints coincide or are too close.")

        dir = v / L
        rot = chrono.QuatFromVec2Vec(chrono.ChVector3d(0, 0, 1), dir)

        body = chrono.ChBody()
        body.SetMass(mass)
        body.SetInertiaXX(inertia)
        body.SetRot(rot)
        body.SetPos((p0 + p1) * 0.5)
        self.sys.AddBody(body)

        shape = chrono.ChVisualShapeCylinder(radius, L)
        shape.SetColor(color)
        body.AddVisualShape(shape)

        return body, L

    def add_strut(self, des_point, base_body, base_center, user_rot):

        #<3 INFO TO ALIGN THE STRUT
        vec_des2base = base_center - des_point
        vec_des2base.Normalize()
        z_axis = vec_des2base
        z_axis.Normalize()

        base_body_axis = user_rot.Rotate(chrono.ChVector3d(0, 0, 1))

        x_axis = base_body_axis.Cross(z_axis)
        x_axis.Normalize()

        y_axis = z_axis.Cross(x_axis)

        A = chrono.ChMatrix33d()
        A.SetFromDirectionAxes(x_axis, y_axis, z_axis)

        rot_strut_alignment = A.GetQuaternion()

        #<3 LEG E (E-D) - shaft of tjoint 
        rot_e = chrono.QuatFromAngleAxis(self.alpha2, chrono.ChVector3d(-1, 0, 0)) #rotation wrt the legs local frame

        e_body = chrono.ChBody()
        e_body.SetMass(self.e_leg_mass)
        e_body.SetInertiaXX(self.e_leg_inertia)
        e_body.SetRot(rot_strut_alignment*rot_e)
        e_frame = e_body.GetFrameCOMToAbs()
    
        e_end_offset_LRF = chrono.ChVector3d(0, 0, self.e_leg_length/2)
        b_end_offset_LRF = chrono.ChVector3d(0, 0, self.b_leg_length/2)
        e_body.SetPos(des_point + e_frame.GetRot().Rotate(-(e_end_offset_LRF + b_end_offset_LRF*2)))

        e_body.EnableCollision(True)
        e_body.AddCollisionShape(chrono.ChCollisionShapeCylinder(self.cmaterial, self.e_leg_radius, self.e_leg_length))
        e_body.GetCollisionModel().SetFamily(self.LANDER_FAMILY)
        e_body.GetCollisionModel().DisallowCollisionsWith(self.LANDER_FAMILY)

        self.sys.AddBody(e_body)

        e_shape = chrono.ChVisualShapeCylinder(self.e_leg_radius, self.e_leg_length)
        e_shape.SetColor(chrono.ChColor(130/255, 150/255, 237/255))
        e_body.AddVisualShape(e_shape)
    
        self.lander_bodies.append(e_body)
        self.lander_radii.append(self.e_leg_radius)
        self.lander_lengths.append(self.e_leg_length)


        #<3 LEG B (B-E) - pistion of tjoint 
        rot_b = rot_e #need to align the piston with the shaft 

        b_body = chrono.ChBody()
        b_body.SetMass(self.b_leg_mass)
        b_body.SetInertiaXX(self.b_leg_inertia)
        b_body.SetRot(rot_strut_alignment*rot_b)
        b_frame = b_body.GetFrameCOMToAbs()
        b_end_offset_LRF = chrono.ChVector3d(0, 0, self.b_leg_length/2)
        b_body.SetPos(des_point + b_frame.GetRot().Rotate(-b_end_offset_LRF))

        b_body.EnableCollision(True)
        b_body.AddCollisionShape(chrono.ChCollisionShapeCylinder(self.cmaterial, self.b_leg_radius, self.b_leg_length))
        b_body.GetCollisionModel().SetFamily(self.LANDER_FAMILY)
        b_body.GetCollisionModel().AllowCollisionsWith(self.LANDER_FAMILY)
        self.sys.AddBody(b_body)

        b_shape = chrono.ChVisualShapeCylinder(self.b_leg_radius, self.b_leg_length)
        b_shape.SetColor(chrono.ChColor(237/255, 187/255, 130/255))
        b_body.AddVisualShape(b_shape)

        self.lander_bodies.append(b_body)
        self.lander_radii.append(self.b_leg_radius)
        self.lander_lengths.append(self.b_leg_length)
       

        #<3 TJOINT BETWEEN LEG E AND LEG B 
        tjoint = chrono.ChLinkLockPrismatic()
        #NOTE: the frame initialization is slightly different from hcff testing, which is potentially why the hcff force model does not need ref_dist offset
        tjoint.Initialize(e_body, b_body, chrono.ChFramed(e_frame.GetPos() + e_frame.GetRot().Rotate(e_end_offset_LRF), e_frame.GetRot()))
        self.sys.AddLink(tjoint)

        tjoint_lim = tjoint.LimitZ()
        tjoint_lim.SetActive(True)
        tjoint_min = 0
        tjoint_lim.SetMin(tjoint_min)
        tjoint_max = self.b_leg_length
        tjoint_lim.SetMax(tjoint_max)

        #====================================================
        #====================================================
        #====================================================
        rest_length = self.b_leg_length

        #<3 HONEYCOMB AT TJOINT 
        honeycomb = chrono.ChLinkTSDA() 
        honeycomb.Initialize(e_body, 
                        b_body, 
                        False, 
                        e_frame.GetPos() + e_frame.GetRot().Rotate(e_end_offset_LRF), 
                        b_frame.GetPos() - b_frame.GetRot().Rotate(b_end_offset_LRF)
                        )
        hcff = HoneycombForceFunctor(b_body, tjoint, rest_length, self.sys, self.fy)
        honeycomb.RegisterForceFunctor(hcff)
        self.sys.AddLink(honeycomb)
        self.honeycomb_list.append(honeycomb)
        self.hcff_list.append(hcff)
        #====================================================
        #====================================================
        #====================================================

        # #<3 POINT D 
        d_point = e_frame.GetPos() + e_frame.GetRot().Rotate(-e_end_offset_LRF)

        #<3 LEG A (A-D) AND LEG C (C-D)
        rot_a = chrono.QuatFromAngleAxis(self.beta, user_rot.Rotate(chrono.ChVector3d(0,0,1)))
        rot_b = chrono.QuatFromAngleAxis(-self.beta, user_rot.Rotate(chrono.ChVector3d(0,0,1)))

        rvec = des_point - base_center

        

        if rvec.Length() < 1e-12:
            rvec = chrono.ChVector3d(1, 0, 0)

        des_point_a = base_center + rot_a.Rotate(rvec)
        des_point_b = base_center + rot_b.Rotate(rvec)
        des_point_a -= user_rot.Rotate(chrono.ChVector3d(0, 0, self.h2_offset))
        des_point_b -= user_rot.Rotate(chrono.ChVector3d(0, 0, self.h2_offset))


        a_body, a_leg_length = self.make_leg_between(
        d_point, des_point_a,
        self.a_leg_radius, self.a_leg_mass, self.a_leg_inertia,
        chrono.ChColor(196/255, 130/255, 237/255)
        )
        a_body.EnableCollision(True)
        a_body.AddCollisionShape(chrono.ChCollisionShapeCylinder(self.cmaterial, self.a_leg_radius, a_leg_length))
        a_body.GetCollisionModel().SetFamily(self.LANDER_FAMILY)
        a_body.GetCollisionModel().DisallowCollisionsWith(self.LANDER_FAMILY)
        a_frame = a_body.GetFrameCOMToAbs()
        a_end_offset_LRF = chrono.ChVector3d(0, 0, a_leg_length/2)


        c_body, c_leg_length = self.make_leg_between(
        d_point, des_point_b,
        self.c_leg_radius, self.c_leg_mass, self.c_leg_inertia,
        chrono.ChColor(235/255, 130/255, 237/255)
        )   
        c_body.EnableCollision(True)
        c_body.AddCollisionShape(chrono.ChCollisionShapeCylinder(self.cmaterial, self.c_leg_radius, c_leg_length))
        c_body.GetCollisionModel().SetFamily(self.LANDER_FAMILY)
        c_body.GetCollisionModel().DisallowCollisionsWith(self.LANDER_FAMILY)
        c_frame = c_body.GetFrameCOMToAbs()
        c_end_offset_LRF = chrono.ChVector3d(0, 0, c_leg_length/2)

        #<3 WELD LEG A AND LEG C TOGETHER AT POINT D (make the wing)
        weld = chrono.ChLinkLockLock()
        weld.Initialize(a_body, c_body, chrono.ChFramed(d_point, chrono.ChQuaterniond(1, 0, 0, 0)))
        self.sys.AddLink(weld)

        #<3 REVOLUTE JOINT TO CONNECT LEG E TO LEG A (AND C) AT POINT D
        z_GRF = chrono.ChVector3d(0, 0, 1)
        e_x_GRF = e_frame.GetRot().Rotate(chrono.ChVector3d(1, 0, 0))
        e_x_GRF.Normalize()

        axis = chrono.Vcross(z_GRF, e_x_GRF)
        axis_len = axis.Length()

        if axis_len < 1e-12:
            if z_GRF.Dot(e_x_GRF) > 0: #already aligned (orientation either same way or opposite)
                rev_rot = chrono.ChQuaterniond(1, 0, 0, 0)  # identity
            else:
                rev_rot = chrono.QuatFromAngleAxis(chrono.CH_PI, chrono.ChVector3d(1, 0, 0))
        else:
            axis.Normalize()
            angle = math.acos(max(-1.0, min(1.0, z_GRF.Dot(e_x_GRF))))
            rev_rot = chrono.QuatFromAngleAxis(angle, axis)

        rev_frame = chrono.ChFramed(d_point, rev_rot)

        rjoint = chrono.ChLinkLockRevolute()
        rjoint.Initialize(e_body, a_body, rev_frame)
        self.sys.AddLink(rjoint)


        #<3 SPHERICAL JOINT TO CONNECT LEGS TO BASE BODY 
        sjoint_b = chrono.ChLinkLockSpherical()
        joint_pos = b_frame.GetPos() + b_frame.GetRot().Rotate(b_end_offset_LRF)
        sjoint_b.Initialize(base_body, b_body, chrono.ChFramed(joint_pos, chrono.ChQuaterniond(1, 0, 0, 0)))
        self.sys.AddLink(sjoint_b)

        sjoint_a = chrono.ChLinkLockSpherical()
        sjoint_a.Initialize(base_body, a_body, chrono.ChFramed(a_frame.GetPos() + a_frame.GetRot().Rotate(a_end_offset_LRF), chrono.ChQuaterniond(1, 0, 0, 0)))
        self.sys.AddLink(sjoint_a)

        sjoint_c = chrono.ChLinkLockSpherical()
        sjoint_c.Initialize(base_body, c_body, chrono.ChFramed(c_frame.GetPos() + c_frame.GetRot().Rotate(c_end_offset_LRF), chrono.ChQuaterniond(1, 0, 0, 0)))
        self.sys.AddLink(sjoint_c)

        #<3 FOOTPAD 
        d_point = e_frame.GetPos() + e_frame.GetRot().Rotate(-e_end_offset_LRF) + e_frame.GetRot().Rotate(chrono.ChVector3d(0, 0, -0.05))
        mesh = chrono.ChTriangleMeshConnected()
        mesh.LoadWavefrontMesh(self.footpad_mesh_file, False, False)

        footpad_body = chrono.ChBody()
        footpad_body.SetMass(self.footpad_mass)
        footpad_body.SetInertiaXX(chrono.ChVector3d(1.0, 1.0, 1.0))
        footpad_body.SetPos(d_point)

     
        rot_footpad_align = chrono.Q_ROTATE_Y_TO_Z
        mesh_correction = chrono.QUNIT
        footpad_body.SetRot(user_rot * rot_footpad_align) #* mesh_correction)
        footpad_body.EnableCollision(True)
        footpad_body.AddCollisionShape(
            chrono.ChCollisionShapeTriangleMesh(self.cmaterial, mesh, False, False)
        )
        footpad_body.GetCollisionModel().SetFamily(self.LANDER_FAMILY)
        footpad_body.GetCollisionModel().DisallowCollisionsWith(self.LANDER_FAMILY)
        self.sys.AddBody(footpad_body)
        vis_mesh = chrono.ChVisualShapeTriangleMesh(mesh)
        vis_mesh.SetBackfaceCull(True)
        vis_mesh.SetColor(chrono.ChColor(200/255, 170/255, 227/255))
        footpad_body.AddVisualShape(vis_mesh)
        self.footpad_bodies.append(footpad_body)

        #<3 SPHERICAL JOINT FOR FOOTPAD
        weld_footpad = chrono.ChLinkLockLock()
        weld_footpad.Initialize(
            e_body,
            footpad_body,
            chrono.ChFramed(d_point, user_rot * rot_footpad_align)
        )
        self.sys.AddLink(weld_footpad)


    #====================================================
    #<3 LANDER MODEL CONSTRUCTION
    #====================================================
    def add_lander(self, lander_bottom_pos = chrono.ChVector3d(0, 0, 0), user_rot = chrono.ChQuaterniond(1, 0, 0, 0)):
        #<3 LANDER BASE 
        base_body = chrono.ChBodyAuxRef() #TODO change to ChBodyAuxRef 
        base_body.SetMass(self.mass)
        base_body.SetInertiaXX(self.inertia)
        base_center = lander_bottom_pos + chrono.ChVector3d(0, 0, self.h2/2.0)
        base_attachment_point = lander_bottom_pos + chrono.ChVector3d(0, 0, self.h2_offset)
        self.base_pos = base_center

        base_body.SetPos(base_center)
        base_body.SetRot(user_rot)
        base_body.SetFrameCOMToAbs(chrono.ChFramed(lander_bottom_pos + self.com_offset, user_rot))
        base_body.EnableCollision(True)
        base_body.AddCollisionShape(chrono.ChCollisionShapeCylinder(self.cmaterial, self.r, self.h2))
        base_body.GetCollisionModel().SetFamily(self.LANDER_FAMILY)
        base_body.GetCollisionModel().DisallowCollisionsWith(self.LANDER_FAMILY)
        self.sys.AddBody(base_body)
        base_shape = chrono.ChVisualShapeCylinder(self.r, self.h2)
        base_shape.SetColor(chrono.ChColor(200/255, 170/255, 227/255))
        base_body.AddVisualShape(base_shape)

        self.lander_bodies.append(base_body)
        self.lander_radii.append(self.r)
        self.lander_lengths.append(self.h2)

        #<3 STRUTS
        base_attachment_point = base_center + user_rot.Rotate(base_attachment_point - base_center)
        for i in range(self.num_struts):
            theta = (2*chrono.CH_PI*i)/self.num_struts
            offset = chrono.ChVector3d(self.r*math.cos(theta), self.r*math.sin(theta), 0)
            des_point = base_attachment_point + user_rot.Rotate(offset)
            self.add_strut(des_point, base_body, base_attachment_point, user_rot)

    def register_lander_on_terrain(self, terrain, sysFSI):
        rot = chrono.QuatFromVec2Vec(chrono.ChVector3d(1,0,0), chrono.ChVector3d(0,0,1))
        # frame = chrono.ChFramed(chrono.ChVector3d(0,0,0), chrono.Q_ROTATE_X_TO_Z)
        frame = chrono.ChFramed(chrono.ChVector3d(0,0,0), rot)
        for i, body in enumerate(self.lander_bodies):
            r = self.lander_radii[i]
            L = self.lander_lengths[i]
            terrain.AddRigidBodyCylinderX(body, frame, r, L, True)


        frame = chrono.ChFramed(chrono.ChVector3d(0,0,0), chrono.QUNIT)
        bce_points = chrono.vector_ChVector3d()

        with open(self.footpad_points_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                x, y, z = map(float, line.split(","))
                bce_points.push_back(chrono.ChVector3d(x, y, z))

        for footpad_body in self.footpad_bodies:
            sysFSI.AddFsiBody(footpad_body, bce_points, frame, False)

    #====================================================
    #<3 LOGGING/STOP CONDITION HELPERS 
    #====================================================

    def get_pos(self): 
        return self.lander_bodies[0].GetPos()

    def hit_ground(self):
        tolerance = 0.5
        return self.lander_bodies[0].GetPos().z <= 0 + self.h1 + (self.h2 / 2.0) - tolerance

    def get_accel(self): 
        return self.lander_bodies[0].GetPosDt2().Length() 

    def get_hc_ref_dist(self): 
        return self.hcff_list[0].get_s_ref()

    def get_energy_absorbed(self):
        total_energy = 0.0

        for hcff in self.hcff_list:
            total_energy += hcff.E_leg

        return total_energy

    def get_impact_velocity(self):
        return math.sqrt(2*9.81*(self.base_pos.z))

    def get_energy_initial(self):
        return self.mass*9.81*(self.base_pos.z)