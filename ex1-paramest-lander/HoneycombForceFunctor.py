'''
DESCRIPTION: 
custom chrono force functor for honeycomb force model 

=====================================================
API
=====================================================
HoneycombForceFunctor(body, tjoint, ref_dist, sys)
DESCRIPTION: constructor
    - sys: assumes system is of type NSC (non-smooth contact)
    - fy: maximum honeycomb force
    - returns a HoneycombForceFunctor object

evaluate(time, rest_length, length, vel, tsda_link)
DESCRIPTION: evaluates the honeycomb force
        - note: this function is called by the chrono system at each time step
        - note: most of the parameters are unused (requirement of chrono force functor)
        - note: there is logging for plots (i needed it for my implementation), feel free to remove it if not needed 
    - time: current time
    - rest_length: rest length of the honeycomb (unused)
    - length: current length of the honeycomb (unused)
    - vel: current velocity of the honeycomb (unused)
    - tsda_link: tsda link of the honeycomb (unused)
    - returns the honeycomb force
'''
import pychrono.core as chrono

# Optional Irrlicht import — not required for headless Ray workers / parameter estimation.
try:
    import pychrono.irrlicht as chronoirr  # noqa: F401
except ImportError:
    import types
    import sys

    sys.modules.setdefault("pychrono.irrlicht", types.ModuleType("pychrono.irrlicht"))
    chronoirr = sys.modules["pychrono.irrlicht"]

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv 



class HoneycombForceFunctor(chrono.ForceFunctor):
    def __init__(self, body, tjoint, ref_dist, sys, fy):
        super().__init__() 

        self.ref_dist = ref_dist
        self.tjoint = tjoint
        self.body = body 

        self.bl = ref_dist 
        self.hc_length = 0.75*ref_dist
        self.bn = self.bl - self.hc_length 
        self.s_ref = self.bl - self.hc_length

        #max hc force and force tracking 
        self.fy = fy
        self.lambda_stash = 0.0 
        self.E_leg = 0.0 

        #system 
        self.sys = sys

        #for plots 
        self.last_x = []  
        self.last_v = [] 
        self.last_bn = [] 
        self.last_lambda = [] 
        self.last_t = [] 
        self.last_fn = [] 

        self.last_xfree = []
        self.last_afree = []
        self.last_ftot = []
        self.last_l_star = []
        self.last_xn = []
        self.last_vn = []
        self.last_bn_internal = []



    def evaluate(self, time, rest_length, length, vel, tsda_link):
        h = self.sys.GetStep() #1e-4
        m = self.body.GetMass()

        xn = self.tjoint.GetDistance()

        R_joint = self.tjoint.GetFrame2Abs().GetRot()

        
        vn = R_joint.RotateBack(self.body.GetPosDt()).z 
        vn = (-1)*vn

        tjoint_force = R_joint.Rotate(self.tjoint.GetReaction2().force) 

        ftot = self.body.GetAppliedForce() #+ self.body.GetContactForce() + tjoint_force
        ftot = R_joint.RotateBack(ftot)

        if self.bn == self.bl:
            fn = ftot.z 
        else:
            if self.lambda_stash > 0:
                fn = ftot.z - self.lambda_stash
            else:
                fn = ftot.z + self.lambda_stash
            
        fn = (-1)*fn

        afree = (1/m)*fn
        xfree = xn + h*vn + ((h**2)*afree)

        l = 0.0 #returned honeycomb force 

        #1. no contact 
            #1a. honeycomb force is zero (l = 0)
            #1b. honeycomb position remains unchanged (self.bn = self.bn)

        #2. contact 
        if xfree >=  self.bn: 
            #candidate honeycomb force 
            l_star = fn - ((m/(h**2))*(self.bn - xn - (h*vn)))

            #2a. in order to remain in contact, honeycomb would have to "pull" mass (violates unilateral constraint)
            if l_star <= 0: 
                pass 
                #honeycomb force is zero (l = 0) 
                #honeycomb position remains unchanged (self.bn = self.bn)
            
            #2b. in contact, but no crushing 
            elif l_star <= self.fy: 
                l = l_star 
                #honeycomb position remains unchanged (self.bn = self.bn)

            #2c. in contact and crushing occurs (plastic deformation) 
            else: 
                l = self.fy 
                xn_p1_tent = xfree - (h**2/m)*self.fy 
                bn_tent = min(xn_p1_tent, self.bl)


                if bn_tent >= self.bl:
                    self.bn = self.bl 
                    l = fn - ((m/(h**2))*(self.bl - xn - (h*vn)))
                    l = max(l, 0.0)
                else:
                    self.bn = bn_tent

        self.last_x.append(xn)
        self.last_v.append(vn)
        self.last_bn.append(self.bn)
        self.last_fn.append(fn)
        self.last_t.append(self.sys.GetChTime())

        self.last_xfree.append(xfree)
        self.last_afree.append(afree)
        self.last_ftot.append(ftot.z)
        self.last_xn.append(xn)
        self.last_vn.append(vn)
        self.last_bn_internal.append(self.bn)

        if(len(self.last_bn) > 1):
            dbn = self.bn - self.last_bn[-2]  # > 0 only when crushing actually occurs (case 2c)
            if dbn > 0:
                self.E_leg += self.fy * dbn

        # handle l_star safely
        try:
            self.last_l_star.append(l_star)
        except:
            self.last_l_star.append(0.0)

        if self.bl == self.bn:
            self.lambda_stash = 0.0
            self.last_lambda.append(0.0)
            return float(0)

        else:
            self.last_lambda.append(l)
            self.lambda_stash = l
            return float(l)


    def reset(self): 
        self.lambda_stash = 0.0 
        self.bn = self.bl - self.hc_length 

    def write_csv(self, filename):
        n = len(self.last_t)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # header
            writer.writerow([
                "t",
                "xn",
                "vn",
                "bn",
                "fn",
                "lambda",
                "xfree",
                "afree",
                "ftot_z",
                "l_star"
            ])

            for i in range(n):
                writer.writerow([
                    self.last_t[i],
                    self.last_x[i],
                    self.last_v[i],
                    self.last_bn[i],
                    self.last_fn[i],
                    self.last_lambda[i] if i < len(self.last_lambda) else 0.0,
                    self.last_xfree[i] if i < len(self.last_xfree) else 0.0,
                    self.last_afree[i] if i < len(self.last_afree) else 0.0,
                    self.last_ftot[i] if i < len(self.last_ftot) else 0.0,
                    self.last_l_star[i] if i < len(self.last_l_star) else 0.0,
                ])