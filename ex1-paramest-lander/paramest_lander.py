'''
DESIRED PARAMETERS: 
l = 1.825 
beta = 0.30077859136349705
alpha2 = 1.0472
fy = 150
'''

from ChronoRay import ChRParamEst
from Lander import Lander 


param_sample_space = {
        "beta": ChRParamEst.ChR_Distr.uniform(0.261799, 1.0472),
        "alpha2": ChRParamEst.ChR_Distr.uniform(0.872665, 1.825),
        "fy": ChRParamEst.ChR_Distr.loguniform(100, 150)
                    }

target_sim_outputs = {
        "l": 1.825,
        "beta": 0.30077859136349705,
        "alpha2": 1.0472,
        "fy": 150
                    }
