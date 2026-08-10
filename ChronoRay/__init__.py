import os, glob

def _is_rocm() -> bool:
    if os.environ.get("ROCM_PATH") or os.path.isdir("/opt/rocm"):
        return True
    return bool(glob.glob("/dev/kfd"))

if _is_rocm():
    from ChronoRay.ChR_ROCmEnv import prepare_rocm_ray_env
    prepare_rocm_ray_env()

from ChronoRay.ChRParamEst import ChRParamEst
from ChronoRay.ChR_Config import ChR_Distr, ChR_SearchAlg
from ChronoRay.ChR_ChronoRay import ChR_ChronoRay
from ChronoRay.ChRBayesOpt import ChRBayesOpt
from ChronoRay.ChRDoE import ChRDoE
from ChronoRay.ChROpt import ChROpt
from ChronoRay.ChRBayesCali import ChRBayesCali
from ChronoRay.ChRCrashProtection import ChRCrashProtection
from ChronoRay.ChRDispersionOpt import ChRDispersionOpt
from ChronoRay.ChRConvergeTest import ChRConvergeTest