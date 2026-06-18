# ROCm-safe Ray defaults before any submodule imports ``ray``.
from ChronoRay.rocm_ray_env import prepare_rocm_ray_env

prepare_rocm_ray_env()

from ChronoRay.ChRParamEst import ChRParamEst
from ChronoRay.ChR_Config import ChR_Distr, ChR_SearchAlg
from ChronoRay.ChR_ChronoRay import ChR_ChronoRay
from ChronoRay.ChRBayesOpt import ChRBayesOpt
from ChronoRay.ChRDoE import ChRDoE
from ChronoRay.ChROpt import ChROpt
from ChronoRay.tune_result_io import trials_as_dicts

__all__ = [
    "prepare_rocm_ray_env",
    "ChRParamEst",
    "ChR_Distr",
    "ChR_SearchAlg",
    "ChR_ChronoRay",
    "ChRBayesOpt",
    "ChRDoE",
    "ChROpt",
    "trials_as_dicts",
]
