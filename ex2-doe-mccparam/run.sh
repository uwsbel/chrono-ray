#!/bin/bash
#SBATCH --job-name=doe_mccparam
#SBATCH --partition=sbel
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/srv/home/kslaton/mcc-euler/build_new/bin/o.out
#SBATCH --error=/srv/home/kslaton/mcc-euler/build_new/bin/e.err

module load conda/miniforge/24.3.0
conda activate ChronoRay

python doe_mccparam.py