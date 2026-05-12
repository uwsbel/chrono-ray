#!/bin/bash
#SBATCH --job-name=doe_mccparam
#SBATCH --partition=sbel
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/srv/home/kslaton/chrono-ray/ex2-doe-mccparam/o.out
#SBATCH --error=/srv/home/kslaton/chrono-ray/ex2-doe-mccparam/e.err

source /opt/apps/miniforge/x86_64/24.3.0/etc/profile.d/conda.sh
conda activate chrono-ray 

python doe_mccparam.py
