#!/bin/bash

#SBATCH -J chrbayescali
#SBATCH -o o-chrbayescali-%j.out
#SBATCH -e e-chrbayescali-%j.err
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH -p mi3258x

source /home1/slaton/venvs/chrono-ray/bin/activate

echo "========================================"
echo "Job started: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "========================================"

SECONDS=0

python ./DEMO_ChRBayesCali_slider_crank.py
exit_code=$?

elapsed=$SECONDS

echo "========================================"
echo "Job finished: $(date)"
printf "Wall-clock time: %02d:%02d:%02d\n" \
    $((elapsed/3600)) \
    $(((elapsed%3600)/60)) \
    $((elapsed%60))
echo "Python exit code: $exit_code"
echo "========================================"

exit $exit_code