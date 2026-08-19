#!/bin/bash

#SBATCH -J rtf
#SBATCH -o o-rtf-%j.out
#SBATCH -e e-rtf-%j.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH -p mi3258x

source /home1/slaton/venvs/chrono-ray/bin/activate

echo "========================================"
echo "========================================"
echo "Job started: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "========================================"
echo "========================================"

SECONDS=0

python ./DEMO_RTF_veh_defm_soil.py 
exit_code=$?

elapsed=$SECONDS

echo "========================================"
echo "========================================"
echo "Job finished: $(date)"
printf "Wall-clock time: %02d:%02d:%02d\n" \
    $((elapsed/3600)) \
    $(((elapsed%3600)/60)) \
    $((elapsed%60))
echo "Python exit code: $exit_code"
echo "========================================"
echo "========================================"

exit $exit_code
