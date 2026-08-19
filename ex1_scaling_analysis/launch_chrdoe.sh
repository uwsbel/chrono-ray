#!/bin/bash

for NUM_GPUS in 1 2 4 8
do
    sbatch \
        --job-name=chrdoe-${NUM_GPUS}gpu \
        --output=o-chrdoe-${NUM_GPUS}gpus-%j.out \
        --error=e-chrdoe-${NUM_GPUS}gpus-%j.err \
        --export=ALL,NUM_GPUS=${NUM_GPUS} \
        run_chrdoe.sh
done