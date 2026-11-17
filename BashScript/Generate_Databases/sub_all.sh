#!/bin/bash

MIN_RICHNESS=3
MAX_RICHNESS=20
REPEATS=1

export REPEATS

mkdir -p logs

TOTAL_TASKS=$(((MAX_RICHNESS - MIN_RICHNESS + 1) * REPEATS))

sbatch --export=ALL,REPEATS="$REPEATS" \
    --array=0-$((TOTAL_TASKS - 1)) \
    sub_each_array.sh