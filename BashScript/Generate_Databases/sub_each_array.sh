#!/bin/bash
#SBATCH -J ver1.9.6_3-20_repeat50
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH -p xahcnormal
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err

source ~/miniconda3/bin/activate py310

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT=/work/home/acychkqsp6/zrm/MasterThesis/predict_emergent_properties/GenerateDatabase/NumStable15.py

# ---------- First layer: array task settings ----------

# ---------- User settings ----------
array_id=$SLURM_ARRAY_TASK_ID
richness=$((3 + array_id / REPEATS))
repeat_id=$((array_id % REPEATS + 1))

job_id=$SLURM_JOB_ID

# Each array task gets its own log directory
log_dir="logs/sp${richness}/rep${repeat_id}_job${job_id}"
mkdir -p "$log_dir"

global_start=$(date +%s)
echo "Array task $array_id, richness $richness, repeat $repeat_id started at $(date)" > "$log_dir/global.log"

# Pass settings to the inner task processes
export RICHNESS=$richness
export REPEAT_ID=$repeat_id
export LOG_DIR=$log_dir

# ---------- Second layer: parallel processes ----------
srun --cpu-bind=cores bash -c '
    task_id=$SLURM_PROCID
    start_time=$(date +%s)

    echo "Task $task_id start at $(date)" > "$LOG_DIR/task_${task_id}.log"

    python '"$SCRIPT"' "$task_id" "$SLURM_JOB_ID" \
        --richness "$RICHNESS" \
        --datasize 20 \
        --repeat 50 \
        --timespan 800 \
        > "$LOG_DIR/task_${task_id}.out" \
        2> "$LOG_DIR/task_${task_id}.err"

    exit_code=$?

    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    runtime_hours=$((runtime / 3600))
    runtime_minutes=$(((runtime % 3600) / 60))
    runtime_seconds=$((runtime % 60))

    echo "Task $task_id end at $(date)" >> "$LOG_DIR/task_${task_id}.log"
    echo "Task $task_id exit code: $exit_code" >> "$LOG_DIR/task_${task_id}.log"
    echo "Task $task_id runtime: ${runtime_hours}h ${runtime_minutes}m ${runtime_seconds}s" >> "$LOG_DIR/task_${task_id}.log"

    exit $exit_code
'

srun_exit_code=$?

global_end=$(date +%s)
global_runtime=$((global_end - global_start))

global_hours=$((global_runtime / 3600))
global_minutes=$(((global_runtime % 3600) / 60))
global_seconds=$((global_runtime % 60))

echo "Array task $array_id, richness $richness, repeat $repeat_id finished at $(date)" >> "$log_dir/global.log"
echo "srun exit code: $srun_exit_code" >> "$log_dir/global.log"
echo "Total runtime: ${global_hours}h ${global_minutes}m ${global_seconds}s" >> "$log_dir/global.log"

exit $srun_exit_code
