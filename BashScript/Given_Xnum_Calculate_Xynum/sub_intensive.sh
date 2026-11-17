#!/bin/bash
#SBATCH -J IntensiveTraining-test
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH -p xahcnormal

#SBATCH --array=0-3

#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err

source ~/miniconda3/bin/activate py310

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

DATA_DIR=/work/home/acychkqsp6/zrm/MasterThesis/Databases/Ver1.9/test_split
OUTPUT_DIR=results
SCRIPT=/work/home/acychkqsp6/zrm/MasterThesis/predict_emergent_properties/GenerateDatabase/IntensiveTraining15.py

MASTER_LIST=master_batch_list.txt

batch_file=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MASTER_LIST")

echo "Batch file: $batch_file"

# ======================
# 开始计时
# ======================

array_start=$(date +%s)

srun --cpu-bind=cores bash -c '

procid=$SLURM_PROCID

input_name=$(sed -n "$((procid+1))p" '"$batch_file"')

if [ -z "$input_name" ]; then
    echo "array=$SLURM_ARRAY_TASK_ID proc=$procid has no input"
    exit 1
fi

input_file='"$DATA_DIR"'/${input_name}

python '"$SCRIPT"' \
    --input "$input_file" \
    --output-root "'"$OUTPUT_DIR"'" \
    --repeat 200 \
    --timespan 800 \
    --job-id "$SLURM_JOB_ID" \
    --array-id "$SLURM_ARRAY_TASK_ID" \
    --proc-id "$SLURM_PROCID"

'

array_end=$(date +%s)

array_runtime=$((array_end-array_start))

array_hours=$((array_runtime/3600))
array_minutes=$(((array_runtime%3600)/60))
array_seconds=$((array_runtime%60))

runtime_log="logs/runtime_${SLURM_ARRAY_TASK_ID}.log"

echo "Array ID: $SLURM_ARRAY_TASK_ID" > "$runtime_log"
echo "Job ID: $SLURM_JOB_ID" >> "$runtime_log"
echo "Started at: $(date -d @$array_start)" >> "$runtime_log"
echo "Finished at: $(date -d @$array_end)" >> "$runtime_log"
echo "Runtime: $array_runtime seconds" >> "$runtime_log"
echo "Runtime: ${array_hours}h ${array_minutes}m ${array_seconds}s" >> "$runtime_log"
