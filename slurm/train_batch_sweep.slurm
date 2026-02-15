#!/bin/bash
#SBATCH --job-name=train_batch_sweep
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=24g
#SBATCH --time=06:00:00
#SBATCH --output=logs/batch_sweep_%j.out
#SBATCH --error=logs/batch_sweep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --mail-user=REPLACE_EMAIL
#SBATCH --array=0-5

# Reset loaded modules to defaults
module reset

# Set up the environment
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"

BATCH_SIZES=("1" "8" "32" "64" "128" "256")
BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

# Create logs directory if it doesn't exist
mkdir -p $SLURM_SUBMIT_DIR/logs

# Navigate to the working directory
cd $SLURM_SUBMIT_DIR

module load python/3

# Activate virtual environment
source ./.venv/bin/activate

# Print some system information
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Number of GPUs: $SLURM_GPUS_PER_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"
nvidia-smi

VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
D_FF=1344
ROPE_THETA=10000.0
LEARNING_RATE=3e-3
MAX_ITERS=10000
BETA1=0.9
BETA2=0.95
EPS=1e-8
WEIGHT_DECAY=0.001
GRAD_CLIP=1.0
WARMUP_ITERS=2000
TRAIN_DATA="artifacts/tinystories_train_encoded.npy"
VAL_DATA="artifacts/tinystories_valid_encoded.npy"
WANDB_PROJECT="ece405-tinystories-batchsweep"

echo "============================================"
echo "Training with batch size: $BATCH_SIZE"
echo "============================================"

# Create checkpoint directory
checkpoint_dir="checkpoints/tinystories_batch_${BATCH_SIZE}"

# Run training
time uv run python -m cs336_basics.train \
      --vocab-size $VOCAB_SIZE \
      --context-length $CONTEXT_LENGTH \
      --d-model $D_MODEL \
      --num-layers $NUM_LAYERS \
      --num-heads $NUM_HEADS \
      --d-ff $D_FF \
      --rope-theta $ROPE_THETA \
      --batch-size $BATCH_SIZE \
      --max-iters $MAX_ITERS \
      --beta1 $BETA1 \
      --beta2 $BETA2 \
      --eps $EPS \
      --weight-decay $WEIGHT_DECAY \
      --grad-clip $GRAD_CLIP \
      --max-lr $LEARNING_RATE \
      --warmup-iters $WARMUP_ITERS \
      --train-data $TRAIN_DATA \
      --val-data $VAL_DATA \
      --checkpoint-dir $checkpoint_dir \
      --wandb-project $WANDB_PROJECT \
      --wandb-run-name "batch_${BATCH_SIZE}"

echo "Job completed at $(date)"
