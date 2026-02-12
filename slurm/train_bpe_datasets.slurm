#!/bin/bash
#SBATCH --job-name=train_bpe_dataset
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_bpe_dataset_%j.out
#SBATCH --error=logs/train_bpe_dataset_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --mail-user=REPLACE_EMAIL

# Reset loaded modules to defaults
module reset

echo "Cache generation job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_CPU MB"

# Create logs directory if it doesn't exist
mkdir -p $SLURM_SUBMIT_DIR/logs

# Navigate to the working directory
cd $SLURM_SUBMIT_DIR

# Load Python module
module load python/3

# Activate virtual environment
source ./.venv/bin/activate

# Print system information
echo "Python version: $(python --version)"
echo "Available CPUs: $(nproc)"

# Run the script
echo "Starting training..."

# time uv run python -m cs336_basics.train_bpe_dataset --dataset tinystories
time uv run python -m cs336_basics.train_bpe_dataset --dataset owt

echo "Job completed at $(date)"