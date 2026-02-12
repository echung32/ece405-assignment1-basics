#!/bin/bash

# submit_job.sh: Submit a SLURM job after replacing placeholders with values from .env

set -e  # Exit on any error

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create a .env file with ACCOUNT and EMAIL variables."
    exit 1
fi

# Source the .env file
source .env

# Check if ACCOUNT and EMAIL are set
if [ -z "$ACCOUNT" ]; then
    echo "Error: ACCOUNT not set in .env file."
    exit 1
fi

if [ -z "$EMAIL" ]; then
    echo "Error: EMAIL not set in .env file."
    exit 1
fi

# Check if SLURM file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <slurm_file>"
    exit 1
fi

SLURM_FILE="$1"

if [ ! -f "$SLURM_FILE" ]; then
    echo "Error: SLURM file '$SLURM_FILE' not found."
    exit 1
fi

# Create a temporary SLURM file
TEMP_SLURM_FILE=$(mktemp "${SLURM_FILE}.XXXXXX")

# Replace placeholders and copy to temp file
sed "s/REPLACE_ACCOUNT/$ACCOUNT/g; s/REPLACE_EMAIL/$EMAIL/g" "$SLURM_FILE" > "$TEMP_SLURM_FILE"

# Submit the job
echo "Submitting job with modified SLURM file: $TEMP_SLURM_FILE"
sbatch "$TEMP_SLURM_FILE"

# Optionally, remove the temp file after submission (uncomment if desired)
rm "$TEMP_SLURM_FILE"

echo "Job submitted successfully."