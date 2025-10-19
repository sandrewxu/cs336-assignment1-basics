#!/bin/bash
#SBATCH --job-name=bpe_owt          # Job name
#SBATCH --output=logs/bpe_owt_%j.out  # Standard output log (%j expands to jobID)
#SBATCH --error=logs/bpe_owt_%j.err   # Standard error log
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=32          # Number of CPU cores (adjust based on node availability)
#SBATCH --mem=100G                   # Total memory (OpenWebText is ~40GB text, needs headroom)
#SBATCH --time=12:00:00             # Time limit (4 hours should be sufficient)
#SBATCH --partition=day             # Partition name

# Exit on first error
set -e

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "========================================"


# Load any required modules (adjust based on your HPC environment)
module load Python/3.10.8-GCCcore-12.2.0
# module load gcc/11.2.0   # Uncomment if needed for compilation

# Activate UV virtual environment
cd /gpfs/radev/home/ax46/project/cs336/cs336-assignment1-basics

# Make sure uv is available
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Sync/create the virtual environment if it doesn't exist
uv sync

# Activate the UV virtual environment
source .venv/bin/activate

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Configuration variables - ADJUST THESE FOR YOUR NEEDS
INPUT_PATH="data/owt_train.txt"           # Path to OpenWebText training data
VOCAB_SIZE=32000                           # Desired vocabulary size (32k is common)
OUTPUT_PREFIX="owt_tokenizer"              # Prefix for output files
OUTPUT_PATH="results/OpenWebText"          # Output directory

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Run the BPE training
echo "Starting BPE training..."
echo "Input: $INPUT_PATH"
echo "Vocab size: $VOCAB_SIZE"
echo "Output: $OUTPUT_PATH/$OUTPUT_PREFIX"
echo "========================================"

python -m cs336_basics.tokenizer.run_train_bpe \
    --input_path "$INPUT_PATH" \
    --vocab_size "$VOCAB_SIZE" \
    --output_prefix "$OUTPUT_PREFIX" \
    --output_path "$OUTPUT_PATH"

echo "========================================"
echo "Job completed at: $(date)"
echo "Output files saved to: $OUTPUT_PATH"

# Calculate and display job statistics
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,MaxVMSize

