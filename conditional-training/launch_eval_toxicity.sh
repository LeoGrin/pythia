#!/bin/bash

#SBATCH --account=neox
#SBATCH --partition=a40x
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=cond_eval
#SBATCH --output=cond_eval_%j.out
#SBATCH --error=cond_eval_%j.err

module load cuda/11.8  # Example, adjust according to your system's module availability
source activate transformers  # Replace with your environment activation command if needed

srun python run_toxicity.py --model hails/cond-410m-20btoks --num_samples 64000 --batch_size 64 --special_token toxic
