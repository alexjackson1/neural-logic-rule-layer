#!/bin/bash
#SBATCH --job-name=nlrl
#SBATCH --output=/scratch/prj/formalpaca/nlrl/logs/output_%A_%a.log
#SBATCH --error=/scratch/prj/formalpaca/nlrl/logs/error_%A_%a.log
#SBATCH --array=0-2399
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu
#SBATCH --partition=gpu


module load cuda
module load python

source env/bin/activate

python /scratch/prj/formalpaca/nlrl/runner.py
