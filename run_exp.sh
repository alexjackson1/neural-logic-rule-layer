#!/bin/bash
#SBATCH --job-name=nlrl
#SBATCH --output=/scratch/prj/formalpaca/nlrl/logs/%A/%a.out
#SBATCH --error=/scratch/prj/formalpaca/nlrl/logs/%A/%a.err
#SBATCH --array=0-2399
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu
#SBATCH --partition=gpu,nmes_gpu


module load cuda
module load python

source env/bin/activate

python /scratch/prj/formalpaca/nlrl/runner.py \
  --save_model \
  --out_dir /scratch/prj/formalpaca/nlrl/new_results
