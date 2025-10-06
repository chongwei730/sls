#!/bin/bash
#SBATCH --time=13:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --account=jusun
#SBATCH --gres=gpu:4
#SBATCH --output=exp_log/exp_%a.out
#SBATCH --error=exp_log/exp_%a.err
#SBATCH --job-name=line_search
#SBATCH -p a100-4,apollo_agate

eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate sls



gpuid=5
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
python trainval.py -e 10-06 -sb ./10-06_results -d ./data -r 4




