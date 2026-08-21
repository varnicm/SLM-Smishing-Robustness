#!/bin/bash
#SBATCH --job-name=rr-dim2
#SBATCH --account=phd-ssanjarip42
#SBATCH --partition=gpu-warp
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=dim2_%j.log

source ~/.bashrc
conda activate slm
unset PYTHONPATH
cd /work/projects/phd-ssanjarip42/ssanjarip42/reasoning-robustness

python scripts/analyze_explanation.py \
    --preds outputs/expl_all.jsonl \
    --out-dir outputs/analysis/dim2_matched \
    --min-chars 10 --seed 0 --batch-size 16
