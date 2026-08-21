#!/bin/bash
#SBATCH --job-name=rr-infer-pilot
#SBATCH --account=phd-ssanjarip42
#SBATCH --partition=gpu-warp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=infer_pilot_%j.log

# --- environment (same proven setup as the perturbation job) ---
source ~/.bashrc
conda activate slm
unset PYTHONPATH

export HF_HOME=/work/projects/phd-ssanjarip42/ssanjarip42/reasoning-robustness/hf_cache
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

cd /work/projects/phd-ssanjarip42/ssanjarip42/reasoning-robustness

python -c "import torch; print('CUDA visible in job:', torch.cuda.is_available())"

# --- pilot: 1 model, 4 templates, 50 perturbed records ---
python scripts/run_inference.py \
    --perturbed outputs/smishtank_perturbed.jsonl \
    --out       outputs/preds_pilot.jsonl \
    --models    microsoft/Phi-4-mini-instruct \
    --limit 50

echo "[done] inference pilot finished"
