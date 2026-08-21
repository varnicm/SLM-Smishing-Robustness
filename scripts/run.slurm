#!/bin/bash
#SBATCH --job-name=rr-perturb
#SBATCH --partition=gpu-warp
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=perturb_%j.log


# --- environment (avoids the PYTHONPATH conflict that has bitten before) ---
source ~/.bashrc
#conda activate slm-rr
conda activate slm

unset PYTHONPATH
python -c "import torch; print('CUDA visible in job:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

# keep all HF/torch downloads in-project and reused across runs
export HF_HOME=/work/projects/phd-ssanjarip42/ssanjarip42/reasoning-robustness/hf_cache
mkdir -p "$HF_HOME"

#cd ~/reasoning-robustness
cd /work/projects/phd-ssanjarip42/ssanjarip42/reasoning-robustness


# --- one-time setup: download counter-fitted vectors if missing ---
CF=data/counter-fitted-vectors.txt
if [ ! -f "$CF" ]; then
    echo "[setup] downloading counter-fitted vectors..."
    wget -q https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip \
        -O data/cf.zip && unzip -o data/cf.zip -d data/ && rm data/cf.zip
fi

# --- run ---
python scripts/run_perturbations.py \
    --input   data/smishtank_clean.csv \
    --output  outputs/smishtank_perturbed.jsonl \
    --counter-fitted "$CF" \
    --frac 0.30 \
    --seeds 0 1 2 \
    --sim-threshold 0.65 \
    --attacks character word sentence multi \
    --device cuda

echo "[done] perturbation pipeline finished"
