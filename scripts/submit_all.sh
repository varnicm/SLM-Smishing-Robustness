#!/bin/bash
# submit_all.sh — launch one inference job per model (parallel).
# Each job writes outputs/preds_<modeltag>.jsonl; concatenate when all finish.
#
# Usage:  bash scripts/submit_all.sh
# Then:   squeue -u ssanjarip42        # watch all 6
# After all done:  cat outputs/preds_*.jsonl > outputs/preds.jsonl

MODELS=(
  "microsoft/Phi-4-mini-instruct"
  "google/gemma-4-E4B-it"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
)

WORKDIR=/work/projects/phd-ssanjarip42/ssanjarip42/reasoning-robustness
mkdir -p "$WORKDIR/logs"

for M in "${MODELS[@]}"; do
    TAG=$(echo "$M" | tr '/' '_')          # microsoft_Phi-4-mini-instruct
    sbatch --account=phd-ssanjarip42 \
           --job-name="inf-${TAG:0:12}" \
           --partition=gpu-warp \
           --gres=gpu:1 \
           --mem=48G \
           --cpus-per-task=4 \
           --time=24:00:00 \
           --output="$WORKDIR/logs/inf_${TAG}_%j.log" \
           --wrap "source ~/.bashrc; conda activate slm; unset PYTHONPATH; \
                   export HF_HOME=$WORKDIR/hf_cache; export TMPDIR=\$HOME/tmp; mkdir -p \$TMPDIR; \
                   cd $WORKDIR; \
                   python -c 'import torch;print(\"CUDA:\",torch.cuda.is_available())'; \
                   python scripts/run_inference.py \
                       --perturbed outputs/smishtank_perturbed.jsonl \
                       --out outputs/preds_${TAG}.jsonl \
                       --models '$M'"
    echo "submitted: $M -> outputs/preds_${TAG}.jsonl"
done
