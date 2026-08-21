#!/bin/bash
# run_gemma_t3t4.sh
# ---------------------------------------------------------------------------
# Gemma got cancelled after finishing t1 + t2 (complete) and 3,314 partial t3.
# This runs ONLY the two explanation templates (t3, t4), batched, to a separate
# file. Merge into the main Gemma file afterward.
#
# Steps:
#   1. (once) back up + strip the main file to t1+t2 only:
#        cp outputs/preds_google_gemma-4-E4B-it.jsonl outputs/preds_google_gemma-4-E4B-it.jsonl.bak
#        python -c "import json; keep=[l for l in open('outputs/preds_google_gemma-4-E4B-it.jsonl') if json.loads(l)['template'] in ('t1_neutral_plain','t2_expert_plain')]; open('outputs/preds_google_gemma-4-E4B-it.jsonl','w').writelines(keep); print('kept',len(keep))"
#   2. bash run_gemma_t3t4.sh
#   3. after it finishes, merge:
#        cat outputs/preds_gemma_t3t4.jsonl >> outputs/preds_google_gemma-4-E4B-it.jsonl
#        wc -l outputs/preds_google_gemma-4-E4B-it.jsonl   # expect 80932

WORKDIR=/work/projects/phd-ssanjarip42/ssanjarip42/reasoning-robustness
cd "$WORKDIR"
mkdir -p logs

M="google/gemma-4-E4B-it"
TAG=$(echo "$M" | tr '/' '_')

sbatch --account=phd-ssanjarip42 --job-name="g-t3t4" \
       --partition=gpu-warp --gres=gpu:1 --mem=48G --cpus-per-task=4 --time=6:00:00 \
       --output="logs/batched_gemma_t3t4_%j.log" \
       --wrap "source ~/.bashrc; conda activate slm; unset PYTHONPATH; \
               export HF_HOME=\$HOME/.cache/huggingface; export TMPDIR=\$HOME/tmp; mkdir -p \$TMPDIR; \
               export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
               cd $WORKDIR; \
               python scripts/run_inference_batched.py \
                   --perturbed outputs/smishtank_perturbed.jsonl \
                   --out outputs/preds_gemma_t3t4.jsonl \
                   --models '$M' \
                   --templates t3_neutral_explanation t4_expert_explanation \
                   --max-new-expl 512 --raw-cap 4000 --batch-size 16"

echo "submitted Gemma t3+t4 -> outputs/preds_gemma_t3t4.jsonl"
echo "when done:  cat outputs/preds_gemma_t3t4.jsonl >> outputs/preds_google_gemma-4-E4B-it.jsonl"
