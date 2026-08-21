
Code, data, and model outputs for the paper:

> **On the Robustness of Small Language Models for Smishing Detection against
> Text Perturbation Attacks** 

This repository contains the full pipeline and all data/outputs behind the paper's
results, so reviewers can inspect and re-run everything.

---

## Which files produce the paper's results

The workspace has many working files (pilots, older versions, experiments). The files
below are the ones the **paper** is built from — start here.

**Data**
- `data/smishtank_clean.csv` — balanced dataset, 1,055 smishing + 1,055 ham (2,110).
- `smish-mandely.csv` — Mendeley SMS corpus, source of the 1,055 benign messages.
- `data/counter-fitted-vectors.txt` — word vectors used for word-level substitution.
- `outputs/smishtank_perturbed_v065.jsonl` — perturbed messages (4 attack levels).
- `outputs/smishtank_perturbed_v065_dedup_sentence.jsonl` — deduplicated sentence set.

**Model predictions**
- `outputs/preds_dedup_2110_final.jsonl` — predictions used for Dimensions 1 & 2.
- `outputs/expl_all.jsonl` — explanation-required generations, used for Dimension 3.

**Analysis outputs (feed the figures and tables)**
- `outputs/analysis/asr_benign_flip_by_prompt.csv` — ASR and benign-flip.
- `outputs/analysis/accuracy_change_by_prompt_msglevel.csv` — ΔAccuracy (message-level).
- `outputs/analysis/transitions_by_prompt.csv` — prediction transitions.
- `outputs/analysis/prompt_level_metrics_bootstrap_1000.csv` — bootstrap CIs.
- `outputs/analysis/metrics.json` — pooled similarity/stability metrics.

**Code**
- Perturbation: `scripts/run_perturbations.py`, `scripts/perturbations.py`
- Inference: `scripts/run_inference_batched.py`
- Analysis: `scripts/analyze_explanation.py`, `scripts/rationale_metrics.py`,
  `scripts/rationale_extract.py`, `scripts/transitions.py`, `scripts/clean_accuracy.py`,
  `scripts/dedup_sentence.py`, `scripts/bootstrap_analysis.py`
- ΔAccuracy (message-level): `recompute_delta_msglevel.py`, `bootstrap_delta_msglevel.py`
- Figures: `visuals/paper_figures.ipynb`, `visuals/figstyle.py`, `visuals/fig_asr.py`
- Verification (re-derives every result): `verify_dim1.py`, `verify_dim2.py`,
  `verify_dim3.py`, `verify_delta.py`, `verify_matched_pairs.py`,
  `compare_benign_selection.py`

---

## Fastest way for a reviewer to check the numbers

No GPU needed — these recompute the paper's results from the committed predictions:

```bash
python verify_dim1.py          # clean accuracy, ASR, benign-flip, transitions
python verify_dim2.py          # prompt-configuration agreement
python verify_delta.py         # ΔAccuracy
python verify_dim3.py          # rationale provision + stability
python verify_matched_pairs.py # rationale similarity scores
```

Each prints per-cell differences against the paper; they should be ~0.

---

## Re-running the full pipeline

Stages 1–2 need the GPU node (A100); stages 3–4 run on CPU.

```bash
# 1. Perturb the dataset
python scripts/run_perturbations.py

# 2. Zero-shot inference: 6 models x 4 prompts x {clean, perturbed}
python scripts/run_inference_batched.py

# 3. Analysis
python scripts/analyze_explanation.py
python scripts/transitions.py
python recompute_delta_msglevel.py

# 4. Figures
jupyter nbconvert --to notebook --execute visuals/paper_figures.ipynb
```

---

## Models

Six instruction-tuned SLMs, evaluated **zero-shot** with **deterministic (greedy)
decoding** (`do_sample=False`, `num_beams=1`):

| Name in paper | Hugging Face repo ID |
|---|---|
| Gemma-4-E4B-it | `google/gemma-4-E4B-it` |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` *(confirm exact ID)* |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` *(confirm exact ID)* |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` |

Reference hardware: NVIDIA A100-PCIE-40GB, AMD EPYC 7713, Slurm (`gpu-warp`).
Software: Python + `transformers`, `torch`, `sentence-transformers`, `bert-score`,
`rouge-score` (pin exact versions in `requirements.txt`).

---

## Dataset notes

The 1,055 benign messages were **uniformly randomly sampled** from the 4,844 ham in the
Mendeley corpus to match the smishing count. The exact 1,055 are fixed in
`smishtank_clean.csv`, so the split is reproducible from the released data.
3 ham with no valid perturbation are excluded from clean-accuracy tables (2,110 → 2,107).

URLs and codes are detected and restored verbatim after perturbation. A bare `://`
fragment with no host is not a detected URL, so it is treated as ordinary text and may
be dropped by paraphrasing — this affects only how an example displays, not any metric.

---

## License


