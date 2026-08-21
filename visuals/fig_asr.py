"""
Attack success rate heatmap: models x perturbation type.

!! PLACEHOLDER VALUES !!
Table VI in the draft reports ASR per PROMPT x perturbation. There is no
models x perturbation ASR table (unlike Table VIII for benign-flip), so
the numbers below are the arithmetic mean of T1-T4 from Table VI.

That mean is NOT the pooled rate: ASR is computed over smishing messages
correctly classified BEFORE perturbation, and that denominator differs by
prompt configuration. Recompute at source-message level the same way
Table VIII was produced, then replace `asr` below.
"""
import matplotlib.pyplot as plt
from figstyle import draw_heatmap, save, NEGATIVE

models = ["Gemma-4-E4B-it", "Qwen3.5-4B", "Qwen3.5-9B",
          "Llama-3.1-8B", "Llama-3.2-3B", "Phi-4-mini"]
cols = ["Character", "Word", "Sentence", "Multi-level"]

asr = [
    [0.20, 0.20, 2.24, 2.02],   # Gemma-4-E4B-it
    [0.07, 0.26, 2.01, 1.89],   # Qwen3.5-4B
    [0.02, 0.05, 1.20, 1.07],   # Qwen3.5-9B
    [0.74, 1.50, 4.95, 2.46],   # Llama-3.1-8B
    [0.17, 0.42, 8.02, 3.12],   # Llama-3.2-3B
    [0.07, 0.07, 1.67, 1.14],   # Phi-4-mini
]

fig, ax = plt.subplots(figsize=(3.45, 2.35))
draw_heatmap(asr, models, cols, ax, cmap=NEGATIVE,
             cbar_label="ASR (%)", fmt="{:.2f}")
save(fig, "fig_asr")
print("written")
