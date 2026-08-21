"""
run_perturbations.py
--------------------
Orchestrates the perturbation pipeline over smishtank_clean.csv.

Produces smishtank_perturbed.jsonl with one record per (message, attack, seed):
  id, original_message, perturbed_message, attack_class, seed,
  original_label, modifications, similarity_score, valid

Both smishing AND ham rows are perturbed (ham serves as a benign-flip control;
metrics separate them downstream). URLs are preserved by the transforms.

Run (on a GPU node, env slm-rr):
    conda activate slm-rr
    unset PYTHONPATH
    export HF_HOME=$HOME/reasoning-robustness/hf_cache
    python run_perturbations.py \
        --input data/smishtank_clean.csv \
        --output outputs/smishtank_perturbed.jsonl \
        --frac 0.30 --seeds 0 1 2 --sim-threshold 0.65
"""

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict

import perturbations_sampled as P


# ----------------------------------------------------------------------------
# Counter-fitted synonym source
# ----------------------------------------------------------------------------

def load_counter_fitted(path, topk=8):
    """Load counter-fitted word vectors and build a nearest-neighbour synonym fn.
    File format: 'word v1 v2 ... vN' per line (standard counter-fitted-vectors.txt).
    Returns a SynonymSource. Uses cosine over a small in-memory matrix.

    For a corpus of a few thousand short messages this is cheap; we restrict the
    vocab to words actually appearing in the dataset to keep the NN search fast."""
    import numpy as np

    vocab, vecs = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 10:
                continue
            vocab.append(parts[0])
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
    mat = np.vstack(vecs)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    index = {w: i for i, w in enumerate(vocab)}

    def neighbours(word):
        i = index.get(word)
        if i is None:
            return []
        sims = mat @ mat[i]
        k = min(topk + 1, len(sims) - 1)          # guard tiny vocab
        if k <= 0:
            return []
        order = np.argpartition(-sims, k)[: k + 1]
        order = order[np.argsort(-sims[order])]
        out = []
        for j in order:
            if j == i:
                continue
            if sims[j] < 0.4:      # counter-fitted synonyms are typically high-sim
                continue
            out.append(vocab[j])
            if len(out) >= topk:
                break
        return out

    return P.SynonymSource(neighbour_fn=neighbours)


# ----------------------------------------------------------------------------
# Similarity gate (sentence-transformers)
# ----------------------------------------------------------------------------

class SimGate:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def score(self, a, b):
        from sentence_transformers import util
        ea, eb = self.model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(ea, eb))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yield r["id"], r["message"], r["label"].strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--counter-fitted", default="data/counter-fitted-vectors.txt")
    ap.add_argument("--frac", type=float, default=0.30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sim-threshold", type=float, default=0.65)
    ap.add_argument("--attacks", nargs="+",
                    default=["character", "word", "sentence", "multi"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip similarity gate (debug/dry runs)")
    args = ap.parse_args()

    needs_syn = any(a in ("word", "multi") for a in args.attacks)
    needs_para = any(a in ("sentence", "multi") for a in args.attacks)

    syn = None
    if needs_syn:
        print(f"[load] counter-fitted vectors from {args.counter_fitted}", file=sys.stderr)
        syn = load_counter_fitted(args.counter_fitted)
    else:
        print("[skip] no synonym source needed for requested attacks", file=sys.stderr)

    paraphraser = P.Paraphraser(device=args.device) if needs_para else None  # lazy load
    if needs_para:
        print("[load] PEGASUS will load on first sentence/multi attack", file=sys.stderr)

    gate = None if args.no_gate else SimGate(device=args.device)

    rows = list(read_rows(args.input))
    print(f"[data] {len(rows)} messages | attacks={args.attacks} | seeds={args.seeds}",
          file=sys.stderr)

    n_written = n_invalid = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for attack in args.attacks:
            for seed in args.seeds:
                rng = random.Random(seed)
                for mid, msg, label in rows:
                    if attack == "character":
                        pert, edits = P.character_level(msg, args.frac, rng)
                    elif attack == "word":
                        pert, edits = P.word_level(msg, args.frac, rng, syn)
                    elif attack == "sentence":
                        pert, edits = P.sentence_level(msg, rng, paraphraser)
                    elif attack == "multi":
                        pert, edits = P.multi_level(msg, args.frac, rng, syn, paraphraser)
                    else:
                        raise ValueError(attack)

                    sim = None
                    valid = True
                    if gate is not None:
                        sim = gate.score(msg, pert)
                        valid = sim >= args.sim_threshold
                        if not valid:
                            n_invalid += 1

                    rec = {
                        "id": mid,
                        "original_message": msg,
                        "perturbed_message": pert,
                        "attack_class": attack,
                        "seed": seed,
                        "original_label": label,
                        "modifications": [asdict(e) for e in edits],
                        "similarity_score": sim,
                        "valid": valid,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
                print(f"[done] attack={attack} seed={seed}", file=sys.stderr)

    print(f"[write] {n_written} variants -> {args.output}", file=sys.stderr)
    if gate is not None:
        print(f"[gate] {n_invalid} below sim<{args.sim_threshold} "
              f"({100*n_invalid/max(n_written,1):.1f}%) flagged valid=false", file=sys.stderr)


if __name__ == "__main__":
    main()
