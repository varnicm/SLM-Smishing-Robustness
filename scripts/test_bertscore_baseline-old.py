"""
test_bertscore_baseline.py
--------------------------
Is the ~0.90 explanation-stability BERTScore MEANINGFUL, or would any two same-label
rationales score that high anyway?

This is the same shuffled-control logic that killed our NLI label-consistency metric.
We must apply it here before claiming rationales are "stable", or a reviewer will.

Three populations (BERTScore F1 between two rationales):
  (A) TRUE PAIR      clean rationale vs its own perturbed rationale (the real metric).
  (B) SAME-LABEL     clean rationale vs a DIFFERENT message's clean rationale with the
                     SAME predicted label. This is the baseline: if models emit generic
                     same-label boilerplate, (B) will be ~as high as (A), and the metric
                     is uninformative.
  (C) CROSS-LABEL    clean rationale vs a different message's clean rationale with the
                     OPPOSITE label. Floor: how low BERTScore goes for genuinely
                     different rationales.

INTERPRETATION
  * A >> B  -> stability is meaningful: a model's rationale for ITS OWN perturbed message
              is more similar than to an unrelated same-label rationale. Good.
  * A ~= B  -> the number reflects same-label boilerplate, not preserved justification.
              The stability claim must be heavily qualified or dropped.
  * B ~= C  -> BERTScore barely separates same-label from cross-label rationales at all;
              it is not tracking rationale content here.

Per model, so we can see if some models are boilerplate-y and others aren't.

Usage:
    python test_bertscore_baseline.py --preds outputs/expl_all.jsonl \\
        --n 150 --bertscore-model roberta-large
"""
import argparse, json, random, re, sys
from collections import defaultdict
import torch

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)


def rationale_text(raw):
    m = _FINAL_RE.search(raw or "")
    return ((raw[:m.start()] if m else raw) or "").strip()


def norm(x):
    x = str(x).strip().lower()
    if x in {"1", "smish", "smishing", "phishing", "phish"}: return "smishing"
    if x in {"0", "ham", "benign", "legitimate", "legit"}:   return "ham"
    return x


def bertscore_batch(refs, cands, model_name, device, batch_size=64, layer=17):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()

    def embed(texts):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
            with torch.no_grad():
                hs = mdl(**enc).hidden_states[layer]
            hs = torch.nn.functional.normalize(hs, dim=-1)
            mask = enc["attention_mask"].bool()
            special = torch.zeros_like(mask)
            for sid in tok.all_special_ids:
                special |= (enc["input_ids"] == sid)
            keep = mask & ~special
            for b in range(hs.size(0)):
                out.append(hs[b][keep[b]])
        return out

    R, C = embed(refs), embed(cands)
    f1s = []
    for r, c in zip(R, C):
        if r.numel() == 0 or c.numel() == 0:
            f1s.append(0.0); continue
        sim = c @ r.T
        p = sim.max(dim=1).values.mean().item()
        rc = sim.max(dim=0).values.mean().item()
        f1s.append(0.0 if p+rc == 0 else 2*p*rc/(p+rc))
    return f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--n", type=int, default=150, help="pairs per population per model")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    rows = [json.loads(l) for l in open(args.preds)]
    rows = [r for r in rows if "cot" in r["template"] or "expl" in r["template"]]

    # clean rationales, indexed; and true pairs (clean vs perturbed, both correct)
    clean = {}
    for r in rows:
        if r["condition"] == "clean":
            clean[(r["model"], r["template"], r["id"])] = r

    # per-model pools of clean rationales by predicted label
    pool = defaultdict(lambda: defaultdict(list))     # model -> label -> [rationale]
    for (m, t, i), r in clean.items():
        pool[m][norm(r["pred"])].append(rationale_text(r["raw"]))

    # true pairs (both-correct)
    true_pairs = defaultdict(list)   # model -> [(clean_rat, pert_rat)]
    for r in rows:
        if r["condition"] != "perturbed":
            continue
        c = clean.get((r["model"], r["template"], r["id"]))
        if not c:
            continue
        if not (norm(c["pred"]) == norm(c["true_label"]) and
                norm(r["pred"]) == norm(r["true_label"])):
            continue
        true_pairs[r["model"]].append(
            (rationale_text(c["raw"]), rationale_text(r["raw"]), norm(r["pred"])))

    models = sorted(true_pairs)
    print(f"[data] models={len(models)}", file=sys.stderr)

    print(f"\n{'model':26s} {'A:true':>9s} {'B:same-lbl':>11s} {'C:cross-lbl':>12s} "
          f"{'A-B gap':>9s}")
    print("-" * 72)

    overall = {"A": [], "B": [], "C": []}
    for m in models:
        tp = true_pairs[m]
        random.shuffle(tp)
        tp = tp[:args.n]
        if not tp:
            continue
        clean_rats = [a for a, _, _ in tp]
        pert_rats  = [b for _, b, _ in tp]
        labels     = [l for _, _, l in tp]

        # (A) true pairs
        A = bertscore_batch(clean_rats, pert_rats, args.bertscore_model, device)

        # (B) same-label: clean rat vs a random DIFFERENT clean rat with same label
        B_refs, B_cands = [], []
        for cr, lab in zip(clean_rats, labels):
            others = [x for x in pool[m][lab] if x != cr]
            if others:
                B_refs.append(cr); B_cands.append(random.choice(others))
        B = bertscore_batch(B_refs, B_cands, args.bertscore_model, device) if B_refs else []

        # (C) cross-label: clean rat vs a random clean rat with OPPOSITE label
        C_refs, C_cands = [], []
        for cr, lab in zip(clean_rats, labels):
            opp = "ham" if lab == "smishing" else "smishing"
            if pool[m][opp]:
                C_refs.append(cr); C_cands.append(random.choice(pool[m][opp]))
        C = bertscore_batch(C_refs, C_cands, args.bertscore_model, device) if C_refs else []

        mean = lambda v: sum(v)/len(v) if v else float("nan")
        overall["A"] += A; overall["B"] += B; overall["C"] += C
        print(f"{m.split('/')[-1]:26s} {mean(A):>9.3f} {mean(B):>11.3f} "
              f"{mean(C):>12.3f} {mean(A)-mean(B):>9.3f}")

    mean = lambda v: sum(v)/len(v) if v else float("nan")
    print("-" * 72)
    print(f"{'POOLED':26s} {mean(overall['A']):>9.3f} {mean(overall['B']):>11.3f} "
          f"{mean(overall['C']):>12.3f} {mean(overall['A'])-mean(overall['B']):>9.3f}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    a, b, c = mean(overall["A"]), mean(overall["B"]), mean(overall["C"])
    gap_ab = a - b
    if gap_ab > 0.05:
        print(f"  A - B = {gap_ab:.3f} (> 0.05): a rationale is meaningfully MORE similar to")
        print("  its own perturbed version than to an unrelated same-label rationale.")
        print("  -> stability is measuring something real; claim is defensible (with wording care).")
    else:
        print(f"  A - B = {gap_ab:.3f} (<= 0.05): the true-pair score is NOT much above the")
        print("  same-label baseline. The ~0.90 largely reflects same-label boilerplate,")
        print("  NOT preserved justification. The stability claim must be heavily qualified.")
    print(f"  (same-label B={b:.3f} vs cross-label C={c:.3f}: BERTScore separates labels "
          f"by {b-c:.3f})")


if __name__ == "__main__":
    main()
