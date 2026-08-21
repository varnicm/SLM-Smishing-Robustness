"""
run_inference.py  (Task 3)
--------------------------
Feed clean + perturbed SMS to each SLM under 4 prompt templates, parse the
smishing/ham label, and save predictions for ASR / dAcc computation (Task 4).

Design:
  - Clean predictions computed ONCE per unique message (efficient), then joined
    to perturbed variants downstream.
  - Only valid==true perturbed variants are evaluated.
  - Per-model chat template via tokenizer.apply_chat_template.
  - Strict output contract: model must end with 'FINAL: smishing' or 'FINAL: ham'.
    Robust regex parse with a fallback keyword scan; unparseable -> label='unparzed'.

Run a PILOT first (1 model, few messages):
    python run_inference.py --perturbed outputs/smishtank_perturbed.jsonl \
        --out outputs/preds_pilot.jsonl \
        --models microsoft/Phi-4-mini-instruct \
        --limit 50

Full run:
    python run_inference.py --perturbed outputs/smishtank_perturbed.jsonl \
        --out outputs/preds.jsonl
"""

import argparse
import json
import re
import sys
import time

# ---- the 6 cached models (Option A: 4 families, 2 scale pairs) --------------
DEFAULT_MODELS = [
    "microsoft/Phi-4-mini-instruct",
    "google/gemma-4-E4B-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
]

# ---- 4 prompt templates: 2 (reasoning) x 2 (role framing) ------------------
# Each returns a list[dict] chat 'messages' given the SMS text.

_INSTR_TASK = (
    "Classify the following SMS message as either smishing (SMS phishing) or ham "
    "(legitimate). "
)
_FORMAT_PLAIN = "Respond with exactly one line: 'FINAL: smishing' or 'FINAL: ham'."
_FORMAT_COT = (
    "First, briefly note any relevant cues (urgency, suspicious links, brand "
    "impersonation, requests for personal info or payment). Then on the LAST line "
    "respond with exactly 'FINAL: smishing' or 'FINAL: ham'."
)
_PERSONA = "You are an expert SMS security analyst specializing in phishing detection. "


def t1_neutral_plain(sms):
    return [{"role": "user", "content": f"{_INSTR_TASK}{_FORMAT_PLAIN}\n\nSMS: {sms}"}]


def t2_expert_plain(sms):
    return [
        {"role": "system", "content": _PERSONA.strip()},
        {"role": "user", "content": f"{_INSTR_TASK}{_FORMAT_PLAIN}\n\nSMS: {sms}"},
    ]


def t3_neutral_cot(sms):
    return [{"role": "user", "content": f"{_INSTR_TASK}{_FORMAT_COT}\n\nSMS: {sms}"}]


def t4_expert_cot(sms):
    return [
        {"role": "system", "content": _PERSONA.strip()},
        {"role": "user", "content": f"{_INSTR_TASK}{_FORMAT_COT}\n\nSMS: {sms}"},
    ]


TEMPLATES = {
    "t1_neutral_plain": (t1_neutral_plain, False),
    "t2_expert_plain": (t2_expert_plain, False),
    "t3_neutral_cot": (t3_neutral_cot, True),
    "t4_expert_cot": (t4_expert_cot, True),
}

# ---- label parsing ----------------------------------------------------------

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)


def parse_label(text):
    """Extract smishing/ham. Prefer the explicit FINAL: tag (last occurrence);
    fall back to a keyword scan of the last line; else 'unparsed'."""
    if not text:
        return "unparsed"
    matches = _FINAL_RE.findall(text)
    if matches:
        return matches[-1].lower()
    # fallback: scan last non-empty line for a bare label word
    for line in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        low = line.lower()
        has_s = "smishing" in low or "phishing" in low
        has_h = "ham" in low or "legitimate" in low or "benign" in low
        if has_s and not has_h:
            return "smishing"
        if has_h and not has_s:
            return "ham"
    return "unparsed"


# ---- inference --------------------------------------------------------------

def load_model(name, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, device_map=device
        )
    except TypeError:
        # older transformers still uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map=device
        )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def generate(tok, model, messages, cot, device="cuda"):
    import torch
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    max_new = 256 if cot else 16   # CoT needs room to reason; plain just the tag
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            temperature=None, top_p=None, pad_token_id=tok.pad_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]  # only the newly generated part
    return tok.decode(gen, skip_special_tokens=True)


def iter_records(path, limit=None):
    n = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        yield r
        n += 1
        if limit and n >= limit:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturbed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--templates", nargs="+", default=list(TEMPLATES.keys()))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #perturbed records (pilot). Clean set derived from these.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-valid", action="store_true", default=True)
    args = ap.parse_args()

    # Collect the working set of perturbed variants (valid only) + unique cleans.
    perturbed = []
    clean = {}   # id -> (message, label)
    for r in iter_records(args.perturbed, args.limit):
        if args.only_valid and not r.get("valid", True):
            continue
        perturbed.append(r)
        clean.setdefault(r["id"], (r["original_message"], r["original_label"]))

    print(f"[data] {len(perturbed)} valid perturbed | {len(clean)} unique clean msgs",
          file=sys.stderr)
    print(f"[plan] {len(args.models)} models x {len(args.templates)} templates",
          file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as out:
        for mname in args.models:
            t0 = time.time()
            print(f"[model] loading {mname}", file=sys.stderr)
            tok, model = load_model(mname, args.device)

            for tname in args.templates:
                build, cot = TEMPLATES[tname]

                # (a) clean predictions, once per unique message
                for mid, (msg, label) in clean.items():
                    raw = generate(tok, model, build(msg), cot, args.device)
                    out.write(json.dumps({
                        "model": mname, "template": tname, "id": mid,
                        "condition": "clean", "attack_class": None, "seed": None,
                        "true_label": label, "pred": parse_label(raw),
                        "raw": raw[:400],
                    }, ensure_ascii=False) + "\n")

                # (b) perturbed predictions
                for r in perturbed:
                    raw = generate(tok, model, build(r["perturbed_message"]), cot, args.device)
                    out.write(json.dumps({
                        "model": mname, "template": tname, "id": r["id"],
                        "condition": "perturbed", "attack_class": r["attack_class"],
                        "seed": r["seed"], "true_label": r["original_label"],
                        "pred": parse_label(raw), "raw": raw[:400],
                    }, ensure_ascii=False) + "\n")

                out.flush()
                print(f"[done] {mname} / {tname}", file=sys.stderr)

            del model
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
            print(f"[model] {mname} finished in {time.time()-t0:.0f}s", file=sys.stderr)

    print(f"[write] predictions -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
