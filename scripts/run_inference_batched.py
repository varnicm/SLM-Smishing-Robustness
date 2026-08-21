"""
run_inference_batched.py
------------------------
BATCHED version of the inference script, for the Dimension 2 explanation re-run.

Why: the original run_inference.py generates one message at a time, leaving the GPU
~45% idle. Batching many prompts through model.generate() at once keeps the GPU busy
and is typically 5-20x faster. This matters for the explanation re-run (long, 512-token
rationales) where per-message latency dominates.

Key correctness detail: batched generation REQUIRES left-padding for decoder-only
models. With right-padding, pad tokens sit between the prompt and the generated tokens
and corrupt the output. We set tokenizer.padding_side="left".

Produces the SAME output schema as run_inference.py (model, template, id, condition,
attack_class, seed, true_label, pred, parse_mode, raw), so analyze_explanation.py and
analyze.py consume it unchanged.

Usage (pilot first!):
    python run_inference_batched.py --perturbed outputs/smishtank_perturbed.jsonl \
        --out outputs/expl_parts/expl_test.jsonl --models Qwen/Qwen3.5-4B \
        --templates t3_neutral_explanation t4_expert_explanation \
        --max-new-expl 512 --raw-cap 4000 --batch-size 16 --limit 30

Then full:
    (drop --limit, run per model)
"""
import argparse, json, os, re, sys, time


# ---- label parsing (identical semantics to run_inference.py) ----------------
_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)

def parse_label(text):
    if not text:
        return "unparsed", "unparsed"
    matches = _FINAL_RE.findall(text)
    if matches:
        return matches[-1].lower(), "strict"
    for line in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        low = line.lower()
        has_s = "smishing" in low or "phishing" in low
        has_h = "ham" in low or "legitimate" in low or "benign" in low
        if has_s and not has_h:
            return "smishing", "fallback"
        if has_h and not has_s:
            return "ham", "fallback"
    return "unparsed", "unparsed"


# ---- prompt templates (must match run_inference.py) -------------------------
_INSTR_TASK = ("Classify the following SMS message as either smishing (SMS phishing) or ham "
               "(legitimate). ")
_FORMAT_PLAIN = "Respond with exactly one line: 'FINAL: smishing' or 'FINAL: ham'."
_FORMAT_EXPL = (
    "First, briefly explain the evidence for the label. Consider suspicious cues "
    "such as urgency, suspicious links, brand impersonation, or requests for personal "
    "info/payment, but also consider benign cues such as ordinary conversation, expected "
    "context, or absence of suspicious requests. Then on the LAST line respond with "
    "exactly 'FINAL: smishing' or 'FINAL: ham'."
)
_PERSONA = "You are an expert SMS security analyst specializing in phishing detection."

def t1_neutral_plain(sms):
    return [{"role": "user", "content": f"{_INSTR_TASK}{_FORMAT_PLAIN}\n\nSMS: {sms}"}]
def t2_expert_plain(sms):
    return [{"role": "system", "content": _PERSONA},
            {"role": "user", "content": f"{_INSTR_TASK}{_FORMAT_PLAIN}\n\nSMS: {sms}"}]
def t3_neutral_explanation(sms):
    return [{"role": "user", "content": f"{_INSTR_TASK}{_FORMAT_EXPL}\n\nSMS: {sms}"}]
def t4_expert_explanation(sms):
    return [{"role": "system", "content": _PERSONA},
            {"role": "user", "content": f"{_INSTR_TASK}{_FORMAT_EXPL}\n\nSMS: {sms}"}]

TEMPLATES = {
    "t1_neutral_plain": (t1_neutral_plain, False),
    "t2_expert_plain": (t2_expert_plain, False),
    "t3_neutral_explanation": (t3_neutral_explanation, True),
    "t4_expert_explanation": (t4_expert_explanation, True),
}


def load_model(name, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"   # REQUIRED for correct batched decoder-only generation
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def build_prompt(tok, messages):
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch(tok, model, batch_messages, explanation, device,
                   max_new_expl, max_new_plain):
    """Generate for a LIST of message-dicts at once. Returns list[str] of decoded
    completions (generated part only), aligned to the input order."""
    import torch
    prompts = [build_prompt(tok, m) for m in batch_messages]
    enc = tok(prompts, return_tensors="pt", truncation=True, max_length=1024,
              padding=True).to(device)
    max_new = max_new_expl if explanation else max_new_plain
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False,
            temperature=None, top_p=None, pad_token_id=tok.pad_token_id,
        )
    # With left-padding, the prompt occupies the first enc["input_ids"].shape[1] columns
    # for EVERY row, so the generated part is everything after that width.
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def iter_records(path, limit=None):
    n = 0
    for line in open(path, encoding="utf-8"):
        yield json.loads(line)
        n += 1
        if limit and n >= limit:
            break


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturbed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--templates", nargs="+",
                    default=list(TEMPLATES.keys()))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-invalid", action="store_true")
    ap.add_argument("--raw-cap", type=int, default=400)
    ap.add_argument("--max-new-expl", type=int, default=256)
    ap.add_argument("--max-new-plain", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="prompts per GPU call. 16-32 is a good range; lower if OOM.")
    args = ap.parse_args()

    def cap(s):
        return s if args.raw_cap <= 0 else s[:args.raw_cap]

    #perturbed, clean = [], {}
    # for r in iter_records(args.perturbed, args.limit):
    #     if not args.include_invalid and not r.get("valid", True):
    #         continue
    #     perturbed.append(r)
    #     clean.setdefault(r["id"], (r["original_message"], r["original_label"]))
    # for r in iter_records(args.perturbed, args.limit):
    #     clean.setdefault(
    #         r["id"],
    #         (r["original_message"], r["original_label"])
    #     )

    #     if not args.include_invalid and not r.get("valid", True):
    #         continue

    #     perturbed.append(r)
    
    # perturbed, clean = [], {}

    # for r in iter_records(args.perturbed, args.limit):
    #     clean.setdefault(
    #         r["id"],
    #         (r["original_message"], r["original_label"])
    #     )

    #     if not args.include_invalid and not r.get("valid", True):
    #         continue

    #     perturbed.append(r)

    # print(
    #     f"[data] {len(perturbed)} valid perturbed | "
    #     f"{len(clean)} unique clean msgs",
    #     file=sys.stderr
    # )

    # if not args.include_invalid and not r.get("valid", True):
    #     continue

    # perturbed.append(r)
    # print(f"[data] {len(perturbed)} valid perturbed | {len(clean)} unique clean msgs",
    #       file=sys.stderr)
    # print(f"[plan] {len(args.models)} models x {len(args.templates)} templates "
    #       f"| batch_size={args.batch_size}", file=sys.stderr)


    perturbed, clean = [], {}

    for r in iter_records(args.perturbed, args.limit):
        clean.setdefault(
            r["id"],
            (r["original_message"], r["original_label"])
        )

        if not args.include_invalid and not r.get("valid", True):
            continue

        perturbed.append(r)

    print(
        f"[data] {len(perturbed)} valid perturbed | "
        f"{len(clean)} unique clean msgs",
        file=sys.stderr
    )
    print(
        f"[plan] {len(args.models)} models x {len(args.templates)} templates "
        f"| batch_size={args.batch_size}",
        file=sys.stderr
    )



    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for mname in args.models:
            t0 = time.time()
            print(f"[model] loading {mname}", file=sys.stderr)
            tok, model = load_model(mname, args.device)

            for tname in args.templates:
                build, explanation = TEMPLATES[tname]

                # (a) clean — batched
                clean_items = list(clean.items())
                for batch in chunks(clean_items, args.batch_size):
                    msgs = [build(msg) for _, (msg, _) in batch]
                    outs = generate_batch(tok, model, msgs, explanation, args.device,
                                          args.max_new_expl, args.max_new_plain)
                    for (mid, (msg, label)), raw in zip(batch, outs):
                        pred, pmode = parse_label(raw)
                        fout.write(json.dumps({
                            "model": mname, "template": tname, "id": mid,
                            "condition": "clean", "attack_class": None, "seed": None,
                            "true_label": label, "pred": pred, "parse_mode": pmode,
                            "raw": cap(raw),
                        }, ensure_ascii=False) + "\n")

                # (b) perturbed — batched
                for batch in chunks(perturbed, args.batch_size):
                    msgs = [build(r["perturbed_message"]) for r in batch]
                    outs = generate_batch(tok, model, msgs, explanation, args.device,
                                          args.max_new_expl, args.max_new_plain)
                    for r, raw in zip(batch, outs):
                        pred, pmode = parse_label(raw)
                        fout.write(json.dumps({
                            "model": mname, "template": tname, "id": r["id"],
                            "condition": "perturbed", "attack_class": r["attack_class"],
                            "seed": r["seed"], "true_label": r["original_label"],
                            "pred": pred, "parse_mode": pmode, "raw": cap(raw),
                        }, ensure_ascii=False) + "\n")

                fout.flush()
                print(f"[done] {mname} / {tname}", file=sys.stderr)

            del model
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
            print(f"[model] {mname} finished in {time.time()-t0:.0f}s", file=sys.stderr)

    print(f"[write] predictions -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
