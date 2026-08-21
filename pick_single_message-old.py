#!/usr/bin/env python3
"""
Find ONE smishing message good for a single-message, four-level table:
  - valid, clean variants for character / word / sentence / multi
  - the MULTI variant must retain a visible character/word edit (i.e. the final
    paraphrase did NOT launder everything away), so it reads as a real combination
  - English, non-adult, short-ish, recognizable

Reads outputs/smishtank_perturbed_v065.jsonl (has modifications + perturbed_message).
Prints the top candidates with the chosen variant per level; pick one and I'll
build the table.

  python pick_single_message.py
"""
import argparse, json, collections

BLOCK = ["sultry","sexy","girls","girl ","explicit","intimate","nude","naked",
         "porn","xxx","horny","hookup","escort","date night","get laid","boobs",
         "web cam","webcam","private images","released 4 you","hot singles"]
FRENCH = ["votre","veuillez","colis","vérifier","recevoir","envoyé"]
EDIT_OPS = {"homoglyph","space","misspell","synonym"}

def english(s):
    if not s: return False
    return sum(1 for c in s if ord(c)<128)/len(s) > 0.95 and not any(f in s.lower() for f in FRENCH)

def clean(s):
    low=(s or "").lower(); return not any(b in low for b in BLOCK)

def multi_keeps_edit(mods, perturbed):
    """True if a char/word edit's replacement survives in the final multi text."""
    for m in mods:
        if m.get("op") in EDIT_OPS:
            rep=(m.get("replacement") or "").strip()
            if len(rep)>=3 and rep in perturbed:
                return True
    return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/smishtank_perturbed_v065.jsonl")
    ap.add_argument("--min-len", type=int, default=45)
    ap.add_argument("--max-len", type=int, default=130)
    ap.add_argument("--topk", type=int, default=6)
    args=ap.parse_args()

    rows=collections.defaultdict(list)      # (id,attack) -> [row,...]
    orig={}
    for line in open(args.src):
        r=json.loads(line)
        if (r.get("original_label") or "").lower()!="smishing": continue
        rows[(r["id"], r["attack_class"])].append(r)
        orig[r["id"]]=r["original_message"]

    def valid_variants(mid, atk):
        return [r for r in rows.get((mid,atk),[]) if r.get("valid")]

    cands=[]
    for mid,o in orig.items():
        if not (args.min_len<=len(o)<=args.max_len): continue
        if not english(o) or not clean(o): continue
        need={}
        ok=True
        for atk in ("character","word","sentence"):
            vs=valid_variants(mid,atk)
            if not vs: ok=False; break
            # character: closest to ~30% edits; others: highest sim
            if atk=="character":
                vs.sort(key=lambda r: abs(len(r.get("modifications",[]))/max(1,len(o.split()))-0.30))
            else:
                vs.sort(key=lambda r: -r["similarity_score"])
            need[atk]=vs[0]
        if not ok: continue
        # multi that keeps a visible edit
        mv=[r for r in valid_variants(mid,"multi") if multi_keeps_edit(r.get("modifications",[]), r["perturbed_message"])]
        if not mv: continue
        mv.sort(key=lambda r: -r["similarity_score"]); need["multi"]=mv[0]
        score=-abs(len(o)-90)
        cands.append((score, mid, o, need))

    cands.sort(key=lambda x:x[0], reverse=True)
    print(f"{len(cands)} single-message candidates (multi keeps a visible edit); top {args.topk}:\n")
    for score, mid, o, need in cands[:args.topk]:
        print("="*82)
        print(f"{mid}   ORIG: {o}")
        for atk in ("character","word","sentence","multi"):
            r=need[atk]
            ne=len([m for m in r.get('modifications',[]) if m.get('op') in EDIT_OPS])
            print(f"  {atk:9s} seed{r['seed']} sim={r['similarity_score']:.3f} edits={ne}: {r['perturbed_message']}")
        print()

if __name__=="__main__":
    main()
