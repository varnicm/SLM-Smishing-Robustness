#!/usr/bin/env python3
"""
Find ONE smishing message for a single-message, four-level example table, with:
  - valid, clean variants for character / word / sentence / multi
  - the MULTI variant retains a visible character/word edit (real combination)
  - the message's protected URL appears VERBATIM in the original AND all four
    perturbed variants (so "URLs and codes are preserved verbatim" is visibly true)
  - English, non-adult, short-ish, recognizable

  python pick_single_message.py
"""
import argparse, json, re, collections

BLOCK = ["sultry","sexy","girls","girl ","explicit","intimate","nude","naked",
         "porn","xxx","horny","hookup","escort","date night","get laid","boobs",
         "web cam","webcam","private images","released 4 you","hot singles"]
FRENCH = ["votre","veuillez","colis","vérifier","recevoir","envoyé"]
EDIT_OPS = {"homoglyph","space","misspell","synonym"}

# URL incl. any leading scheme (http(s):// or a bare ://), domain, and path
URL_RE = re.compile(r'(?:https?://|://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?', re.I)

def english(s):
    return s and sum(1 for c in s if ord(c)<128)/len(s) > 0.95 and not any(f in s.lower() for f in FRENCH)
def clean(s):
    low=(s or "").lower(); return not any(b in low for b in BLOCK)

def primary_url(text):
    """the longest URL-like token that has a path or a dotted domain."""
    cands=[m.group(0).rstrip('.,);>') for m in URL_RE.finditer(text or "")]
    cands=[c for c in cands if '/' in c or c.count('.')>=1]
    return max(cands, key=len) if cands else None

def multi_keeps_edit(mods, perturbed):
    for m in mods:
        if m.get("op") in EDIT_OPS:
            rep=(m.get("replacement") or "").strip()
            if len(rep)>=3 and rep in perturbed: return True
    return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/smishtank_perturbed_v065.jsonl")
    ap.add_argument("--min-len", type=int, default=45)
    ap.add_argument("--max-len", type=int, default=135)
    ap.add_argument("--topk", type=int, default=8)
    args=ap.parse_args()

    rows=collections.defaultdict(list); orig={}
    for line in open(args.src):
        r=json.loads(line)
        if (r.get("original_label") or "").lower()!="smishing": continue
        rows[(r["id"], r["attack_class"])].append(r); orig[r["id"]]=r["original_message"]
    def vv(mid, atk): return [r for r in rows.get((mid,atk),[]) if r.get("valid")]

    cands=[]
    for mid,o in orig.items():
        if not (args.min_len<=len(o)<=args.max_len): continue
        if not english(o) or not clean(o): continue
        url=primary_url(o)
        if not url: continue                                   # need a protected URL to show
        need={}; ok=True
        for atk in ("character","word","sentence"):
            vs=vv(mid,atk)
            if not vs: ok=False; break
            if atk=="character":
                vs.sort(key=lambda r: abs(len(r.get("modifications",[]))/max(1,len(o.split()))-0.30))
            else:
                vs.sort(key=lambda r: -r["similarity_score"])
            need[atk]=vs[0]
        if not ok: continue
        mv=[r for r in vv(mid,"multi") if multi_keeps_edit(r.get("modifications",[]), r["perturbed_message"])]
        if not mv: continue
        mv.sort(key=lambda r: -r["similarity_score"]); need["multi"]=mv[0]
        # URL must be verbatim in ALL four perturbed variants
        if not all(url in need[a]["perturbed_message"] for a in ("character","word","sentence","multi")):
            continue
        cands.append((-abs(len(o)-90), mid, o, url, need))

    cands.sort(key=lambda x:x[0], reverse=True)
    print(f"{len(cands)} candidates (URL verbatim in all rows); top {args.topk}:\n")
    for score,mid,o,url,need in cands[:args.topk]:
        print("="*84)
        print(f"{mid}   URL(protected)={url}")
        print(f"  ORIG   : {o}")
        for atk in ("character","word","sentence","multi"):
            r=need[atk]
            print(f"  {atk:9s}: {r['perturbed_message']}")
        print()

if __name__=="__main__":
    main()
