#!/usr/bin/env python3
"""
v2 — pin down HOW the 1,055 benign were drawn from smish-mandely.csv, restricting
comparisons to ham-vs-ham so length signals aren't confounded by smishing/spam.

  python compare_benign_selection.py --mendeley smish-mandely.csv --clean outputs/smishtank_clean.csv
"""
import argparse, csv, collections, statistics as st

def norm(s): return " ".join(str(s or "").split())

def load(path):
    rows=list(csv.DictReader(open(path, newline="", encoding="utf-8", errors="replace")))
    cols=rows[0].keys()
    text=max(cols, key=lambda c: st.mean(len(str(r.get(c) or "")) for r in rows))
    label=None
    for c in cols:
        v={str(r.get(c) or "").strip().lower() for r in rows[:200]}
        if v & {"ham","spam","smishing","smish","legit","benign"}: label=c; break
    return rows, text, label

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mendeley", required=True)
    ap.add_argument("--clean", default="outputs/smishtank_clean.csv")
    a=ap.parse_args()
    mrows, mtext, mlabel = load(a.mendeley)
    is_ham=lambda r: str(r.get(mlabel)).strip().lower() in ("ham","legit","benign")

    # ham rows in file order, ranked 0..H-1
    ham_pos=[i for i,r in enumerate(mrows) if is_ham(r)]
    rank={p:k for k,p in enumerate(ham_pos)}
    idx=collections.defaultdict(list)
    for i,r in enumerate(mrows): idx[norm(r.get(mtext))].append(i)

    clean=[r for r in csv.DictReader(open(a.clean, newline="", encoding="utf-8", errors="replace"))
           if (r.get("label") or "").strip().lower()=="ham"]
    sel_pos=[]; unmatched=0; nonham=0
    for r in clean:
        h=idx.get(norm(r.get("message")))
        if not h: unmatched+=1; continue
        p=h[0]; sel_pos.append(p)
        if p not in rank: nonham+=1
    sel_pos.sort()
    print(f"Mendeley ham={len(ham_pos)}  total={len(mrows)}   selected={len(sel_pos)}  "
          f"unmatched={unmatched}  selected-but-not-ham={nonham}")

    ranks=sorted(rank[p] for p in sel_pos if p in rank)
    H=len(ham_pos)
    print("\n== among HAM only ==")
    print(f"  first {len(ranks)} ham?  {ranks==list(range(len(ranks)))}")
    print(f"  contiguous ham block?   {ranks==list(range(ranks[0], ranks[0]+len(ranks)))}")
    gaps=[b-a for a,b in zip(ranks, ranks[1:])]
    mg=st.mean(gaps); sg=st.pstdev(gaps)
    print(f"  ham-rank gaps: mean={mg:.2f} std={sg:.2f} CV={sg/mg:.2f}  "
          f"(expected mean {H/len(ranks):.2f}; CV~0=systematic, CV~1=random)")

    # length: selected ham vs UNSELECTED ham
    selset=set(sel_pos)
    sl=[len(str(mrows[p].get(mtext) or "")) for p in sel_pos]
    ul=[len(str(mrows[p].get(mtext) or "")) for p in ham_pos if p not in selset]
    def pct(x,q):
        x=sorted(x); return x[min(len(x)-1,int(q*len(x)))]
    print("\n== length, ham-vs-ham ==")
    print(f"  selected ham : mean={st.mean(sl):.0f} med={st.median(sl):.0f} p10={pct(sl,.1)} p90={pct(sl,.9)} max={max(sl)}")
    print(f"  unselected ham: mean={st.mean(ul):.0f} med={st.median(ul):.0f} p10={pct(ul,.1)} p90={pct(ul,.9)} max={max(ul)}")
    # shortest-N test: overlap of selected with the 1055 shortest ham
    shortest=set(sorted(ham_pos, key=lambda p: len(str(mrows[p].get(mtext) or "")))[:len(sel_pos)])
    print(f"  overlap of selected with the {len(sel_pos)} SHORTEST ham: "
          f"{len(selset&shortest)/len(sel_pos):.1%}  (100%=took shortest; ~22%=length-blind)")

    # dedup: did selection avoid duplicate texts?
    dup_all=sum(1 for c in collections.Counter(norm(r.get(mtext)) for r in mrows if is_ham(r)).values() if c>1)
    sel_texts=[norm(mrows[p].get(mtext)) for p in sel_pos]
    dup_sel=len(sel_texts)-len(set(sel_texts))
    print(f"\n  duplicate ham texts in file: {dup_all};  duplicates WITHIN selected: {dup_sel} "
          f"({'deduped' if dup_sel==0 else 'not deduped'})")

if __name__=="__main__":
    main()
