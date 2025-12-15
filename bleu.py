# bleu.py
import math
from collections import Counter

def ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

def clipped_precision(cand, ref, n):
    c_ngr = Counter(ngrams(cand, n))
    r_ngr = Counter(ngrams(ref, n))
    overlap = {g: min(c_ngr[g], r_ngr.get(g,0)) for g in c_ngr}
    num = sum(overlap.values())
    den = max(1, sum(c_ngr.values()))
    return num / den

def corpus_bleu(cands, refs, max_n=4):
    # cands/refs: list of token lists (single reference per sample)
    precisions = [0.0]*max_n
    for n in range(1, max_n+1):
        num = den = 0
        for cand, ref in zip(cands, refs):
            c_ngr = Counter(ngrams(cand, n))
            r_ngr = Counter(ngrams(ref, n))
            num += sum(min(c_ngr[g], r_ngr.get(g,0)) for g in c_ngr)
            den += max(1, sum(c_ngr.values()))
        precisions[n-1] = (num/den) if den>0 else 0.0
    # brevity penalty
    cand_len = sum(len(c) for c in cands)
    ref_len  = sum(len(r) for r in refs)
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len/max(1,cand_len))
    if any(p==0 for p in precisions): return 0.0
    score = bp * math.exp(sum(math.log(p) for p in precisions)/max_n)
    return score
