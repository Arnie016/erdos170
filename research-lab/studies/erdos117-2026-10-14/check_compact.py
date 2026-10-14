#!/usr/bin/env python3
# Public compact-certificate verifier. NOT run as part of Oct 14 local workers.
# See study README for independent already-executed JSONL verification.
from pathlib import Path
from collections import Counter
pairs=[(i,j) for i in range(5) for j in range(i+1,5)]
digits="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def form(mask,x,y):
    rows=[0]*5
    for k,(i,j) in enumerate(pairs):
        if mask>>k&1: rows[i]^=1<<j; rows[j]^=1<<i
    return sum(((rows[i]&y).bit_count()&1) for i in range(5) if x>>i&1)&1
def nonzero(F,x,y):
    return any(form(m,x,y) for m in F)
def check(line,expected):
    idx,a,b,c,k,w,s=line.split()
    idx,a,b,c,k,w=map(int,(idx,a,b,c,k,w))
    assert idx==expected and len(s)==31 and all(q in digits for q in s)
    F=(a,b,c); C=[digits.index(q) for q in s]
    span={a,b,c,a^b,a^c,b^c,a^b^c}
    assert 0 not in span and len(span)==7 and max(F)<1024
    assert w.bit_count()==k and w<(1<<31) and max(C)==k-1 and set(C)==set(range(k))
    for x in range(1,32):
        assert any(nonzero(F,x,1<<i) for i in range(5))
        for y in range(x+1,32):
            noncomm=nonzero(F,x,y)
            if w>>(x-1)&1 and w>>(y-1)&1: assert noncomm
            if noncomm: assert C[x-1]!=C[y-1]
    return k, tuple(sorted(span))
raw=Path(__file__).with_name("certificates.txt").read_text().splitlines()
assert len(raw)==64
keys=[]; hist=Counter()
for i,line in enumerate(raw):
    k,key=check(line,i);keys.append(key);hist[k]+=1
assert len(set(keys))==64
assert dict(hist)=={14:16,13:29,15:11,17:5,11:3}
print("64 matching exact clique/color certificates checked",dict(sorted(hist.items())))
