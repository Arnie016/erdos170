#!/usr/bin/env python3
"""Exact two-replacement existence test for a specified near-ruler, integer arithmetic.
Every complete length-500 ruler must realize 493; all 493-pairs force at least
one new mark in TARGET. Iterate all 666 deletion pairs, all 8 distinguished
inserts; for each, any remaining missing difference d forces second insertion
onto +/-d from a retained mark (including the first new mark).
"""
import json,sys,time
from itertools import combinations
from pathlib import Path
p=Path(sys.argv[1]); o=Path(sys.argv[2]); t0=time.monotonic()
candidate=json.loads(p.read_text()); n=candidate['N']; A=sorted(candidate['marks']);assert n==500 and len(A)==39 and len(set(A))==39
ALL=(1<<(n+1))-2
old=set(A); unused=set(range(n+1))-old
T=sorted(set([a,b] for a in range(8) for b in (a,a+493)) if False else ({7}|set(range(493,500))))
assert set(T)<=unused and {0,500}<=old and T==[7,493,494,495,496,497,498,499]

def diff_mask(vals):
 out=0
 for i,x in enumerate(vals):
  for y in vals[i+1:]:out |= 1<<abs(x-y)
 return out

assert (ALL & ~diff_mask(A)).bit_count()==1 and ((ALL & ~diff_mask(A)).bit_length()-1)==493
# Smoke / control: altered candidate should not accidentally satisfy original scope.
assert diff_mask([0,500]) == (1<<500)
counts={'removed_pairs':0,'distinguished_trials':0,'second_candidates_tested':0,'found':False,'pair_trials_equivalent':0}
found=None
for deleted in combinations(A[1:-1],2):
 counts['removed_pairs']+=1
 R=sorted(old-set(deleted)); base=diff_mask(R)
 for v in T:
  counts['distinguished_trials']+=1
  vals=R+[v]; c=base
  for x in R:c|=1<<abs(v-x)
  missing=ALL & ~c
  if not missing: choices=unused-{v}
  else:
   d=(missing & -missing).bit_length()-1
   choices={x+s*d for x in vals for s in (-1,1)} & unused
   choices.discard(v)
  for w in choices:
   counts['second_candidates_tested']+=1
   newmask=c | (1<<abs(v-w))
   for x in R:newmask|=1<<abs(w-x)
   if (ALL & ~newmask)==0:
    B=sorted(R+[v,w]); assert len(B)==39 and len(set(B))==39
    found={'removed':list(deleted),'added':sorted([v,w]),'marks':B,'missing':[]}
    break
  if found: break
 if found:break
# Exact potential two-insert domain size, not number individually tested:
from math import comb
counts['pair_trials_equivalent']=comb(37,2)*(8*(len(unused)-8)+comb(8,2))
counts['found']=found is not None
res={'domain':'all two-replacements of the fixed 39-mark A with 0,500 preserved','target':T,'counts':counts,'found':found,'elapsed':time.monotonic()-t0}
o.write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res))

