#!/usr/bin/env python3
"""Independent complete 2-move checker: DIRECTLY tests every eligible final set.
No first-missing pruning; precompute each new mark's difference bitset.
The pairwise added-mark distance is separately included.
"""
from itertools import combinations
import json,time,sys
from pathlib import Path
start=time.monotonic()
d=json.loads(Path(sys.argv[1]).read_text()); A=tuple(d['marks']); n=d['N']; S=set(A)
assert n==500 and len(A)==39 and len(S)==39 and {0,n}<=S
all_bits=(1<<(n+1))-2
missing=lambda B: sorted(k for k in range(1,n+1) if not any(a+k in B for a in B))
assert missing(S)==[493]
assert missing(S|{499})==[]
# Derive target endpoints from ALL candidate representations of d=493.
target=set()
for low in range(n-493+1):
 hi=low+493
 assert not ({low,hi}<=S)
 target.update({low,hi}-S)
target=sorted(target)
outs=sorted(set(range(n+1))-S)
restouts=[w for w in outs if w not in target]
count=0;found=[]
for rem in combinations(A[1:-1],2):
 retained=sorted(S-set(rem))
 # Build baseline by rebuilding every pair distance, not incremental removal.
 base=0
 for i in range(len(retained)):
  a=retained[i]
  for b in retained[i+1:]:base|=1<<abs(a-b)
 gain={}
 for w in outs:
  g=0
  for x in retained:g|=1<<abs(w-x)
  gain[w]=g
 # All unordered pairs with >=1 member in target; no duplicate search.
 for a in target:
  for b in restouts:
   count+=1
   if (base|gain[a]|gain[b]|(1<<abs(a-b)))&all_bits == all_bits:
    found.append({'removed':list(rem),'added':[a,b]})
 for a,b in combinations(target,2):
  count+=1
  if (base|gain[a]|gain[b]|(1<<abs(a-b)))&all_bits == all_bits:
   found.append({'removed':list(rem),'added':[a,b]})
assert count==666*(8*454+28)==2437560
res={'target':target,'total_exactly_tested':count,'complete_replacements':found,'elapsed':time.monotonic()-start,
'baseline_missing':[493],'40_mark_control_complete':True}
Path(sys.argv[2]).write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res))

