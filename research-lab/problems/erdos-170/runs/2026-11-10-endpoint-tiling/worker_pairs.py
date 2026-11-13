#!/usr/bin/env python3
"""Independent finite checker 1: reconstruct actual ruler differences, N=23."""
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

start=perf_counter()
N=23; t=12; q=6
allowed={1,2,3,6}
valid=[]; checked=0
for a in range(1,t):
    for chosen in combinations(range(1,t),q-1):
        R=(0,)+chosen
        marks=sorted(set((0,a) + tuple(N-r for r in R)))
        assert len(marks)==q+2  # two tails do not overlap for N >= 23
        distances={y-x for i,x in enumerate(marks) for y in marks[i+1:]}
        success=all(N-s in distances for s in range(t))
        checked+=1
        if success: valid.append({'a':a,'R':list(R)})
assert checked==5082
assert [p['a'] for p in valid]==sorted(allowed)
assert valid[2]['R']==[0,1,2,6,7,8]
report={'method':'actual-ruler-pair-differences','domain':'a=1..11; R={0}+5 of [1..11]; N=23; top 12 distances','checked':checked,'accepted':len(valid),'valid':valid,'negative_control_a4_rejected':not any(v['a']==4 for v in valid),'status':'PASS','wall_seconds_internal':round(perf_counter()-start,6)}
Path(__file__).with_name('worker_pairs.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
