#!/usr/bin/env python3
"""Independent finite checker 2: truncated generating-polynomial bit masks."""
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

start=perf_counter()
t=12; q=6; full=(1<<t)-1
valid=[]; checked=0
for a in range(1,t):
    for choice in combinations(range(1,t),q-1):
        mask=1|sum(1<<r for r in choice)
        # Support of (1+x^a)*R(x) including all terms, not modulo x^t.
        output_mask=mask | (mask<<a)
        success=(output_mask==full)
        checked+=1
        if success: valid.append([a, [0,*choice]])
assert checked==5082
assert len(valid)==4
assert [x[0] for x in valid]==[1,2,3,6]
for a,R in valid:
    assert q%a==0
    expected=[2*a*j+k for j in range(q//a) for k in range(a)]
    assert R==expected
# controls: a false candidate and a deliberately damaged true candidate
bad=(1|sum(1<<x for x in [1,2,3,4,5]))
assert (bad|(bad<<4))!=full
assert ((1|sum(1<<x for x in [1,2,6,7,9])) | ((1|sum(1<<x for x in [1,2,6,7,9]))<<3))!=full
report={'method':'integer generating-polynomial support','domain':'a=1..11; six monomials R including constant; target 1+x+...+x^11','checked':checked,'accepted':len(valid),'valid':[{'a':a,'R':R} for a,R in valid],'controls':'PASS','status':'PASS','wall_seconds_internal':round(perf_counter()-start,6)}
Path(__file__).with_name('worker_masks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
