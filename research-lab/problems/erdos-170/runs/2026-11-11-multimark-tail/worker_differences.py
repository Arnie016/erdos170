#!/usr/bin/env python3
"""Independent direct integer-distance enumeration for a fixed 35-unit ruler."""
from itertools import combinations
from collections import Counter
import json, math, sys
N=35;t=18;q=6;r=3
all_good=[]; count=0
for a in range(1,9):
    near=[0,a,2*a]
    for extra in combinations(range(1,t),q-1):
        refl=(0,)+extra
        far=[N-i for i in refl]
        marks=sorted(near+far)
        assert len(marks)==r+q and len(set(marks))==r+q
        distances=Counter(marks[j]-marks[i] for i in range(len(marks)) for j in range(i+1,len(marks)))
        count+=1
        if all(distances[d] == 1 for d in range(18,36)):
            all_good.append((a,list(refl)))
assert count==49504
assert {a for a,_ in all_good}=={1,2,3,6}
assert len(all_good)==4
assert all(all_good[i][1] == [3*a*j+k for j in range(6//a) for k in range(a)] for i,(a,_) in enumerate(all_good))
# Changed individual mark in the known-good a=3 right tail, demonstrating sensitivity.
wrong_refl=(0,1,2,8,10,11)
marks=sorted([0,3,6]+[N-i for i in wrong_refl]); dd={marks[j]-marks[i] for i in range(len(marks)) for j in range(i+1,len(marks))}
assert not all(d in dd for d in range(18,36))
print(json.dumps({'worker':'independent_ruler_differences','r':r,'q':q,'N':N,'total_profiles':count,'accepted':[{'a':a,'R':R} for a,R in all_good],'accepted_count':len(all_good),'negative_control':True,'status':'PASS','python':sys.version.split()[0]},sort_keys=True))
