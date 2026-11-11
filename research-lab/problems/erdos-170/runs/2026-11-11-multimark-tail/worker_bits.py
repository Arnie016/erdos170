#!/usr/bin/env python3
"""Complete bitset sumset census, r=3,q=6,t=18, with frozen acceptance."""
from itertools import combinations
import json, math, sys
r,q=3,6
t=r*q
mask=(1<<t)-1
records=[]
checked=0
for a in range(1,(t-2)//(r-1)+1):
    L=[j*a for j in range(r)]
    for tail in combinations(range(1,t),q-1):
        R=(0,)+tail
        shifts=sum(1<<k for k in R)
        cover=0
        for l in L:
            cover |= shifts<<l
        checked+=1
        if cover==mask:
            records.append({'a':a,'R':list(R)})
expected=[{'a':a,'R':[r*a*j+k for j in range(q//a) for k in range(a)]} for a in (1,2,3,6)]
assert checked==8*math.comb(17,5)==49504
assert records==expected,(records,expected)
assert (sum(1<<k for k in (0,1,2,8,10,11)) | (sum(1<<k for k in (0,1,2,8,10,11))<<3) | (sum(1<<k for k in (0,1,2,8,10,11))<<6)) != mask
print(json.dumps({'worker':'bitset_sumset','r':r,'q':q,'N':35,'total_profiles':checked,'accepted':records,'accepted_count':len(records),'negative_control':True,'status':'PASS','python':sys.version.split()[0]},sort_keys=True))
