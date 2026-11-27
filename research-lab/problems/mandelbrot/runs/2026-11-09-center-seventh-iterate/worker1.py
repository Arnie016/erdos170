#!/usr/bin/env python3
"""Exact Gaussian-rational recurrence for the 22 sixth-iterate nonescaping centers.
Frozen domain reconstructed from 7 Nov parent masks at PR #8 dd482e9.
"""
import json
from fractions import Fraction as Q
from pathlib import Path
PARENTS=[(3,-9,'CEFF'),(4,-8,'EFFF'),(5,-8,'008C'),(5,-7,'EFFF'),(6,-7,'008C'),(6,-6,'EFFF'),(7,-6,'08CE'),(7,-5,'FFFF'),(8,-4,'FFFF')]
def z_iter(c,n):
    x=y=Q(0)
    for _ in range(n):
        x,y=x*x-y*y+c[0],2*x*y+c[1]
    return x*x+y*y
def domain():
    for i,j,h in PARENTS:
        for a in range(4):
            for b in range(4):
                if not (int(h,16) & (1 << (4*a+b))):
                    p,q=-6000+8*i+2*a+1,4000+8*j+2*b+1
                    c=Q(p,8000),Q(q,8000)
                    if z_iter(c,6)<4:
                        yield (i,j,a,b,p,q)
assert z_iter((Q(0),Q(0)),7)==0
assert z_iter((Q(-3,4),Q(1,2)),6)>4
points=list(domain())
assert len(points)==22 and len(set(x[:4] for x in points))==22
records=[]
for i,j,a,b,p,q in points:
    c=Q(p,8000),Q(q,8000)
    n6=z_iter(c,6); n7=z_iter(c,7)
    assert n6<4 and n7!=4
    records.append({'cell':[i,j,a,b],'p':p,'q':q,
                    'z6_less_4':True,'z7_num':str(n7.numerator),
                    'z7_den':str(n7.denominator),'z7_greater_4':n7>4})
result={'claim_id':'MAND-CENTER7-2026-11-09-A','method':'Gaussian rational coordinates',
        'domain':'22 sixth-iterate nonescaping child centers from 7 Nov mask',
        'n':len(records),'z7_escape_count':sum(r['z7_greater_4'] for r in records),
        'z7_nonescape_count':sum(not r['z7_greater_4'] for r in records),'records':records}
print(json.dumps(result,sort_keys=True))
