#!/usr/bin/env python3
"""Exact Fraction-pair recurrence for exactly the 42 zero-bit child centers."""
import json
from fractions import Fraction as F
PARENTS=[(3,-9,'CEFF'),(4,-8,'EFFF'),(5,-8,'008C'),(5,-7,'EFFF'),(6,-7,'008C'),(6,-6,'EFFF'),(7,-6,'08CE'),(7,-5,'FFFF'),(8,-4,'FFFF')]
def domain():
  for i,j,h in PARENTS:
    for a in range(4):
      for b in range(4):
        if not (int(h,16)>>(4*a+b))&1: yield i,j,a,b

def mul(z,w):return (z[0]*w[0]-z[1]*w[1],z[0]*w[1]+z[1]*w[0])
def norm(z):return z[0]*z[0]+z[1]*z[1]
def z6(c):
  z=(F(0),F(0))
  for _ in range(6):
    q=mul(z,z);z=q[0]+c[0],q[1]+c[1]
  return z
assert norm(z6((F(0),F(0))))==0
assert norm(z6((F(-3,4),F(1,2))))>4
cells=list(domain());assert len(cells)==42 and len(set(cells))==42
out=[]
for i,j,a,b in cells:
  re=F(-3,4)+F(8*i+2*a+1,8000)
  im=F(1,2)+F(8*j+2*b+1,8000)
  n=norm(z6((re,im)))
  out.append({'cell':[i,j,a,b],'norm2_num':str(n.numerator),'norm2_den':str(n.denominator),'escapes_by_6':n>4,'equal_4':n==4})
print(json.dumps({'claim_id':'MAND-CENTER-2026-11-08-A','method':'Fraction complex-pair recurrence','n':len(out),'escaped_centers':sum(x['escapes_by_6'] for x in out),'non_escaped_centers':sum(not x['escapes_by_6'] for x in out),'equal_centers':sum(x['equal_4'] for x in out),'records':out},sort_keys=True))
