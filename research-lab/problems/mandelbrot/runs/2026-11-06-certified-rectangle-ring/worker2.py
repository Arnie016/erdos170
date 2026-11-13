#!/usr/bin/env python3
"""Frozen worker 2: independent integer P6(c), exact cell-center Taylor series via binomial translation,
rigorous rational radius q=1/1400 > cell half-diagonal sqrt(2)/2000.
Accept cell if |P6(c_center)|^2 > (2 + sum L1(bk) q^k)^2.
Compare accepted cells with worker1 but never use worker1 in mathematical reconstruction.
"""
from fractions import Fraction as Q
from math import comb
import json
from pathlib import Path
zero=(Q(0),Q(0));one=(Q(1),Q(0))
def plus(a,b):return (a[0]+b[0],a[1]+b[1])
def times(a,b):return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def norm2(a):return a[0]*a[0]+a[1]*a[1]
def eligible(i,j):
    def near(k):return 0 if k<=0<=k+1 else min(abs(k),abs(k+1))
    return near(i)**2+near(j)**2>=16 and max(abs(i),abs(i+1))**2+max(abs(j),abs(j+1))**2<=100
# Entire P6 as integer polynomial in c, no complex delta recursion.
poly=[0]
for _ in range(6):
    b=[0]*(2*len(poly)-1)
    for a,x in enumerate(poly):
        for d,y in enumerate(poly):b[a+d]+=x*y
    if len(b)==1:b.append(1)
    else:b[1]+=1
    poly=b
assert len(poly)==33
q=Q(1,1400)
assert q*q>Q(2,2000**2)
def cert(i,j):
    c=(Q(-3,4)+Q(2*i+1,2000),Q(1,2)+Q(2*j+1,2000))
    powers=[one]
    for _ in range(32):powers.append(times(powers[-1],c))
    shifted=[]
    for k in range(33):
        s=zero
        for z in range(k,33):
            if poly[z]:
                v=powers[z-k]
                s=plus(s,(poly[z]*comb(z,k)*v[0],poly[z]*comb(z,k)*v[1]))
        shifted.append(s)
    b0=shifted[0]
    E=Q(0);r=q
    for v in shifted[1:]:
        E+=(abs(v[0])+abs(v[1]))*r
        r*=q
    return norm2(b0)>(2+E)**2
# Baseline/negative-control: P6(0)=0; a corrupt small radius of zero near c=0 cannot escape
assert poly[0]==0
assert not (norm2(zero)>Q(4))
full=[];good=[]
for i in range(-10,10):
    for j in range(-10,10):
        if eligible(i,j):
            full.append((i,j))
            if cert(i,j):good.append((i,j))
prev=json.loads(Path(__file__).with_name('worker1.json').read_text())
assert full==[tuple(x) for x in prev['eligible']]
first={tuple(x) for x in prev['accepted']}; second=set(good);common=sorted(first & second)
assert full, 'declared eligible domain unexpectedly empty'
print(json.dumps({'claim':'MAND-RECT-2026-11-06-A','method':'integer polynomial in c, independent binomial translation per center, exact rational L1 disk bound','q':str(q),'total_eligible':len(full),'accepted':good,'accepted_count':len(good),'intersection':common,'intersection_count':len(common),'negative_control':'P6(0)=0','source':'worker2.py'},sort_keys=True))
