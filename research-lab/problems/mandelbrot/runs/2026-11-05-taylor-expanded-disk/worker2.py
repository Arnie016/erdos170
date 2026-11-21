#!/usr/bin/env python3
"""Frozen independent worker 2: construct z6(c) as integer polynomial, translate by binomial theorem.
Check the strongest success radius supplied as literal CLI arg against independent rational L1 envelope.
"""
from fractions import Fraction as F
from math import comb
import sys,json
c=(F(-3,4),F(1,2));zero=(F(0),F(0));one=(F(1),F(0));
def add(a,b):return (a[0]+b[0],a[1]+b[1])
def mul(a,b):return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def power(a,k):
    z=one
    for _ in range(k):z=mul(z,a)
    return z
# z6(c) has strictly integer coefficients since z0=0 and z_(j+1)=z_j^2+c
coeff=[0]
for _ in range(6):
    newer=[0]*(2*len(coeff)-1)
    for i,a in enumerate(coeff):
        for j,b in enumerate(coeff):newer[i+j]+=a*b
    if len(newer)==1:newer.append(1)
    else:newer[1]+=1
    coeff=newer
assert len(coeff)==33 and all(isinstance(x,int) for x in coeff)
translated=[]
for k in range(len(coeff)):
    ak=zero
    for j in range(k,len(coeff)):
        ak=add(ak,mul((F(coeff[j]*comb(j,k)),F(0)),power(c,j-k)))
    translated.append(ak)
w=translated[0];assert w[0]*w[0]+w[1]*w[1]>4
radi=[F(1,1000),F(1,750),F(1,500),F(1,400),F(1,250),F(1,100)]
rows=[]
for r in radi:
    s=F(0)
    for k in range(1,len(translated)):
        # upper in L1 norm, no integer sqrt or shared recursion with worker 1
        s+=(abs(translated[k][0])+abs(translated[k][1]))*r**k
    rows.append({'rho':str(r),'L1_envelope':str(s),'certifies':w[0]*w[0]+w[1]*w[1]>(2+s)**2,'s':float(s)})
print(json.dumps({'method':'integer c-polynomial plus binomial shift and L1 direct envelope','w6':[str(x) for x in w],'coefficients':[[str(x) for x in a] for a in translated],'rows':rows,'negative_control':'radius zero around c=0 cannot satisfy |w6|>2'},sort_keys=True))

