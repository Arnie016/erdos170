#!/usr/bin/env python3
"""Independently build integer P6(c) and certify a full 1/4000 child square
by rational binomial Taylor translation at its center and a conservative
rational Euclidean-radius q=1/5600. Same 9*4*4 frozen domain as worker1.
"""
import json
from math import comb
from fractions import Fraction as Q
from pathlib import Path
HOLES=[(3,-9),(4,-8),(5,-8),(5,-7),(6,-7),(6,-6),(7,-6),(7,-5),(8,-4)]
poly=[0]
for _ in range(6):
    n=[0]*(2*len(poly)-1)
    for i,a in enumerate(poly):
        for j,b in enumerate(poly):n[i+j]+=a*b
    if len(n)==1:n.append(1)
    else:n[1]+=1
    poly=n
assert len(poly)==33 and poly[0]==0
zero=(Q(0),Q(0));one=(Q(1),Q(0))
def plus(a,b):return (a[0]+b[0],a[1]+b[1])
def times(a,b):return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def norm2(a):return a[0]*a[0]+a[1]*a[1]
q=Q(1,5600)
assert q*q>Q(2,8000**2) # half diagonal of a 1/4000 square

def certified(i,j,a,b):
    center=(Q(-3,4)+Q(8*i+2*a+1,8000),Q(1,2)+Q(8*j+2*b+1,8000))
    powers=[one]
    for _ in range(32):powers.append(times(powers[-1],center))
    bs=[]
    for k in range(33):
        s=zero
        for z in range(k,33):
            if poly[z]:
                v=powers[z-k];w=poly[z]*comb(z,k)
                s=plus(s,(w*v[0],w*v[1]))
        bs.append(s)
    E=Q(0);radius=q
    for v in bs[1:]:
        E+=(abs(v[0])+abs(v[1]))*radius
        radius*=q
    return norm2(bs[0])>(Q(2)+E)**2

def admissible(i,j):
    def near(k):return 0 if k<=0<=k+1 else min(abs(k),abs(k+1))
    return near(i)**2+near(j)**2>=16 and max(abs(i),abs(i+1))**2+max(abs(j),abs(j+1))**2<=100
assert len(HOLES)==len(set(HOLES))==9 and all(admissible(i,j) for i,j in HOLES)
assert poly[0]==0 and norm2(zero)!=4 # c=0 negative control
ok=[]
for i,j in HOLES:
  for a in range(4):
    for b in range(4):
      if certified(i,j,a,b):ok.append([i,j,a,b])
prior=json.loads(Path(__file__).with_name('worker1.json').read_text())
intersection=sorted(set(map(tuple,prior['accepted'])) & set(map(tuple,ok)))
print(json.dumps({'id':'MAND-SUBGRID-2026-11-07-A','method':'integer z6(c), exact binomial translation, rational ball majorant','domain':'9 frozen parents x 4x4 closed subdivision; denominator 4000','q':str(q),'eligible':144,'accepted':ok,'accepted_count':len(ok),'intersection_count':len(intersection),'intersection':intersection,'unaccepted_count':144-len(ok),'controls':['z6(0)=0']},sort_keys=True))
