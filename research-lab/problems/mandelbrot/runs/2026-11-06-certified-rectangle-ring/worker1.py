#!/usr/bin/env python3
"""Frozen worker 1: centered Taylor coefficients and complex rectangle-Horner enclosure.
All cells: (i,j) in {-10,...,9}^2, [i,i+1]x[j,j+1] /1000,
restricted to wholly within radii 4..10 in units of 1/1000.
For each cell, require rational lower bound |z6|^2 >4.
"""
from fractions import Fraction as Q
import json
ZERO=(Q(0),Q(0)); ONE=(Q(1),Q(0)); C0=(Q(-3,4),Q(1,2))
def add(x,y): return (x[0]+y[0],x[1]+y[1])
def mul(x,y): return (x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0])
# p6(c0 + t) as exact Gaussian rational coefficient list
p=[ZERO]
for _ in range(6):
    new=[ZERO]*(2*len(p)-1)
    for i,a in enumerate(p):
        for j,b in enumerate(p):new[i+j]=add(new[i+j],mul(a,b))
    new[0]=add(new[0],C0)
    if len(new)==1: new.append(ONE)
    else: new[1]=add(new[1],ONE)
    p=new
assert len(p)==33

def interval_plus(a,b): return (a[0]+b[0],a[1]+b[1])
def interval_minus(a,b): return (a[0]-b[1],a[1]-b[0])
def interval_times(a,b):
    x=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]);return (min(x),max(x))
def cmul(a,b):
    x=interval_minus(interval_times(a[0],b[0]),interval_times(a[1],b[1]))
    y=interval_plus(interval_times(a[0],b[1]),interval_times(a[1],b[0]))
    return x,y
def ca(x,y): return interval_plus(x[0],y[0]),interval_plus(x[1],y[1])
def point(a):return ((a[0],a[0]),(a[1],a[1]))
def enclose(dx,dy):
    t=(dx,dy)
    r=point(p[-1])
    for a in reversed(p[:-1]):r=ca(cmul(r,t),point(a))
    return r

def absmin(v):
    if v[0]<=0<=v[1]:return Q(0)
    return min(abs(v[0]),abs(v[1]))
def region(i,j):
    def nearest(k): return 0 if k<=0<=k+1 else min(abs(k),abs(k+1))
    def farthest(k): return max(abs(k),abs(k+1))
    return nearest(i)**2+nearest(j)**2>=16 and farthest(i)**2+farthest(j)**2<=100

def escaped(R):return absmin(R[0])**2+absmin(R[1])**2>4
assert not escaped(enclose((Q(3,4),Q(3,4)),(Q(-1,2),Q(-1,2)))), 'zero-c parameter incorrectly escaped'
assert escaped(enclose((Q(0),Q(0)),(Q(0),Q(0)))), 'center itself must escape'
full=[];good=[]
for i in range(-10,10):
    for j in range(-10,10):
        if not region(i,j):continue
        R=enclose((Q(i,1000),Q(i+1,1000)),(Q(j,1000),Q(j+1,1000)))
        ok=escaped(R)
        full.append((i,j));
        if ok:good.append((i,j))
assert full, 'declared eligible domain unexpectedly empty'
print(json.dumps({'claim':'MAND-RECT-2026-11-06-A','method':'exact complex rectangle Horner, independent re/im interval operations','grid':'i,j=-10..9, 1/1000 boxes, min radius >=4/1000, max radius <=10/1000','total_eligible':len(full),'eligible':full,'accepted':good,'accepted_count':len(good),'negative_controls':['c=0 not certified','c0 center certified'],'source':'worker1.py'},sort_keys=True))
