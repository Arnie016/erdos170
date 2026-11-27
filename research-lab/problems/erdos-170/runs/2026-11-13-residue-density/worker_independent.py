#!/usr/bin/env python3
"""Independent covering recurrence and small-domain exhaustive subset check."""
import json
from itertools import combinations

def min_cover_residue(m,r):
    # Dynamic programming over a line: available translated windows are [j,j+r-1].
    # Find exact min number needed to cover the whole line without using the formula.
    f=[0]+[10**9]*m
    for i in range(1,m+1):
        for j in range(i):
            if j+r>=i:
                f[i]=min(f[i],f[j]+1)
    return f[m]

def exhaustive_small(r,t,a):
    L={k*a for k in range(r)}
    target=(1<<t)-1
    # Iterate every R subset of the whole reflected t-window, not just q-subsets.
    best=t+1; witnesses=0
    for mask in range(1<<t):
        k=mask.bit_count()
        if k>best or not (mask&1):continue  # offset 0 forced by longest distance
        seen=0
        for b in range(t):
            if mask>>b & 1:
                for u in L:
                    if u+b<t:seen|=1<<(u+b)
        if seen==target:
            if k<best:best=k;witnesses=0
            if k==best:witnesses+=1
    return best,witnesses

def main():
    rows=[]
    for a in range(1,15):
        t,r=30,3
        lens=[]
        for s in range(a):
            lens.append(len(range(s,t,a)))
        rows.append({'a':a,'M':sum(min_cover_residue(m,r) for m in lens)})
    assert [z['a'] for z in rows if z['M']==10]==[1,2,5,10]
    cross=[]
    for a in range(1,6):
        m,w=exhaustive_small(3,12,a)
        independent=sum(min_cover_residue(len(range(s,12,a)),3) for s in range(a))
        assert m==independent and w>0
        cross.append({'a':a,'minimum':m,'number_of_minimal_R':w})
    # Corrected negative control: a=5 requires 5 marks, whereas a=4 needs 4.
    assert next(z['minimum'] for z in cross if z['a']==5)>4
    print(json.dumps({'status':'PASS','domain':{'r':3,'t':30,'a_min':1,'a_max':14},'rows':rows,'independent_exhaustive_domain':{'r':3,'t':12,'a_min':1,'a_max':5,'subsets_per_a':4096},'small':cross,'negative_control':'PASS'},sort_keys=True))
if __name__=='__main__':main()
