#!/usr/bin/env python3
"""Exact residue-chain formula and constructive difference check, frozen r=3,t=30."""
import json
from math import ceil

def construct(r,t,a):
    m=[1+(t-1-s)//a for s in range(a)]
    R=sorted(s+j*r*a for s in range(a) for j in range((m[s]+r-1)//r))
    return m,R

def main():
    rows=[]
    for a in range(1,15):
        r,t=3,30
        m,R=construct(r,t,a)
        assert max((r-1)*a,0)<t and len(R)==len(set(R))
        L=[j*a for j in range(r)]
        covered={u+b for u in L for b in R if u+b<t}
        assert covered==set(range(t))
        expected=sum((v+r-1)//r for v in m)
        assert len(R)==expected
        rows.append({'a':a,'M':expected,'residue_lengths':m,'R':R})
    d={x['a']:x['M'] for x in rows}
    assert d[3]==12 and d[5]==10 and d[7]==14
    assert [a for a in d if d[a]==10]==[1,2,5,10]
    # Controlled corruption: remove one essential initial offset from the witness a=5.
    m,R=construct(3,30,5)
    changed=[x for x in R if x!=0]
    assert set(range(30))!={u+b for u in [0,5,10] for b in changed if u+b<30}
    print(json.dumps({'status':'PASS','domain':{'r':3,'t':30,'a_min':1,'a_max':14},'rows':rows,'negative_control':'PASS'},sort_keys=True))
if __name__=='__main__':main()
