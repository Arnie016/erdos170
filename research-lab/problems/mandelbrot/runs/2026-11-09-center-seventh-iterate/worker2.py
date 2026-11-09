#!/usr/bin/env python3
"""Independent homogeneous Gaussian integer recurrence, checked against worker1."""
import json,math
from pathlib import Path
PARENTS=[(3,-9,0xCEFF),(4,-8,0xEFFF),(5,-8,0x008C),(5,-7,0xEFFF),(6,-7,0x008C),(6,-6,0xEFFF),(7,-6,0x08CE),(7,-5,0xFFFF),(8,-4,0xFFFF)]
def norm_integer(p,q,n):
    A=B=0;D=1
    for _ in range(n):
        d2=D*D
        A,B,D=8000*(A*A-B*B)+p*d2,16000*A*B+q*d2,8000*d2
        g=math.gcd(math.gcd(A,B),D)
        if g>1:A//=g;B//=g;D//=g
    num=A*A+B*B;den=D*D
    g=math.gcd(num,den)
    return num//g,den//g
assert norm_integer(0,0,7)==(0,1)
assert norm_integer(-6000,4000,6)[0]>4*norm_integer(-6000,4000,6)[1]
base=json.loads(Path(__file__).with_name('worker1.json').read_text())
assert base['n']==22
fresh=[]
for i,j,h in PARENTS:
    for a in range(4):
        for b in range(4):
            if not (h>>(4*a+b))&1:
                p=-6000+8*i+2*a+1;q=4000+8*j+2*b+1
                n6,d6=norm_integer(p,q,6)
                if n6<4*d6:
                    fresh.append((i,j,a,b,p,q))
assert len(fresh)==22
assert [x[:4] for x in fresh]==[tuple(r['cell']) for r in base['records']]
records=[]
for (i,j,a,b,p,q),r in zip(fresh,base['records']):
    num,den=norm_integer(p,q,7)
    assert (num,den)==(int(r['z7_num']),int(r['z7_den']))
    assert (num>4*den)==r['z7_greater_4']
    records.append({'cell':[i,j,a,b],'z7_greater_4':num>4*den})
result={'claim_id':'MAND-CENTER7-2026-11-09-A','method':'Gaussian integer recurrence with gcd normalization',
        'n':22,'agrees_exactly':True,'z7_escape_count':sum(r['z7_greater_4'] for r in records),
        'z7_nonescape_count':sum(not r['z7_greater_4'] for r in records),'records':records}
print(json.dumps(result,sort_keys=True))
