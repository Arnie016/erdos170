#!/usr/bin/env python3
"""Independent homogeneous Gaussian-integer iteration over denominator powers."""
import json
from fractions import Fraction
from pathlib import Path
PARENTS=[(3,-9,0xCEFF),(4,-8,0xEFFF),(5,-8,0x008C),(5,-7,0xEFFF),(6,-7,0x008C),(6,-6,0xEFFF),(7,-6,0x08CE),(7,-5,0xFFFF),(8,-4,0xFFFF)]
def points():
  return [(i,j,a,b) for i,j,h in PARENTS for a in range(4) for b in range(4) if not h&(1<<(4*a+b))]
def power_iter(p,q):
  # c=(p+i*q)/8000; z=A+iB over D; start z=0.
  A=B=0;D=1
  for _ in range(6):
    d2=D*D
    A,B,D=8000*(A*A-B*B)+p*d2,16000*A*B+q*d2,8000*d2
  return Fraction(A*A+B*B,D*D)
assert power_iter(0,0)==0
assert power_iter(-6000,4000)>4
keys=points();assert len(keys)==42 and len(set(keys))==42
r1=json.loads(Path(__file__).with_name('worker1.json').read_text())
assert [tuple(x['cell']) for x in r1['records']]==keys
records=[];bad=[]
for index,(i,j,a,b) in enumerate(keys):
  p=-6000+8*i+2*a+1;q=4000+8*j+2*b+1
  exact=power_iter(p,q)
  expected=Fraction(int(r1['records'][index]['norm2_num']),int(r1['records'][index]['norm2_den']))
  if exact!=expected:bad.append(keys[index])
  records.append({'cell':[i,j,a,b],'norm2_num':str(exact.numerator),'norm2_den':str(exact.denominator),'escapes_by_6':exact>4,'equal_4':exact==4})
assert not bad, f'Exact norm mismatch: {bad}'
assert records==r1['records']
print(json.dumps({'claim_id':'MAND-CENTER-2026-11-08-A','method':'independent homogeneous Gaussian integer recurrence','n':len(records),'agrees_with_worker1':True,'escaped_centers':sum(x['escapes_by_6'] for x in records),'non_escaped_centers':sum(not x['escapes_by_6'] for x in records),'equal_centers':sum(x['equal_4'] for x in records),'non_escaped_cells':[x['cell'] for x in records if not x['escapes_by_6']],'records':records},sort_keys=True))
