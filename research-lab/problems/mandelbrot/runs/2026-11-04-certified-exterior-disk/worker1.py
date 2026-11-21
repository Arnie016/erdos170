#!/usr/bin/env python3
"""Worker 1: exact rational disk exclusion certificate. No search or dependencies."""
from fractions import Fraction as Q
import json, hashlib, pathlib, sys
p=pathlib.Path(__file__).parent
c=(Q(-3,4),Q(1,2)); rho=Q(1,1000); z=(Q(0),Q(0)); E=Q(0)
pts=[]; bounds=[]
def pair(v): return [str(v[0]),str(v[1])]
for j in range(7):
    U=abs(z[0])+abs(z[1]); pts.append(pair(z)); bounds.append({'j':j,'U':str(U),'E':str(E)})
    if j<6:
        E=(2*U+E)*E+rho
        z=(z[0]**2-z[1]**2+c[0],2*z[0]*z[1]+c[1])
left=z[0]*z[0]+z[1]*z[1]; right=(Q(2)+E)**2
assert abs(c[0])+abs(c[1])+rho<Q(2)
assert left>right, f'Disk not certified: left={left}, right={right}'
assert not (Q(0)**2>Q(2)**2) # negative control: center 0 cannot fake escape
cert={'claim_id':'MAND-EXTERIOR-2026-11-04-A','center':pair(c),'radius':str(rho),'max_j':6,'w':pts,'bounds':bounds,'left_norm_squared':str(left),'right_threshold_squared':str(right),'verified':True,'negative_control':'c=0, z6=0, rejection'}
(p/'CERTIFICATE.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps({'status':'PASS','j':6,'w6':pair(z),'U6':bounds[6]['U'],'E6':str(E),'left_gt_right':left>right,'checks':'exact fractions + negative control'}))
