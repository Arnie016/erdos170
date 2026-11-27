#!/usr/bin/env python3
"""Reproduce exactly two independent checks, sequentially, with 35s hard limits."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent
logs=[]
for name in ('worker_pairs.py','worker_masks.py'):
    cmd=[sys.executable, str(D/name)]
    t0=time.monotonic()
    p=subprocess.Popen(cmd,cwd=D,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    timed_out=False
    try: out,err=p.communicate(timeout=35)
    except subprocess.TimeoutExpired:
        timed_out=True
        try: os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        out,err=p.communicate()
    elapsed=time.monotonic()-t0
    (D/(name+'.stdout')).write_text(out)
    (D/(name+'.stderr')).write_text(err)
    row={'command':cmd,'timeout_sec':35,'exit_code':p.returncode,'timed_out':timed_out,'wall_seconds':round(elapsed,6),'sha256':hashlib.sha256((D/name).read_bytes()).hexdigest()}
    logs.append(row)
    print(name,row['exit_code'],row['wall_seconds'])
    if p.returncode!=0 or timed_out: break
status='PASS' if len(logs)==2 and all(v['exit_code']==0 and not v['timed_out'] for v in logs) else 'FAIL'
if status=='PASS':
    x=json.loads((D/'worker_pairs.json').read_text());y=json.loads((D/'worker_masks.json').read_text())
    assert x['valid']==y['valid'] and x['checked']==y['checked']==5082
report={'status':status,'environment':{'python':sys.version.split()[0],'platform':sys.platform},'receipts':logs,'frozen_acceptance':'5082 full tail profiles, exactly four permitted, a in (1,2,3,6), matching exact right tails, negative controls pass'}
(D/'RUN_RECEIPTS.json').write_text(json.dumps(report,indent=2)+'\n')
if status!='PASS': raise SystemExit(1)
