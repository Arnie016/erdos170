#!/usr/bin/env python3
"""Run exactly two frozen experimental workers sequentially, each <=35 sec."""
import hashlib,json,subprocess,sys,time,os,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parent
results=[]
for name in ('worker1','worker2'):
    src=ROOT/(name+'.py');start=time.monotonic()
    cmd=[sys.executable,str(src)]
    p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    timed_out=False
    try:out,err=p.communicate(timeout=35)
    except subprocess.TimeoutExpired:
        timed_out=True
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        out,err=p.communicate()
    (ROOT/(name+'.json')).write_text(out)
    (ROOT/(name+'.stderr.txt')).write_text(err)
    results.append({'worker':name,'command':cmd[0]+' '+name+'.py','exit_code':p.returncode,'timeout':timed_out,'wall_seconds':round(time.monotonic()-start,5),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'stdout_sha256':hashlib.sha256(out.encode()).hexdigest()})
    if timed_out or p.returncode!=0:break
(ROOT/'RUN_REPORT.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
if len(results)!=2 or any(x['exit_code']!=0 or x['timeout'] for x in results):sys.exit(1)
