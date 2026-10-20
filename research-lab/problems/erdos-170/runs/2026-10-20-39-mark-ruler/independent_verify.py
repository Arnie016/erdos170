#!/usr/bin/env python3
"""Independent 2026-10-20 checker, fixed before the second (and final) worker.
Domain: one explicit 39-mark integer set at N=500. Proof-relevant checks use
integer bitsets and one separate witness enumeration, not the C++ generator.
"""
import csv, hashlib, json, pathlib, time
root=pathlib.Path(__file__).resolve().parent
start=time.perf_counter()
raw=json.loads((root/'raw_result.json').read_text())
marks=raw['winning_marks']
assert len(marks)==39 and len(set(marks))==39
assert marks==sorted(marks) and marks[0]==0 and marks[-1]==500
P=sum(1<<x for x in marks)
covered={d for d in range(1,501) if P & (P>>d)}
assert covered==set(range(1,501))
# Independently produce an explicit 500-row witness, no coverage bit-set reuse.
witness={}
for a in marks:
    for b in marks:
        if b>a and b-a not in witness: witness[b-a]=(a,b)
assert len(witness)==500 and all(b-a==d for d,(a,b) in witness.items())
with (root/'witnesses.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(('distance','smaller_mark','larger_mark'))
    for d in range(1,501):w.writerow((d,*witness[d]))
# Positive control: prior 40-mark complete ruler.
old39=[0,1,2,3,4,5,6,13,26,39,52,65,78,91,118,145,172,199,226,253,280,307,334,361,388,402,416,430,444,458,472,486,487,488,489,490,491,492,500]
oldP=sum(1<<x for x in old39)
oldMissing=[d for d in range(1,501) if not oldP&(oldP>>d)]
assert oldMissing==[493],oldMissing
plusP=oldP|(1<<499)
assert all(plusP&(plusP>>d) for d in range(1,501))
# Negative control: delete the interior mark 262, reject 38-mark candidate.
minusP=P^(1<<262)
minusMissing=[d for d in range(1,501) if not minusP&(minusP>>d)]
assert minusMissing, 'Negative control unexpectedly passed'
proof_bytes=(root/'witnesses.csv').read_bytes()
report={'date':'2026-10-20','N':500,'marks':39,'all_500_distances':True,
  'min_distance':min(witness),'max_distance':max(witness),
  'distinct_pair_distances':len(witness),'pairs_total':39*38//2,
  'prior_39_missing':oldMissing,'prior_40_complete':True,
  'negative_delete_262_missing':minusMissing,
  'witnesses_sha256':hashlib.sha256(proof_bytes).hexdigest(),
  'elapsed_seconds':round(time.perf_counter()-start,6)}
(root/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
