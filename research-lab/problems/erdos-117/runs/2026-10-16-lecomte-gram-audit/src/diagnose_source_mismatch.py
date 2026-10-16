import json, sys
from pathlib import Path
base=Path(__file__).resolve().parents[1]
d=json.loads((base/'data/lecomte_13_vectors.json').read_text())
pts=d['vectors'];tab=d['printed_gram_upper_rows']
errors=[];zeros=[];matches=0
for i,ai in enumerate(pts):
    for j in range(i+1,13):
        aj=pts[j]
        blocks=tuple((ai[2*k]*aj[2*k+1]-ai[2*k+1]*aj[2*k])%3 for k in range(3))
        value=sum(blocks)%3;source=int(tab[i][j-i-1])
        if value!=source:errors.append({'i':i+1,'j':j+1,'computed':value,'paper':source,'block_terms':blocks,'vectors':[ai,aj]})
        if value==0:zeros.append({'i':i+1,'j':j+1,'block_terms':blocks,'vectors':[ai,aj]})
        matches+=int(value==source)
report={'status':'DIAGNOSTIC_FAIL' if errors or zeros else 'DIAGNOSTIC_PASS','pair_count':78,'matching_printed_entries':matches,'mismatch_count':len(errors),'zero_pairings_count':len(zeros),'zero_pairings':zeros,'all_mismatches':errors,'python':sys.version.split()[0]}
(base/'data/DIAGNOSTIC.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('all_mismatches','zero_pairings')},indent=2))
print('first_five_mismatches='+json.dumps(errors[:5]))
print('first_five_zero_pairs='+json.dumps(zeros[:5]))
