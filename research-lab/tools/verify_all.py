#!/usr/bin/env python3
"""Reproduce all eight imported studies in a fresh directory.

This is publication/CI validation, not a scheduled research search. Each of the
13 study programs and two compilations runs sequentially with a hard 35-second
walltime limit. Exact acceptance checks, logs and data are preserved on failure.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 35
H4PAIR = {'1,1':1,'3,3':105,'5,5':1050,'6,6':1260,'9,9':1680}
H4ALL = {'1,1':1,'3,3':35,'5,5':189,'6,6':210,'7,7':15,'9,9':1485,'11,11':490,'13,13':315,'15,15':85}
H5PAIR = {'5':2821,'6':6510,'9':112840,'11':52080,'end':0}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def main() -> int:
    generated = ROOT / '_generated'
    generated.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='verification-', dir=generated))
    logs = work / 'logs'
    logs.mkdir()
    receipts: list[dict] = []
    accepted: list[dict] = []
    source_hashes = {}
    for source in sorted((ROOT/'studies').glob('*/src/*')):
        if source.suffix not in {'.py','.cpp'}:
            continue
        relative = source.relative_to(ROOT/'studies')
        target = work/'studies'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source,target)
        source_hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    for src in (work/'studies').glob('*/src'):
        (src.parent/'results').mkdir()
        (src.parent/'.build').mkdir()

    report = {'status':'RUNNING','python':sys.version,'scope':'fresh publication/CI reproduction of eight recovered studies','source_sha256':source_hashes,'receipts':receipts,'accepted_studies':accepted}
    def save() -> None:
        (work/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(study: str, command: list[str]) -> str:
        label = f'{len(receipts)+1:02d}-{study}'
        cwd = work/'studies'/study
        start = time.monotonic()
        process = subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        timeout = False
        try:
            stdout,stderr = process.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            timeout = True
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout,stderr = process.communicate()
        (logs/f'{label}.stdout.txt').write_text(stdout)
        (logs/f'{label}.stderr.txt').write_text(stderr)
        receipts.append({'study':study,'command':command,'timeout_seconds':TIMEOUT,'timed_out':timeout,'returncode':process.returncode,'wall_seconds':time.monotonic()-start,'stdout':f'logs/{label}.stdout.txt','stderr':f'logs/{label}.stderr.txt'})
        save()
        require(not timeout and process.returncode==0,f'{label} failed; inspect saved logs')
        return stdout
    def read(study: str, name: str) -> dict:
        return json.loads((work/'studies'/study/name).read_text())
    def passed(study: str, detail: dict) -> None:
        accepted.append({'study':study,**detail})
        save()
        print(f'PASS {study}',flush=True)

    try:
        s='erdos117-2026-10-03'
        run(s,[sys.executable,'src/verify_coupling_rank_budget.py'])
        x=read(s,'coupling_rank_budget_regression.json')
        require(x['status']=='PASS' and x['total_cases']==16896,'coupling domain/result mismatch')
        passed(s,{'matrix_cases':16896})

        s='erdos117-2026-10-04'
        run(s,[sys.executable,'src/check_coordinate_shear.py'])
        x=read(s,'coordinate_shear_regression.json')
        require(x['status']=='PASS' and x['cases']==36 and all(r['pass'] for r in x['results']),'shear regression mismatch')
        passed(s,{'matrix_cases':36})

        s='erdos117-2026-10-05'
        run(s,[sys.executable,'src/verify_pencil_crossrank_envelope.py'])
        x=read(s,'pencil_crossrank_regression.json')
        require(x['status']=='PASS' and x['parameter_cases']==136 and x['pair_checks']==2856 and not x['failures'],'cross-rank regression mismatch')
        passed(s,{'parameter_cases':136,'scalar_pairs':2856})

        s='erdos117-2026-10-06'
        run(s,[sys.executable,'src/worker1_graph_exact.py'])
        run(s,[sys.executable,'src/worker2_certificate_check.py'])
        x=read(s,'results/worker1_graph_exact.json'); y=read(s,'results/worker2_certificate.json')
        require(x['status']==y['status']=='PASS','order-64 checker status mismatch')
        require(len(x['cases'])==len(y['cases'])==2,'missing order-64 case')
        for r,z,expected,center in zip(x['cases'],y['cases'],(9,5),(4,8)):
            require(r['omega']==r['chromatic_number']==expected and r['center_size']==center and r['associativity_triples_checked']==262144,'order-64 graph invariant mismatch')
            require(z['omega_lower_witness_size']==z['abelian_cover_upper_witness_size']==expected and z['full_group_covered'] and z['subgroups_closed'] and z['subgroups_abelian'] and z['negative_controls_passed']==2,'order-64 witness mismatch')
        passed(s,{'matching_cover_clique_sizes':[9,5]})

        s='erdos117-2026-10-07'
        run(s,[sys.executable,'src/worker1_census.py'])
        run(s,[sys.executable,'src/worker2_independent.py'])
        x=read(s,'results/worker1_census.json'); y=read(s,'results/worker2_independent.json')
        require(x['status']==y['status']=='PASS' and x['maps_checked']==y['maps_checked']==4096,'ordered-map domain/status mismatch')
        require(x['pair_counts']==y['pair_counts']==H4PAIR and y['subspaces_enumerated']==67 and not y['failures'],'ordered-map histogram mismatch')
        passed(s,{'complete_ordered_maps':4096,'histogram':H4PAIR})

        s='erdos117-2026-10-08'
        run(s,[sys.executable,'src/worker1_rref_graph.py'])
        run(s,[sys.executable,'src/worker2_bfs_subspace_cover.py'])
        x=read(s,'results/worker1_summary.json'); y=read(s,'results/worker2_summary.json')
        require(x['status']==y['status']=='PASS' and x['number_of_kernels']==y['kernels_checked']==2825,'kernel domain/status mismatch')
        require(x['histogram']==y['histogram']==H4ALL and y['disagreements_with_worker1']==0,'kernel histogram/comparison mismatch')
        passed(s,{'complete_kernels':2825,'histogram':H4ALL})

        s='erdos117-2026-10-09'
        require(shutil.which('g++') is not None,'g++ is required; no automatic package installation')
        run(s,['g++','-std=c++17','-O3','src/rank5_two_pencils.cpp','-o','.build/first'])
        run(s,['g++','-std=c++17','-O3','src/worker2_iso_cover.cpp','-o','.build/second'])
        x=json.loads(run(s,['.build/first']))
        y=json.loads(run(s,['.build/second']))
        require(x['status']=='FULL_NO_GAP' and y['status']=='INDEPENDENT_PASS','pencil success status mismatch')
        require(x['checked']==y['checked']==174251 and x['gaps']==y['gap_count']==0,'pencil domain/gap mismatch')
        require(x['histogram']==y['histogram']==H5PAIR and y['subspaces']==374,'pencil histogram mismatch')
        passed(s,{'complete_pencils':174251,'histogram':H5PAIR})

        s='modular69'
        run(s,[sys.executable,'src/generate.py','results/candidates.json'])
        run(s,[sys.executable,'src/verify.py','results/candidates.json','results/verification.json'])
        x=read(s,'results/verification.json')
        require(x['status']=='EXACT_FINITE_CERTIFICATE' and x['n']==6 and x['N']==69,'modular scope/status mismatch')
        require(x['canonical_count']==66 and x['full_count']==4224 and x['minimum_overlap']==5 and x['explicit_signed_dyadic_orbits']==1408,'modular count mismatch')
        require(x['overlap_histogram']=={'0':0,'1':0,'2':0,'3':0,'4':0,'5':44,'6':22} and all(x['negative_controls'].values()),'modular overlap/negative-control mismatch')
        passed(s,{'canonical_sets':66,'admissible_sets':4224,'minimum_overlap':5})

        require(len(accepted)==8 and len(receipts)==15,'incomplete publication replay')
        report['status']='PASS'
        save()
        print(f'PASS: all eight studies; logs and raw outputs: {work}')
        return 0
    except Exception as error:
        report['status']='FAIL'
        report['failure']=str(error)
        save()
        print(f'FAIL: {error}. Evidence retained at {work}',file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
