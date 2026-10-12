#!/usr/bin/env python3
"""Independent exact verifier: bits-as-vectors and symmetric coordinate matrices.
Does not call or import the C++ clique/coloring search.
"""
import json
from itertools import combinations, permutations
from pathlib import Path

EDGE_UNIVERSE=list(combinations(range(5),2))
REPS={
 'triangle':((0,1),(0,2),(1,2)),
 'star':((0,1),(0,2),(0,3)),
 'path4':((0,1),(1,2),(2,3)),
 'path3_plus_edge':((0,1),(1,2),(3,4)),
}

def matrix_form(edges, x, y):
    # independent representation: each form is the 5x5 symmetric matrix with
    # two unit off-diagonal entries; multiply x^T M y in F_2.
    values=[]
    for (i,j) in edges:
        M=[0]*5
        M[i]=1<<j
        M[j]=1<<i
        acc=0
        for row in range(5):
            acc^=((x>>row)&1)&((M[row]&y).bit_count()&1)
        values.append(acc)
    return tuple(values)

def check_one(rec):
    edges=tuple(tuple(q) for q in rec['edges'])
    assert edges==REPS[rec['case']]
    k=rec['omega']; C=rec['clique']; colors=rec['colors']
    assert isinstance(k,int) and 1<=k<=31
    assert len(C)==k and len(set(C))==k and all(1<=x<=31 for x in C)
    assert len(colors)==31 and set(colors)==set(range(k))
    for x,y in combinations(C,2): assert any(matrix_form(edges,x,y)), ('not clique',rec['case'],x,y)
    for x,y in combinations(range(1,32),2):
        if colors[x-1]==colors[y-1]: assert not any(matrix_form(edges,x,y)),('bad color',rec['case'],x,y)
    # Every color class spans an isotropic F2-subspace; test ALL its span vectors.
    for color in range(k):
        gens=[v for v in range(1,32) if colors[v-1]==color]
        span={0}
        for v in gens: span|={u^v for u in tuple(span)}
        for x,y in combinations(sorted(span),2):
            assert not any(matrix_form(edges,x,y)), ('span bad',rec['case'],color,x,y)
    return k

def relabel(edge_set,p):
    return tuple(sorted(tuple(sorted((p[i],p[j]))) for i,j in edge_set))

def main():
    path=Path('raw_witnesses.jsonl')
    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(records)==4 and {r['case'] for r in records}==set(REPS)
    sizes={r['case']:check_one(r) for r in records}
    # All C(10,3)=120 labelled graphs have exactly one representative orbit.
    orbits={name:{relabel(edges,p) for p in permutations(range(5))} for name,edges in REPS.items()}
    all_graphs={tuple(sorted(S)) for S in combinations(EDGE_UNIVERSE,3)}
    assert len(all_graphs)==120 and all(len(O)>0 for O in orbits.values())
    for x,(name,O) in enumerate(orbits.items()):
        assert O<=all_graphs
        for name2,O2 in list(orbits.items())[x+1:]: assert O.isdisjoint(O2)
    assert set().union(*orbits.values())==all_graphs
    # Negative controls: malformed witnesses must be rejected.
    bad=json.loads(json.dumps(records[0]))
    bad['clique'][0]=bad['clique'][1]
    assert_rejected=False
    try: check_one(bad)
    except AssertionError: assert_rejected=True
    assert assert_rejected
    bad2=json.loads(json.dumps(records[0]))
    bad2['colors']=[0]*31
    assert_rejected=False
    try: check_one(bad2)
    except AssertionError: assert_rejected=True
    assert assert_rejected
    print(json.dumps({'status':'INDEPENDENT_CHECK_PASS','representatives':sizes,'orbits':{k:len(v) for k,v in orbits.items()},'labelled_total':len(all_graphs),'negative_controls_rejected':2},sort_keys=True))

if __name__=='__main__':main()
