# Strict four-cycle penalty for incident added edges

**E60-INCIDENT-2ADD-20261028**: ANALYTICAL_PROOF / PRIORITY_UNCHECKED / UNFORMALIZED.

## Precise claim
Let q>=3 be an odd prime power, H the orthogonal polarity graph of PG(2,q), and f an original H-edge. For distinct nonedges e1,e2 of H sharing an endpoint, G=(H\{f})+e1+e2 has **at least 2q-3 unoriented, noninduced C4**. This is attained for every such q.

Moreover, for **any** two distinct added nonedges, exactly 2q-4 cycles remain if and only if both new edges join disjoint pairs of absolute (degree-q) vertices and f lies on an H-path of length three between each added pair. This concerns only a two-addition/one-deletion neighborhood of H, not all graphs with ex(n,C4)+1 edges.

## Dependencies, with exact source
He, Ma and Yang, *Some exact results on 4-cycles: stability and supersaturation*, arXiv:1912.00986v4, Lemmas 2.7-2.8: distinct absolute vertices are nonadjacent; adding a nonedge xy creates q-1, q or q+1 new C4, with q-1 if and only if both endpoints have degree q; for each fixed xy the new cycles share only the added edge. Thus a deleted original H-edge f destroys at most one of the cycles for each added edge. H is C4-free. Orthogonal polarities in odd characteristic have a nondegenerate conic of q+1 absolute vertices, with no three collinear.

## Proof
Let c_i count the cycles of H+e_i containing e_i. The cycles of H+e1+e2 partition into those using e1 alone, e2 alone, or both. The first two families have sizes c1,c2. Removing f deletes at most one per single-edge family, so at least c1+c2-2 cycles survive. If either e_i has a nonabsolute endpoint, then c1+c2>=2q-1, proving >=2q-3.

Now suppose e1=AB and e2=AC with distinct absolute A,B,C. We have c1=c2=q-1. The unique common neighbor w of B,C gives precisely one additional cycle containing both new edges, A-B-w-C-A. Hence H+e1+e2 has 2q-1 cycles.

To destroy three cycles by removing a single H-edge f, that edge would have to appear in both single-edge cycle families and also in the mixed cycle. The only old edges in the mixed cycle are Bw,Cw. But Bw cannot occur in any H-length-three path from A to C: if B is first internal vertex, then AB would be an H-edge; if second, BC would be an H-edge. Neither is possible between distinct absolute vertices. Likewise Cw belongs to no A-to-B H-length-three path. Therefore f kills at most two of these cycles, and >=2q-3 survive.

To attain 2q-3, choose any distinct absolute A,B,C and w the common H-neighbor of B,C. The polar line of w contains B,C but not A, since a conic has no three collinear points. Thus A and w are H-nonadjacent, with a unique common neighbor u different from A,w,B,C. Consequently A-u-w-B is a simple length-three H-path. Delete f=Bw, then add AB and AC. Removing Bw kills exactly one AB-only cycle plus the mixed cycle but no AC-only cycle. The resulting count is (q-2)+(q-1)=2q-3.

Finally, if any two-addition/one-deletion configuration has exactly 2q-4 cycles, the first argument forces both added edges to join absolute points, while the strict incident bound forces them to be disjoint. With four distinct absolute endpoints, no C4 can use both new edges because that would require an H-edge between distinct absolute endpoints. The count is therefore 2(q-1) minus at most one cycle per new edge removed by f. Equality at 2q-4 is equivalent to f lying in one H-length-three path for each pair. QED.

## Scope
For odd prime powers q>=17, Füredi's theorem identifies e(H)=q(q+1)^2/2=ex(q^2+q+1,C4). The theorem constrains one particular edit neighborhood and proves no unrestricted lower bound for Erdős #60. The general local 2q-4 lower bound is already present in He–Ma–Yang; literature novelty of this incidence/equality characterization is **unchecked**. This is an ordinary proof, not a Lean proof.
