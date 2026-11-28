# Quantitative follow-up, outside the scored match

Let $\delta(r)$ be the minimum number of missing positive distances after deleting any two distinct original marks of $W(r,2r)$ and inserting one fresh integer in the original interval. Endpoint deletions are included. A complete edited ruler would give $\delta(r)=0$.

The initial Metal run measured $\delta(1)=4$ and $\delta(r)=r+5$ for $2\le r\le24$. For $3\le r\le24$, the lexicographically first minimizing triple deletes the two middle junctions $L,R=L+2rq$ and inserts $L+b$. At $r=2$ its reported first minimizer has a different pair of deletions.

**Prediction frozen before the holdout.** For every $25\le r\le32$, the unrestricted minimum over these edits equals $r+5$. The same explicit junction edit should attain the minimum. Verify all edit triples on Metal and reproduce the minima, coordinates, and lost-length checksums with the separate CPU implementation.

**Universal questions.** Does the junction edit always miss exactly $r+5$ lengths for $r\ge2$? Does every allowed edit miss at least that many? A proof for the explicit edit settles the first question. A single better edit refutes the second.
