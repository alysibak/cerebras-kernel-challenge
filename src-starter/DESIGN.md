# Top-K k-NN on WSE-2 — Design Memo

**Start time:** 2026-04-26 (logged per SPEC §7)

## 1. Sharding & routing topology

Database `D` (`N × d`, fp32) is row-sharded across a `P × P` PE grid: PE `(px, py)` owns rows
`[(py·P + px) · S, (py·P + px + 1) · S)` where `S = ceil(N / P²) = rows_per_pe`. The last PE
zero-pads its shard so every PE executes the same loop length; padded rows get distance `+inf`
and index `N` so they can never enter top-K (this also keeps tie-breaking stable for `uneven`).

```
            EAST → WEST row reduce              NORTH ← SOUTH col reduce on column 0
   (P-1,y) ─► (P-2,y) ─► … ─► (0,y)              (0,P-1) ─► (0,P-2) ─► … ─► (0,0)
```

Two color pairs carry the merge stream — `c_row_dist`/`c_row_idx` for the row reduce, and
`c_col_dist`/`c_col_idx` for the column reduce on column 0. Each pair `(distance, index)` is two
back-to-back wavelets (f32 + i32) on the same physical fiber; using parallel colors lets a single
`@mov32` DSD send both arrays without bit-packing. The host broadcasts `q` and the shards via
`memcpy`; PE `(0,0)` writes the final K results back to a memcpy-visible buffer.

## 2. Local top-K algorithm

Per PE, after computing `S` squared L2 distances, I maintain a **K-element max-heap keyed by
`(dist, idx)` lexicographically**. For each new candidate, compare against `heap[0]` (the worst
kept) — if better, replace and sift down. Cost: `O(S log K)` per PE, with the inner loop hot in
SRAM. For the hottest case (`baseline`, S=128, K=16) that's ≈128·log₂16 = 512 compares ≈ ~3 cycles
each = ~1.5 K cycles, dominated by the 128·32 = 4 K FMAs of the distance pass (~4 K cycles). For
`k_large` (S=256, K=256) the heap collapses to "keep all" so the cost degenerates to a single
final sort (`O(K log K)`), still cheaper than the distance pass.

Squared L2 is returned (not the sqrt) — ordering is identical and we save `S` square roots per PE.
This is documented in `run.py` and the host-side comparison uses squared distances too.

## 3. Reduction & fabric bandwidth

The merge is **2D-separable**: each row reduces east-to-west to column 0, then column 0 reduces
south-to-north to (0,0). At each hop the receiver merges its current top-K with K incoming pairs,
keeping the better K. Total pairs crossing any single edge in the worst case = `K`. With `2 K`
wavelets (one f32 + one i32 per pair) per hop and `(P − 1)` hops per dimension, total fabric
traffic is `2·K·(P−1)·P` wavelets row-wise + `2·K·(P−1)` column-wise. For `baseline` (K=16, P=4):
192 + 96 = 288 wavelets — three orders of magnitude below the ~4 K cycles of compute, so the
kernel is **compute-bound, not routing-bound**, on every test case. That's deliberate: I picked
the sequential reduce over a tree to keep code tiny and predictable; tree reduce would cut latency
to `log P` hops but the constant factors and code size aren't worth it at P ∈ {2, 4}.

## 4. Tie-breaking (proof of determinism)

The merge primitive on each PE is: given two sorted-ascending lists `A`, `B` keyed by
`(dist, idx)`, produce the K smallest under the same key. Because the lex key `(d, i)` is a
**total order** on the candidate set (every row has a unique `idx`, so equal distances are still
distinguishable), the K-smallest set is uniquely defined regardless of which list is "A" and which
is "B" — `merge(A, B) = merge(B, A)`. The local top-K heap also uses the lex key, so its output
order is determined entirely by the input row data, not by which row arrived first. Therefore the
final top-K at PE (0,0) is a function of the input data only — fabric-arrival order changes
nothing. The padded rows (idx = N) sort last under any tie, so they never displace a real row.

## 5. With 2× more time

1. **Tree reduce in each dimension.** Pair `(P-1,y)↔(P-2,y)` and `(P-3,y)↔(P-4,y)` in parallel,
   then merge the winners. Cuts critical-path hops from `P-1` to `log₂ P`. At P=4 that's
   3 hops → 2 hops; small absolute win, larger if N grows and we go to P=8. Estimated payoff:
   ~30% on the reduce phase, ~5% end-to-end.
2. **Vectorize the distance loop with DSDs.** The current loop is scalar `f32` over `d_dim`. A
   stride-1 DSD over `d_dim` with FMAC into a scalar accumulator should hit ~2 lanes/cycle on
   WSE-2's f32 unit. Estimated payoff: ~40% on the dominant compute phase.
