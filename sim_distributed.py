"""Pure-Python simulation of the distributed Top-K algorithm.

Mimics what pe_program.csl does step-for-step:
  1. Shard D into P*P chunks of rows_per_pe rows (zero-padded).
  2. Each PE computes squared L2 distances and a K-element local top-K
     via a (dist, idx) lex-keyed max-heap.
  3. Row reduce east → west: each row collapses to column 0.
  4. Column reduce south → north on column 0: collapses to PE (0, 0).

If this matches reference.py's oracle on all 6 test cases, the algorithm
and the tie-break are correct. Any remaining bug in the actual submission
is then a CSL syntax / fabric routing bug, not an algorithmic one.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from reference import ALL_CASES, topk_reference


def _local_topk(D_shard: np.ndarray, q: np.ndarray, K: int,
                row_offset: int, N_total: int) -> list[tuple[float, int]]:
    """Mirror pe_program.csl::compute_local_topk."""
    POS_INF = float("inf")
    # Python's heapq is a min-heap. We want a max-heap keyed by (dist, idx),
    # so we negate via tuple inversion: store (-dist, -idx) for max-of-K.
    # But that breaks tie-breaking (we want SMALLEST idx to win at equal dist,
    # which means at the top of a max-heap we want the LARGEST idx).
    # Trick: store keys as (dist, idx) and use a "size-K heap of negatives".
    # Easier: maintain the list, push all S, then heapq.nsmallest.
    pairs: list[tuple[float, int]] = []
    for r in range(D_shard.shape[0]):
        global_idx = row_offset + r
        if global_idx >= N_total:
            d2 = POS_INF
            ix_for_break = N_total
        else:
            diff = D_shard[r] - q
            d2 = float(np.dot(diff, diff))
            ix_for_break = global_idx
        pairs.append((d2, ix_for_break))
    # heapq.nsmallest with default tuple ordering = lex (dist, idx) ascending.
    return heapq.nsmallest(K, pairs)


def _merge_topk(a: list[tuple[float, int]], b: list[tuple[float, int]],
                K: int) -> list[tuple[float, int]]:
    """Mirror pe_program.csl::merge_sorted_topk: keep K smallest under lex."""
    # a and b are already sorted ascending by (dist, idx). Standard 2-way merge.
    out: list[tuple[float, int]] = []
    i = j = 0
    while len(out) < K:
        if i >= len(a):
            out.append(b[j]); j += 1
        elif j >= len(b):
            out.append(a[i]); i += 1
        elif a[i] <= b[j]:           # tuple compare = lex (dist, idx)
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out


def simulate(D: np.ndarray, q: np.ndarray, K: int, P: int):
    N, d = D.shape
    rows_per_pe = math.ceil(N / (P * P))
    N_total = P * P * rows_per_pe

    # Pad D so every PE has rows_per_pe rows.
    D_pad = np.zeros((N_total, d), dtype=np.float32)
    D_pad[:N] = D
    shards = D_pad.reshape(P * P, rows_per_pe, d)

    # Phase 1: local top-K on each PE. Index by (px, py) with pe_id = py*P+px.
    local: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for py in range(P):
        for px in range(P):
            pe_id = py * P + px
            local[(px, py)] = _local_topk(
                shards[pe_id], q, K,
                row_offset=pe_id * rows_per_pe,
                N_total=N,
            )

    # Phase 2: row reduce east → west. Source is px=P-1; sink is px=0.
    row_winner: dict[int, list[tuple[float, int]]] = {}
    for py in range(P):
        cur = local[(P - 1, py)]
        for px in range(P - 2, -1, -1):
            cur = _merge_topk(cur, local[(px, py)], K)
        row_winner[py] = cur

    # Phase 3: column reduce on column 0, south → north. Source py=P-1, sink py=0.
    cur = row_winner[P - 1]
    for py in range(P - 2, -1, -1):
        cur = _merge_topk(cur, row_winner[py], K)

    # Truncate sentinel pads (idx == N) — they'd only appear if K > N which the
    # oracle also doesn't allow.
    indices = np.array([ix for _, ix in cur], dtype=np.int32)
    distances = np.array([d for d, _ in cur], dtype=np.float32)
    return indices, distances


def main() -> int:
    failures = 0
    for maker in ALL_CASES:
        case = maker()
        sim_idx, sim_dist = simulate(case["D"], case["q"], case["K"], case["P"])
        ref_idx, ref_dist = topk_reference(case["D"], case["q"], case["K"], squared=True)

        idx_ok = np.array_equal(sim_idx, ref_idx)
        dist_ok = np.allclose(sim_dist, ref_dist, atol=1e-3, rtol=1e-3)
        status = "PASS" if (idx_ok and dist_ok) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"{status}: {case['name']:12s}  K={case['K']:>3d}  P={case['P']}  "
              f"idx_ok={idx_ok}  dist_ok={dist_ok}")
        if not idx_ok:
            print(f"   sim idx[:8]:  {sim_idx[:8].tolist()}")
            print(f"   ref idx[:8]:  {ref_idx[:8].tolist()}")
        if not dist_ok:
            print(f"   sim dist[:4]: {sim_dist[:4].tolist()}")
            print(f"   ref dist[:4]: {ref_dist[:4].tolist()}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
