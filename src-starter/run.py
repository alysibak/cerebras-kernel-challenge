"""Host driver for the Top-K k-NN kernel.

Loads the test case from `reference.py`, memcpys the sharded database and
query vector to the wafer, kicks off `compute`, reads back the K results
from PE (0,0), and verifies against the NumPy oracle. Prints
`PASS: <case>` on success, exits non-zero on any mismatch.

CLI (matches the grader contract in tests/test_correctness.py):
    cs_python run.py --name <build_dir> --case <case_name>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Repo root holds reference.py with the oracle + case generators.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reference import ALL_CASES, topk_reference  # type: ignore

from cerebras.sdk.runtime.sdkruntimepybind import (  # type: ignore
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)


# Display name (reference.py) → snake_case key (grader, --case).
_DISPLAY_TO_KEY = {
    "baseline": "baseline",
    "k=1": "k_eq_1",
    "k=256": "k_large",
    "uneven": "uneven",
    "all-equal": "all_equal",
    "duplicates": "duplicates",
}


def _load_case(case_key: str) -> dict:
    for maker in ALL_CASES:
        case = maker()
        if _DISPLAY_TO_KEY[case["name"]] == case_key:
            return case
    raise SystemExit(f"unknown case: {case_key}")


def _shard(D: np.ndarray, P: int, rows_per_pe: int) -> np.ndarray:
    """Return shape (P*P, rows_per_pe, d) shard tensor, padded with zeros.

    PE id = py * P + px (row-major) owns rows
    [pe_id * rows_per_pe : (pe_id + 1) * rows_per_pe). Padded rows are
    treated as +inf inside the kernel via the global-index check.
    """
    N, d = D.shape
    total = P * P * rows_per_pe
    padded = np.zeros((total, d), dtype=np.float32)
    padded[:N] = D
    return padded.reshape(P * P, rows_per_pe, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="cslc output build dir")
    ap.add_argument("--case", required=True, help="snake_case case key")
    args = ap.parse_args()

    case = _load_case(args.case)
    D, q, K, P = case["D"], case["q"], case["K"], case["P"]
    N, d = D.shape
    rows_per_pe = math.ceil(N / (P * P))

    # Oracle (squared distances; kernel returns squared too).
    oracle_idx, oracle_dist = topk_reference(D, q, K, squared=True)

    # Build shard tensor and per-PE q broadcast tensor.
    D_shards = _shard(D, P, rows_per_pe).astype(np.float32)        # (P*P, rows_per_pe, d)
    q_broadcast = np.broadcast_to(q.astype(np.float32), (P, P, d)).copy()

    # Reshape for memcpy: SdkRuntime memcpy_h2d expects (height, width, ...)
    # with width = x dim (px), height = y dim (py).
    D_h2d = D_shards.reshape(P, P, rows_per_pe, d)  # [py, px, row, dim]

    runner = SdkRuntime(args.name, suppress_simfab_trace=True)

    # Resolve symbols.
    sym_D = runner.get_id("D_shard")
    sym_q = runner.get_id("q_data")
    sym_idx = runner.get_id("result_indices")
    sym_dist = runner.get_id("result_distances")

    runner.load()
    runner.run()

    # Send shards: (P, P, rows_per_pe * d) f32 per PE.
    runner.memcpy_h2d(
        sym_D,
        D_h2d.reshape(-1).astype(np.float32),
        0, 0, P, P, rows_per_pe * d,
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    # Send q to every PE.
    runner.memcpy_h2d(
        sym_q,
        q_broadcast.reshape(-1).astype(np.float32),
        0, 0, P, P, d,
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    # Trigger compute on every PE via the exported RPC.
    runner.launch("compute", nonblock=False)

    # Read back K results from PE (0, 0) only.
    out_idx = np.zeros(K, dtype=np.int32)
    out_dist = np.zeros(K, dtype=np.float32)
    runner.memcpy_d2h(
        out_idx, sym_idx,
        0, 0, 1, 1, K,
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )
    runner.memcpy_d2h(
        out_dist, sym_dist,
        0, 0, 1, 1, K,
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    runner.stop()

    # Diff vs oracle.
    if not np.array_equal(out_idx, oracle_idx):
        print(f"FAIL: {args.case} — index mismatch")
        print(f"  got:      {out_idx.tolist()}")
        print(f"  expected: {oracle_idx.tolist()}")
        return 1
    if not np.allclose(out_dist, oracle_dist, atol=1e-3, rtol=1e-3):
        print(f"FAIL: {args.case} — distance mismatch")
        print(f"  got:      {out_dist.tolist()}")
        print(f"  expected: {oracle_dist.tolist()}")
        return 1

    print(f"PASS: {args.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
