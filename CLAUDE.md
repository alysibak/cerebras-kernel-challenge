# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Cerebras AI Engineering Internship take-home: implement a **Top-K k-NN kernel in CSL** (Cerebras Software Language) that runs on the Wafer-Scale Engine (WSE-2). The repo as committed is a *challenge packet*, not a working kernel — `src-starter/` contains only a README and the candidate is expected to add `layout.csl`, one or more PE program files, `run.py`, `commands.sh`, and `DESIGN.md`.

`SPEC.md` is the authoritative problem statement; `README.md` is the onramp. Read both before changing anything.

## Toolchain

The Cerebras SDK (`cslc`, `cs_python`, `csdb`) is **Linux + Apptainer/Singularity only** and is **not vendored in this repo**. It must be downloaded separately (link in `README.md`). On macOS / Windows, build the image in `docker/` (see `docker/README.md`) and work inside the container. Apple Silicon: turn OFF Rosetta in Docker Desktop — the SDK uses AVX2 and Rosetta crashes with `Illegal instruction` (exit 132).

Sanity check the toolchain before writing any CSL:

```bash
cslc --help
cs_python -c 'import numpy; print(numpy.__version__)'
python3 reference.py    # prints expected top-K for all 6 cases
```

## Build & run a candidate submission

The grader (`tests/test_correctness.py`) compiles each case with these exact flags — match them in any local `commands.sh`:

```bash
cslc --arch=wse2 layout.csl \
  --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 \
  --memcpy --channels=1 \
  -o out_baseline

cs_python run.py --name out_baseline --case baseline
```

`--memcpy --channels=1` is mandatory (selects SdkRuntime over the deprecated CSELFRunner). `run.py` must accept `--name <build_dir>` and `--case <case_name>`, and must print `PASS: <case_name>` on success — the grader greps stdout for that exact marker.

## Run the grader

```bash
cd tests
python3 -m pytest test_correctness.py -v --submission=../src-starter
# or point at wherever your sources live
python3 -m pytest test_correctness.py -v --submission=../my-solution
# single case
python3 -m pytest test_correctness.py -v --submission=../src-starter -k baseline
```

The grader requires `--submission`; it asserts `layout.csl` and `run.py` exist there, then for each of the 6 cases compiles + runs + checks for the `PASS:` marker. CI runs this same script.

## Test cases (must all pass)

Defined in `reference.py::ALL_CASES` (Python display names) and re-keyed in `tests/test_correctness.py::CASE_PARAMS` (snake_case keys passed to `cslc --params`). **Keep these two lists in sync** — `cycles.json` also keys off the snake_case names.

| case | P | d | rows/PE | K | tests |
|---|---|---|---|---|---|
| `baseline`   | 4 | 32 | 128 | 16  | golden path, N=2048 |
| `k_eq_1`     | 2 | 32 | 256 | 1   | argmin |
| `k_large`    | 2 | 16 | 256 | 256 | K saturates SRAM |
| `uneven`     | 4 | 32 | 64  | 16  | N=1009 (prime) → short last shard |
| `all_equal`  | 2 | 16 | 256 | 16  | every tie-break must favor smaller index → output is `[0..K-1]` |
| `duplicates` | 2 | 16 | 256 | 8   | exact-duplicate rows at non-adjacent indices; naive merges drop one |

`all_equal` and `duplicates` are the tie-breaking tests — **don't skip them**; they catch merge bugs that the random cases mask.

## Correctness contract

- Output `(indices, distances)` is sorted ascending by `(distance, original_index)` — see `reference.py::topk_reference` (uses `np.lexsort((idx_all, d2))`).
- Squared L2 is acceptable as long as ordering matches; document the choice.
- Distances compared with `allclose` at `1e-3`; **indices must be bit-identical**.
- Must be deterministic across runs (no nondeterminism from fabric arrival order — the merge reduction has to break ties by index, not by who arrived first).

## Performance gate

`cycles.json` holds per-case reference cycle counts measured by `csdb`. Gate is `candidate_cycles ≤ multiplier * reference_cycles` (multiplier currently 1.3). Entries currently `null` are **waived with no retroactive enforcement** — failing perf on a case loses only that case's share of perf points; correctness on `baseline` is the only hard fail.

## Architecture notes for kernel work

- The database `D` is sharded `ceil(N / P²)` rows per PE across a `P × P` PE grid. Host broadcasts `q`, then no host involvement until result readback.
- PE SRAM is **~48 KB**. Oversized buffers fail at link-time, not compile-time. The `k_large` case (`2 * K * P * P` u32s of pair buffer + 16 KB shard) is the tightest fit.
- Task IDs share a 32-slot namespace with colors; many are reserved by `memcpy` and collectives — check the color-map comment in `topic-11-collectives/layout.csl` before picking IDs.
- Collective callbacks fire on **every PE in the group**, not just the destination — structure the task state machine accordingly.
- Suggested SDK reading (from `src-starter/README.md`): `gemv-00-basic-syntax`, `gemv-05-multiple-pes`, `gemv-06..08-routes`, `topic-11-collectives`, `topic-05-sentinels`, `benchmarks/gemv-collectives_2d` (best P×P template), `benchmarks/histogram-torus` (wavelet bit-packing, termination detection).

## Submission deliverables

Per SPEC §3, a complete submission includes `src/layout.csl`, PE program files, `run.py`, `tests/test_correctness.py` (already provided), `commands.sh`, and **`DESIGN.md` (one page, ~400–600 words)** covering: routing topology, local top-K algorithm + Big-O, fabric bandwidth accounting, tie-breaking proof, and what you'd do with 2× more time. Per the spec, the memo is the primary anti-plagiarism device — graders read it closely.
