#!/usr/bin/env bash
# Build and run the baseline case end-to-end. Mirrors what the grader does
# in tests/test_correctness.py for case `baseline`.
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-out_baseline}"

cslc --arch=wse2 layout.csl \
  --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 \
  --memcpy --channels=1 \
  -o "${BUILD_DIR}"

cs_python run.py --name "${BUILD_DIR}" --case baseline
