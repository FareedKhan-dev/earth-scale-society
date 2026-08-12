#!/usr/bin/env bash
# The teacher. Run this in its own shell and leave it up: the notebook talks to
# it over HTTP when it reaches the Herald cells.
#
#   235 GB of FP8 weights into 320 GB of HBM, leaving ~19 GB per GPU for KV.
#   Prefix caching matters more than it looks. Sorting the 270,000 questions by
#   speaker makes the 590-token shared prefix hit 96% of the time, which takes
#   the effective bill from 1,048 tokens a call down to 470.
set -euo pipefail

vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --guided-decoding-backend xgrammar \
  --port 8000
