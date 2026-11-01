#!/bin/bash
# Phase B0: every arm, in order of cost, at most 4 processes. A rerun skips what is done. Run it inside tmux, from warehouse/.
set -u
cd "$(dirname "$0")/.."
source ../env.sh
export OMP_NUM_THREADS=1
mkdir -p results/warehouse/phaseB0
for what in base online o1c o4cbs; do
    echo "=== $what  $(date)"
    nice -n 10 python -u scripts/phaseB0_margins.py $what --workers 4 || echo "FAILED $what"
done
echo "=== ALL DONE $(date)"
