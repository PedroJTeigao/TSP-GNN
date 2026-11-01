#!/bin/bash
# Phase A: every experiment, in order, at most 4 processes. A rerun skips what is done. Run it inside tmux, from warehouse/.
set -u
cd "$(dirname "$0")/.."
source ../env.sh
export OMP_NUM_THREADS=1
for what in seq e1 e2 e3 gif; do
    echo "=== $what  $(date)"
    nice -n 10 python -u scripts/phaseA_baselines.py $what --workers 4 || echo "FAILED $what"
done
echo "=== ALL DONE $(date)"
