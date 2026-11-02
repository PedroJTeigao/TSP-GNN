#!/bin/bash
# Phase B0, revised plan: O3, O1, O2, the free-flow limits, O4 and the CBS arm last. At most 4 processes; a rerun skips what is done.
set -u
cd "$(dirname "$0")/.."
source ../env.sh
export OMP_NUM_THREADS=1
for what in edges o3 o1 o2 ggo o2g ff o4; do
    echo "=== $what  $(date)"
    nice -n 10 python -u scripts/phaseB0_margins.py $what --workers 4 || echo "FAILED $what"
done
echo "=== NEW ARMS DONE $(date)"
echo "=== o4cbs  $(date)"
B0_RUN_CBS=1 nice -n 10 python -u scripts/phaseB0_margins.py o4cbs --workers 4 || echo "FAILED o4cbs"
echo "=== ALL DONE $(date)"
