#!/bin/bash
# Phase B0b: T1, T2, T3 from the lowest density up, then the check over every trip. At most 4 processes; a rerun skips what is done.
set -u
cd "$(dirname "$0")/.."
source ../env.sh
export OMP_NUM_THREADS=1
step() {
    echo "=== $*  $(date)"
    nice -n 10 python -u scripts/phaseB0b_margins.py "$@" --workers 4 || echo "FAILED $*"
}
step t1
step t2 --sizes medium
step t2 --sizes large
step t3edges
for x in 1.0 1.5 2.0 3.0 4.0 5.0; do step t3 --densities $x; done
step t1all
echo "=== ALL DONE $(date)"
