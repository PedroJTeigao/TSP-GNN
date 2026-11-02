#!/bin/bash
# Phase B0c: calibration, the trips, then PIBT, the main planner, cheapest cell first, each with a and b before the arms that read them,
# and prioritized last, where the budget can cut it. At most 4 processes; a rerun skips what is done, and the .py is read at each stage.
set -u
cd "$(dirname "$0")/.."
source ../env.sh
export OMP_NUM_THREADS=1
step() {
    echo "=== $*  $(date)"
    nice -n 10 python -u scripts/phaseB0c_realism.py "$@" --workers 4 || echo "FAILED $*"
}
stages() {
    step run --cells "$1" --planners "$2" --arms a b
    step run --cells "$1" --planners "$2" --arms ${3:-c d e e0 f}
}
step calibrate
step trips
for cell in m30 l60 m90; do stages $cell pibt; done
stages m30 prioritized
stages m90 prioritized "c d e f"
echo "=== ALL DONE $(date)"
