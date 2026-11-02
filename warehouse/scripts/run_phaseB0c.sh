#!/bin/bash
# Phase B0c: calibration, then PIBT, the main planner, and prioritized last, cheapest cells first, each cell with a and b before the
# arms that read them. At most 4 processes; a rerun skips what is done, and the .py is read anew at each stage.
set -u
cd "$(dirname "$0")/.."
source ../env.sh
export OMP_NUM_THREADS=1
step() {
    echo "=== $*  $(date)"
    nice -n 10 python -u scripts/phaseB0c_realism.py "$@" --workers 4 || echo "FAILED $*"
}
step calibrate
for planner in pibt prioritized; do
    for cell in m30 l60 m90; do
        step run --cells $cell --planners $planner --arms a b
        step run --cells $cell --planners $planner --arms c d e e0 f
    done
done
echo "=== ALL DONE $(date)"
