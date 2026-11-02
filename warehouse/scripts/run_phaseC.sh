#!/bin/bash
# Phase C after the parser is frozen: L2 on the test requests, the translation arms, the hybrid, the episodes, and last the LLM latency with
# nothing of mine running. One model at a time; at most 4 processes for the episodes. A rerun resumes every stage.
# Launch from warehouse/: tmux new-session -d -s phC "scripts/run_phaseC.sh >> results/warehouse/phaseC/run.log 2>&1"
set -u
cd "$(dirname "$0")/.."
source ../env.sh
export OMP_NUM_THREADS=1
step() {
    echo "=== $*  $(date)"
    nice -n 10 python -u scripts/phaseC_language.py "$@" || echo "FAILED $*"
}
step l2 --split test
step b --arms p0 p1 q14 q7
step hybrid
step c --workers 4
step latency
echo "=== ALL DONE $(date)"
