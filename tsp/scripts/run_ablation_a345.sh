#!/usr/bin/env bash
# Ablations A3, A4, A5 (2026-11-03): see docs/ablations_2026-11-03.md. CPU only, at most 4 of our processes.
#
# Waits for "ALL DONE a12" in results/ablations/a12.log (A1/A2 measure times and run alone), then in parallel:
#   lane 1: A5, warehouse (scripts/ablation_a5.py run, 3 workers), then its table
#   lane 2: A3, the empty-rule episodes of phase C (1 worker) and both tables; then A4: tau, features, retraining,
#           window and triage runs one after the other
# Everything resumes (finished outputs are skipped). Log: results/ablations/a345.log ("ALL DONE a345" at the end).
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s abl_a345 "nohup bash scripts/run_ablation_a345.sh > results/ablations/a345.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi
source ../env.sh
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
OUT=results/ablations
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
LOG=$OUT/a345.log
FAILDIR=$LOGDIR/failures_a345
mkdir -p "$FAILDIR"
TSP=$PWD
WH=$PWD/../warehouse

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$TSP/$LOG"; }
run() {       # DIR LABEL command...
    local dir=$1 label=$2; shift 2
    local out="$TSP/$LOGDIR/$(echo "$label" | tr ' /' '__').log"
    log "START $label"
    local start=$SECONDS
    if (cd "$dir" && "$@") > "$out" 2>&1; then
        log "OK    $label ($(( SECONDS - start )) s)"
    else
        touch "$FAILDIR/$(echo "$label" | tr ' /' '__')"
        log "FAILED $label (see $out)"
    fi
}

log "=== ablations A3-A5 — commit $(git rev-parse --short HEAD); waiting for A1/A2 ==="
until grep -q "ALL DONE a12" $OUT/a12.log 2>/dev/null; do sleep 60; done
log "A1/A2 finished: starting"

lane1() {
    run "$WH" "a5 run" python -u scripts/ablation_a5.py run --workers 3
    run "$WH" "a5 table" python -u scripts/ablation_a5.py table
}
lane2() {
    run "$WH" "a3 warehouse run" python -u scripts/ablation_a3_warehouse.py run --workers 1
    run "$WH" "a3 warehouse table" python -u scripts/ablation_a3_warehouse.py table
    run "$TSP" "a3 tsp table" python -u scripts/ablation_a3_tsp.py
    run "$TSP" "a4 tau" python -u scripts/ablation_a4.py tau
    run "$TSP" "a4 features" python -u scripts/ablation_a4.py features
    run "$TSP" "a4 train" python -u scripts/learned_triage_train.py --seed 0 --features-root $OUT/a4/features \
        --out-root $OUT/a4/models --test-sets tsp1000_test tsp10000_test_concorde
    for set in fu1k fu10k; do
        for m in greedy beam10 2opt; do run "$TSP" "a4 window $set $m" python -u scripts/ablation_a4.py window --set $set --method $m; done
        for s in dis learned_old learned_new; do run "$TSP" "a4 triage $set $s" python -u scripts/ablation_a4.py triage --set $set --score $s; done
    done
}
lane1 &
lane2 &
wait

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE a345 with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE a345 (0 failures)"
fi
