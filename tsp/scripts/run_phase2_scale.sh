#!/usr/bin/env bash
# Phase 2, scale experiment: how far does the whole-instance GNN go? Gap vs. n, no retraining.
#
# Two lanes run at the same time:
#   data lane (background)  new test sets with LKH tours and LKH times, smallest first
#                           (scripts/make_scale_data.py -> data/tsp<n>_test.npz; 3 instances in parallel)
#       n      200  500  1000  2000  5000  10000
#       num    100   50    20    10     5      3
#       seed     9   10    11    12    13     14
#       runs    10   10    10     1     1      1   (LKH RUNS; 1 run above 1000 cities, to keep the time sane)
#   eval lane (foreground)  for n = 20, 50, 100 (phase 1 test sets) and then the new sizes, smallest first,
#                           as soon as each set exists:
#       baselines   scripts/scale_baselines.py -> results/phase2_scale/baselines/
#       models      scripts/scale_eval.py, k-modes k50 and khalf100 -> results/phase2_scale/<run>/<k-mode>/
#                   GCN, GATv2, Gated (phase 1) and TAGCN K=3, best.pt, seeds 0-2
# At the end the report results/phase2_scale_report.md is generated (if the report script exists).
# Progress goes to results/phase2_scale.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
#
# Launch (survives closing SSH):
#   tmux new-session -d -s scale "nohup bash scripts/run_phase2_scale.sh > results/phase2_scale.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=2
OUT=results/phase2_scale
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" results/figs

LOG=results/phase2_scale.log
RUNS=()
for m in gcn gat gated tagcn_k3; do for s in 0 1 2; do RUNS+=("${m}_seed${s}"); done; done
DATA_SPECS=("200 100 9 10" "500 50 10 10" "1000 20 11 10" "2000 10 12 1" "5000 5 13 1" "10000 3 14 1")
SIZES=(20 50 100 200 500 1000 2000 5000 10000)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {
    local desc=$1 out=$2; shift 2
    log "START $desc"
    local start=$SECONDS
    "$@" > "$out" 2>&1
    local code=$?
    if [ "$code" -eq 0 ]; then
        log "OK    $desc ($(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s)"
        return 0
    fi
    touch "$FAILDIR/$(echo "$desc" | tr -c 'A-Za-z0-9_=.-' '_')"
    log "FAILED $desc (exit code $code). Last lines of $out:"
    tail -n 15 "$out" | sed 's/^/    /' | tee -a "$LOG"
    return 1
}

data_lane() {
    local spec n num seed runs
    for spec in "${DATA_SPECS[@]}"; do
        read -r n num seed runs <<< "$spec"
        if [ -f "data/tsp${n}_test.npz" ]; then
            log "SKIP  data tsp${n}_test (already exists)"
        elif ! run "data tsp${n}_test ($num instances, seed $seed, LKH runs $runs)" "$LOGDIR/data_tsp${n}.log" \
                python scripts/make_scale_data.py --n "$n" --num "$num" --seed "$seed" --runs "$runs" --procs 3; then
            touch "$LOGDIR/data_tsp${n}.failed"; continue
        fi
        touch "$LOGDIR/data_tsp${n}.done"       # the eval lane waits for this file, not for the .npz
    done
}

log "=== phase 2 scale experiment — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for n in 20 50 100; do touch "$LOGDIR/data_tsp${n}.done"; done   # phase 1 test sets: already there
data_lane &
DATA_PID=$!

for n in "${SIZES[@]}"; do
    name=tsp${n}_test
    while [ ! -f "$LOGDIR/data_tsp${n}.done" ] && [ ! -f "$LOGDIR/data_tsp${n}.failed" ]; do sleep 60; done
    if [ -f "$LOGDIR/data_tsp${n}.failed" ]; then
        log "SKIP  everything on $name (no data)"; continue
    fi
    if [ -f "$OUT/baselines/${name}.json" ]; then
        log "SKIP  baselines $name (already done)"
    else
        run "baselines $name" "$LOGDIR/baselines_${name}.log" python scripts/scale_baselines.py \
            --data "data/${name}.npz" --out-dir "$OUT/baselines" --workers 4
    fi
    for mode in k50 khalf100; do
        missing=0
        for r in "${RUNS[@]}"; do [ -f "$OUT/$r/$mode/${name}.json" ] || missing=1; done
        if [ "$missing" -eq 0 ]; then log "SKIP  models $name $mode (already done)"; continue; fi
        run "models $name $mode" "$LOGDIR/models_${name}_${mode}.log" python scripts/scale_eval.py \
            --data "data/${name}.npz" --k-mode "$mode" --runs "${RUNS[@]}" --out-root "$OUT" --workers 4
    done
done
wait "$DATA_PID"

if [ -f scripts/phase2_scale_report.py ]; then
    if run "scale report" "$LOGDIR/report.log" python scripts/phase2_scale_report.py; then
        cp results/phase2_scale_report.md docs/phase2_scale_report.md && mkdir -p docs/figs \
            && cp results/figs/phase2_scale_*.png docs/figs/ && log "copied to docs/phase2_scale_report.md (+ figures)"
    fi
fi

FAILURES=$(find "$FAILDIR" -type f | wc -l)
log "failed steps: $FAILURES"
log "ALL DONE ($FAILURES failures)"
