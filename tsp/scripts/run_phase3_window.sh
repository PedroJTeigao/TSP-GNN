#!/usr/bin/env bash
# Phase 3: window refinement of the greedy edge + 2-opt tour (scripts/window_eval.py), no retraining.
#
# For n = 200, 500, 1000, 2000, 5000, 10000 (existing test sets, smallest first), 5 jobs in parallel:
# TAGCN K=3 seeds 0-2 (greedy and beam10 paths), LKH windows, 2-opt windows; widths 20, 30, 50.
#   -> results/phase3_window/{tagcn_k3_seed<s>,lkh,2opt}/tsp<n>_test.{json,npz}
# At the end the report results/phase3_window_report.md is generated (if the report script exists) and copied to docs/.
# Progress goes to results/phase3_window.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
#
# Launch (survives closing SSH):
#   tmux new-session -d -s window "nohup bash scripts/run_phase3_window.sh > results/phase3_window.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase3_window
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" results/figs

LOG=results/phase3_window.log
SIZES=(200 500 1000 2000 5000 10000)
JOBS=("model 0" "model 1" "model 2" "lkh -" "2opt -")

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

log "=== phase 3 window refinement — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for n in "${SIZES[@]}"; do
    name=tsp${n}_test
    PIDS=()
    for job in "${JOBS[@]}"; do
        read -r solver seed <<< "$job"
        folder=$solver; extra=()
        if [ "$solver" = model ]; then folder=tagcn_k3_seed$seed; extra=(--seed "$seed"); fi
        if [ -f "$OUT/$folder/$name.json" ]; then log "SKIP  $name $folder (already done)"; continue; fi
        run "$name $folder" "$LOGDIR/${name}_$folder.log" python scripts/window_eval.py \
            --data "data/$name.npz" --solver "$solver" "${extra[@]}" --out-root "$OUT" &
        PIDS+=($!)
    done
    [ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"
done

if [ -f scripts/phase3_window_report.py ]; then
    if run "window report" "$LOGDIR/report.log" python scripts/phase3_window_report.py; then
        cp results/phase3_window_report.md docs/phase3_window_report.md && mkdir -p docs/figs \
            && cp results/figs/phase3_window_*.png docs/figs/ && log "copied to docs/phase3_window_report.md (+ figures)"
    fi
fi

FAILURES=$(find "$FAILDIR" -type f | wc -l)
log "failed steps: $FAILURES"
log "ALL DONE ($FAILURES failures)"
