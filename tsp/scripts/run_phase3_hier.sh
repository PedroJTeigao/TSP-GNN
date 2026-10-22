#!/usr/bin/env bash
# Phase 3: hierarchical TSP with the TSP-20 model (TAGCN K=3, best.pt, seeds 0-2), no retraining.
#
# For n = 200, 500, 1000, 2000, 5000, 10000 (the existing scale-experiment test sets, smallest first),
# the 3 seeds run in parallel (scripts/hier_eval.py, all 16 variants) -> results/phase3_hier/tagcn_k3_seed<s>/.
# At the end the report results/phase3_hier_report.md is generated (if the report script exists) and copied to docs/.
# Progress goes to results/phase3_hier.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
#
# Launch (survives closing SSH):
#   tmux new-session -d -s hier "nohup bash scripts/run_phase3_hier.sh > results/phase3_hier.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=2
OUT=results/phase3_hier
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" results/figs

LOG=results/phase3_hier.log
SIZES=(200 500 1000 2000 5000 10000)
SEEDS=(0 1 2)

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

log "=== phase 3 hierarchical TSP — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for n in "${SIZES[@]}"; do
    name=tsp${n}_test
    PIDS=()
    for s in "${SEEDS[@]}"; do
        if [ -f "$OUT/tagcn_k3_seed$s/$name.json" ]; then log "SKIP  $name seed $s (already done)"; continue; fi
        run "$name seed $s" "$LOGDIR/${name}_seed$s.log" python scripts/hier_eval.py \
            --data "data/$name.npz" --seed "$s" --out-dir "$OUT/tagcn_k3_seed$s" &
        PIDS+=($!)
    done
    [ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"
done

if [ -f scripts/phase3_hier_report.py ]; then
    if run "hier report" "$LOGDIR/report.log" python scripts/phase3_hier_report.py; then
        cp results/phase3_hier_report.md docs/phase3_hier_report.md && mkdir -p docs/figs \
            && cp results/figs/phase3_hier_*.png docs/figs/ && log "copied to docs/phase3_hier_report.md (+ figures)"
    fi
fi

FAILURES=$(find "$FAILDIR" -type f | wc -l)
log "failed steps: $FAILURES"
log "ALL DONE ($FAILURES failures)"
