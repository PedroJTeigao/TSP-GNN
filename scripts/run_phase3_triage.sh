#!/usr/bin/env bash
# Phase 3: window triage — nothing / model / LKH per window (scripts/triage_log.py, scripts/triage_eval.py).
#
# 1. Per-window record (TAGCN K=3 seeds 0-2, w = 20) on tsp200 (validation), tsp1000, tsp10000
#    -> results/phase3_triage/log/tagcn_k3_seed<s>/tsp<n>_test.{json,npz}
# 2. Policies, thresholds from the tsp200 record of each seed: P0, P2, P3 per seed, P1 (LKH everywhere) once
#    -> results/phase3_triage/policies/{tagcn_k3_seed<s>,lkh}/tsp<n>_test.{json,npz}
# At the end the report results/phase3_triage_report.md is generated (if the report script exists) and copied to docs/.
# Progress goes to results/phase3_triage.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
#
# Launch (survives closing SSH):
#   tmux new-session -d -s triage "nohup bash scripts/run_phase3_triage.sh > results/phase3_triage.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source env.sh
export MPLCONFIGDIR="$PWD/tmp/mplconfig" XDG_CACHE_HOME="$PWD/tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase3_triage
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" results/figs

LOG=results/phase3_triage.log
SIZES=(200 1000 10000)
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

# one size: the 3 seed records in parallel with P1, then the 3 seeds' policies in parallel
size() {
    local n=$1 name=tsp$1_test PIDS=()
    for s in "${SEEDS[@]}"; do
        if [ -f "$OUT/log/tagcn_k3_seed$s/$name.json" ]; then log "SKIP  $name log seed $s (already done)"; continue; fi
        run "$name log seed $s" "$LOGDIR/${name}_log_seed$s.log" \
            python scripts/triage_log.py --data "data/$name.npz" --seed "$s" --out-root "$OUT/log" &
        PIDS+=($!)
    done
    if [ -f "$OUT/policies/lkh/$name.json" ]; then log "SKIP  $name P1 (already done)"
    else
        run "$name P1 lkh" "$LOGDIR/${name}_P1.log" \
            python scripts/triage_eval.py --data "data/$name.npz" --seed lkh --out-root "$OUT/policies" &
        PIDS+=($!)
    fi
    [ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"
    PIDS=()
    for s in "${SEEDS[@]}"; do
        if [ -f "$OUT/policies/tagcn_k3_seed$s/$name.json" ]; then log "SKIP  $name policies seed $s (already done)"; continue; fi
        if [ ! -f "$OUT/log/tagcn_k3_seed$s/tsp200_test.npz" ]; then log "SKIP  $name policies seed $s (no tsp200 record: no thresholds)"; continue; fi
        run "$name policies seed $s" "$LOGDIR/${name}_policies_seed$s.log" python scripts/triage_eval.py \
            --data "data/$name.npz" --seed "$s" --log-root "$OUT/log" --out-root "$OUT/policies" &
        PIDS+=($!)
    done
    [ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"
}

log "=== phase 3 window triage — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for n in "${SIZES[@]}"; do
    size "$n"
done

if [ -f scripts/phase3_triage_report.py ]; then
    if run "triage report" "$LOGDIR/report.log" python scripts/phase3_triage_report.py; then
        cp results/phase3_triage_report.md docs/phase3_triage_report.md && mkdir -p docs/figs \
            && cp results/figs/phase3_triage_*.png docs/figs/ && log "copied to docs/phase3_triage_report.md (+ figures)"
    fi
fi

FAILURES=$(find "$FAILDIR" -type f | wc -l)
log "failed steps: $FAILURES"
log "ALL DONE ($FAILURES failures)"
