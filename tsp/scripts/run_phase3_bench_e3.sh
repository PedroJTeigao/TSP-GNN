#!/usr/bin/env bash
# Phase 3 benchmark, E3: our methods on the standard test sets of Fu et al. (tsp1000_test_concorde, 128
# instances; tsp10000_test_concorde, 16), with the gap against the E0 LKH-3 reference.
#
# 0. scripts/bench_datasets.py -> results/phase3_benchmark/data/<set>.npz (lengths = E0 reference)
# Per set:
# 1. scripts/triage_eval.py, TAGCN K=3 seeds 0-2 (P0, P2 top 10/20/30/50 %, P3; thresholds from the
#    phase 3 tsp200 record) and P1 (LKH on all windows), 4 processes in parallel as in the triage experiment
#    -> results/phase3_benchmark/e3/policies/{tagcn_k3_seed<s>,lkh}/<set>.{json,npz}
#    (also gives the greedy edge + 2-opt baseline, "initial__gap")
# 2. scripts/window_eval.py, model windows (greedy and beam B=10), w = 20, seeds 0-2 in parallel
#    -> results/phase3_benchmark/e3/window/tagcn_k3_seed<s>/<set>.{json,npz}
# E1 at the 10k budgets on these sets was already run by run_phase3_bench_e1.sh.
# Log: results/phase3_benchmark/e3.log (START / OK / FAILED / SKIP lines, then "ALL DONE e3").
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s bench_e3 "nohup bash scripts/run_phase3_bench_e3.sh > results/phase3_benchmark/e3.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase3_benchmark
E3=$OUT/e3
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures_e3
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" "$E3"
LOG=$OUT/e3.log
SETS=(tsp1000_test_concorde tsp10000_test_concorde)
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

one_set() {
    local name=$1 data=$OUT/data/$1.npz PIDS=()
    for s in "${SEEDS[@]}"; do
        if [ -f "$E3/policies/tagcn_k3_seed$s/$name.json" ]; then log "SKIP  $name triage seed $s (already done)"; continue; fi
        run "$name triage seed $s" "$LOGDIR/e3_${name}_triage_seed$s.log" python -u scripts/triage_eval.py \
            --data "$data" --seed "$s" --log-root results/phase3_triage/log --out-root "$E3/policies" &
        PIDS+=($!)
    done
    if [ -f "$E3/policies/lkh/$name.json" ]; then log "SKIP  $name P1 (already done)"
    else
        run "$name P1 lkh" "$LOGDIR/e3_${name}_P1.log" python -u scripts/triage_eval.py \
            --data "$data" --seed lkh --out-root "$E3/policies" &
        PIDS+=($!)
    fi
    [ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"
    PIDS=()
    for s in "${SEEDS[@]}"; do
        if [ -f "$E3/window/tagcn_k3_seed$s/$name.json" ]; then log "SKIP  $name window seed $s (already done)"; continue; fi
        run "$name window seed $s" "$LOGDIR/e3_${name}_window_seed$s.log" python -u scripts/window_eval.py \
            --data "$data" --solver model --seed "$s" --widths 20 --out-root "$E3/window" &
        PIDS+=($!)
    done
    [ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"
}

log "=== E3 — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
run "datasets with the E0 reference" "$LOGDIR/e3_datasets.log" python scripts/bench_datasets.py \
    --sets tsp1000_test tsp10000_test "${SETS[@]}"
for name in "${SETS[@]}"; do
    one_set "$name"
done

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE e3 with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE e3 (0 failures)"
fi
