#!/usr/bin/env bash
# Phase 3, learned triage, step 1: window logs of the Fu sets (1k: train/validation, 10k: test) for the 3 TAGCN K=3
# seeds, then per-window features and labels for every log (tsp200 / 1000 / 10000 from the triage experiment,
# Fu-1k / Fu-10k from here). The 3 seeds run in parallel (3 processes, one LKH call at a time each; GPU ~0.25 GiB
# each).
# Outputs: results/phase3_learned_triage/{log,features}/tagcn_k3_seed<s>/<dataset>.{npz,json}.
# Log: results/phase3_learned_triage/data.log.
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s learned_data "nohup bash scripts/run_phase3_learned_data.sh > results/phase3_learned_triage/data.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase3_learned_triage
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
LOG=$OUT/data.log
FAILS=0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {
    local desc=$1 out=$2; shift 2
    log "START $desc"
    local start=$SECONDS
    if "$@" > "$out" 2>&1; then
        log "OK    $desc ($(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s)"
    else
        log "FAILED $desc (see $out)"
        return 1
    fi
}

seed_job() {
    local s=$1 fails=0
    for name in tsp1000_test_concorde tsp10000_test_concorde; do
        run "window log $name seed $s" "$LOGDIR/log_${name}_seed$s.log" python -u scripts/triage_log.py \
            --data results/phase3_benchmark/data/$name.npz --seed $s --out-root $OUT/log || fails=$((fails + 1))
    done
    for name in tsp200_test tsp1000_test tsp10000_test; do
        run "features $name seed $s" "$LOGDIR/features_${name}_seed$s.log" python -u scripts/learned_triage_data.py \
            --log results/phase3_triage/log/tagcn_k3_seed$s/$name.npz --data data/$name.npz --seed $s \
            || fails=$((fails + 1))
    done
    for name in tsp1000_test_concorde tsp10000_test_concorde; do
        run "features $name seed $s" "$LOGDIR/features_${name}_seed$s.log" python -u scripts/learned_triage_data.py \
            --log $OUT/log/tagcn_k3_seed$s/$name.npz --data results/phase3_benchmark/data/$name.npz --seed $s \
            || fails=$((fails + 1))
    done
    return $fails
}

log "=== learned triage data — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
PIDS=()
for s in 0 1 2; do seed_job $s & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait $p || FAILS=$((FAILS + $?)); done
log "ALL DONE learned triage data ($FAILS failures)"
