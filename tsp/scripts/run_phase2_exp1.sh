#!/usr/bin/env bash
# Phase 2, experiment 1: edge-length normalization (knn_mean) vs. phase 1 (none).
#
# Runs sequentially:
#   a+b) train GCN, GATv2 and Gated GCN with --edge-norm knn_mean, seeds 0, 1, 2, on TSP-20
#        (phase 1 configuration) in results/runs/<model>_knnmean_seed<s>/, and evaluate each one
#        on TSP-20, TSP-50 and TSP-100
#   c)   evaluate the 9 phase 1 models on TSP-100 in results/phase2_exp1/phase1_tsp100/<model>_seed<s>/
#        (the phase 1 eval.json files are never touched)
#   d)   baselines on TSP-100 -> results/phase2_exp1/baselines_tsp100.json
#   e)   report -> results/phase2_exp1_table.md
# Progress goes to results/phase2_exp1.log, which ends with "ALL DONE" or "FAILED ...".
# Steps whose output already exists are skipped, so the script can be relaunched after a failure.
#
# Launch (survives closing SSH):
#   tmux new-session -d -s phase2_exp1 "bash scripts/run_phase2_exp1.sh"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

# Low CPU priority for the whole experiment (the children inherit it). Re-exec only once.
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
# $HOME is on AFS, whose tokens may expire after logout: keep caches and git config away from it.
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" results/phase2_exp1/phase1_tsp100

LOG=results/phase2_exp1.log
MODELS=(gcn gat gated)
SEEDS=(0 1 2)
TESTS=(data/tsp20_test.npz data/tsp50_test.npz data/tsp100_test.npz)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# run <description> <step log file> <command...>: run one step; on error, write FAILED and stop.
run() {
    local desc=$1 out=$2; shift 2
    log "START $desc"
    local start=$SECONDS
    if "$@" > "$out" 2>&1; then
        log "OK    $desc ($(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s)"
    else
        local code=$?
        log "FAILED $desc (exit code $code). Last lines of $out:"
        tail -n 15 "$out" | sed 's/^/    /' | tee -a "$LOG"
        exit "$code"
    fi
}

log "=== phase 2, experiment 1 — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for f in "${TESTS[@]}" data/tsp20_train.npz data/tsp20_val.npz; do
    [ -f "$f" ] || { log "FAILED missing data file $f"; exit 1; }
done

# a+b) knn_mean training + evaluation on TSP-20/50/100 (train_model.py evaluates the best model at the end)
for s in "${SEEDS[@]}"; do
    for m in "${MODELS[@]}"; do
        dir=results/runs/${m}_knnmean_seed${s}
        if [ -f "$dir/eval.json" ]; then log "SKIP  $dir (already done)"; continue; fi
        mkdir -p "$dir"
        run "train $m knn_mean seed $s" "$dir/train.log" \
            python scripts/train_model.py --model "$m" --seed "$s" --edge-norm knn_mean \
                --run-name "${m}_knnmean_seed${s}" --test "${TESTS[@]}"
    done
done

# c) phase 1 models (none) on TSP-100, written to a new folder
for m in "${MODELS[@]}"; do
    for s in "${SEEDS[@]}"; do
        out=results/phase2_exp1/phase1_tsp100/${m}_seed${s}
        if [ -f "$out/eval.json" ]; then log "SKIP  $out (already done)"; continue; fi
        mkdir -p "$out"
        run "evaluate phase 1 $m seed $s on TSP-100" "$out/evaluate.log" \
            python scripts/evaluate.py --run-dir "results/runs/${m}_seed${s}" \
                --test data/tsp100_test.npz --out-dir "$out"
    done
done

# d) baselines on TSP-100
if [ -f results/phase2_exp1/baselines_tsp100.json ]; then
    log "SKIP  baselines on TSP-100 (already done)"
else
    run "baselines on TSP-100" results/phase2_exp1/baselines_tsp100.log \
        python scripts/baselines.py --data data/tsp100_test.npz --out results/phase2_exp1/baselines_tsp100.json
fi

# e) report
run "report" results/phase2_exp1/table.log python scripts/phase2_exp1_table.py
log "table: results/phase2_exp1_table.md"
log "ALL DONE"
