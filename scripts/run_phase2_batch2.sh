#!/usr/bin/env bash
# Phase 2, batch 2: experiments 2, 2b, 3 and the TSP-50 control, then the report.
#
# Runs sequentially (fast steps first):
#   1) exp. 2   evaluate the phase 1 and knn_mean models on TSP-50/100 with --k-frac 0.5
#                 -> results/phase2_batch2/kfrac05/<run>/
#   2) exp. 2b  evaluate the phase 1 models on TSP-20/50/100 with --bn-batch-stats
#                 -> results/phase2_batch2/bnbatch/<run>/
#   3) exp. 3   train with --norm layernorm (TSP-20) -> results/runs/<model>_layernorm_seed<s>/
#               (evaluated with k = 10 at the end of training), then with --k-frac 0.5
#                 -> results/phase2_batch2/kfrac05/<model>_layernorm_seed<s>/
#   4) control  generate data/tsp50_train.npz + data/tsp50_val.npz, train on TSP-50
#                 -> results/runs/<model>_train50_seed<s>/ (evaluated on TSP-50/100)
#   5) report   results/phase2_batch2_report.md, copied to docs/phase2_batch2_report.md
# Progress goes to results/phase2_batch2.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
# A failed step does not stop the script: only the steps that depend on it are skipped.
# Steps whose output already exists are skipped, so the script can be relaunched safely.
#
# Launch (survives closing SSH):
#   tmux new-session -d -s phase2_batch2 "bash scripts/run_phase2_batch2.sh"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

# Low CPU priority for the whole batch (the children inherit it). Re-exec only once.
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source env.sh
# $HOME is on AFS, whose tokens may expire after logout: keep caches and git config away from it.
export MPLCONFIGDIR="$PWD/tmp/mplconfig" XDG_CACHE_HOME="$PWD/tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
OUT=results/phase2_batch2
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$OUT/kfrac05" "$OUT/bnbatch" "$OUT/logs"

LOG=results/phase2_batch2.log
MODELS=(gcn gat gated)
SEEDS=(0 1 2)
FAILURES=0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# run <description> <step log file> <command...>: returns 0 on success; on error logs FAILED and returns 1.
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
    FAILURES=$((FAILURES + 1))
    log "FAILED $desc (exit code $code). Last lines of $out:"
    tail -n 15 "$out" | sed 's/^/    /' | tee -a "$LOG"
    return 1
}

# evaluate <description> <run dir> <out dir> <extra evaluate.py args...>: skip if already done.
evaluate() {
    local desc=$1 run_dir=$2 out=$3; shift 3
    if [ -f "$out/eval.json" ]; then log "SKIP  $desc (already done)"; return 0; fi
    if [ ! -f "$run_dir/best.pt" ]; then
        FAILURES=$((FAILURES + 1)); log "FAILED $desc: $run_dir/best.pt not found"; return 1
    fi
    mkdir -p "$out"
    run "$desc" "$out/evaluate.log" python scripts/evaluate.py --run-dir "$run_dir" --out-dir "$out" "$@"
}

# train <description> <run name> <train_model.py args...>: skip if already done.
train() {
    local desc=$1 name=$2; shift 2
    local dir=results/runs/$name
    if [ -f "$dir/eval.json" ]; then log "SKIP  $desc (already done)"; return 0; fi
    mkdir -p "$dir"
    run "$desc" "$dir/train.log" python scripts/train_model.py --run-name "$name" "$@"
}

log "=== phase 2, batch 2 — commit $(git rev-parse --short HEAD), niceness $(nice) ==="

# 1) Experiment 2: bigger test graphs (k = 0.5 n), evaluation only.
for variant in "" "_knnmean"; do
    for m in "${MODELS[@]}"; do
        for s in "${SEEDS[@]}"; do
            r=${m}${variant}_seed${s}
            evaluate "exp2 k_frac 0.5: $r" "results/runs/$r" "$OUT/kfrac05/$r" \
                --k-frac 0.5 --batch-size 64 --test data/tsp50_test.npz data/tsp100_test.npz
        done
    done
done

# 2) Experiment 2b: BatchNorm with the statistics of each test batch, evaluation only.
for m in "${MODELS[@]}"; do
    for s in "${SEEDS[@]}"; do
        r=${m}_seed${s}
        evaluate "exp2b bn-batch-stats: $r" "results/runs/$r" "$OUT/bnbatch/$r" \
            --bn-batch-stats --test data/tsp20_test.npz data/tsp50_test.npz data/tsp100_test.npz
    done
done

# 3) Experiment 3: LayerNorm instead of BatchNorm (TSP-20, raw distances, k = 10).
for s in "${SEEDS[@]}"; do
    for m in "${MODELS[@]}"; do
        r=${m}_layernorm_seed${s}
        if train "exp3 train $r" "$r" --model "$m" --seed "$s" --norm layernorm \
                --test data/tsp20_test.npz data/tsp50_test.npz data/tsp100_test.npz; then
            evaluate "exp3 k_frac 0.5: $r" "results/runs/$r" "$OUT/kfrac05/$r" --k-frac 0.5 --batch-size 64 \
                --test data/tsp20_test.npz data/tsp50_test.npz data/tsp100_test.npz
        else
            log "SKIP  exp3 k_frac 0.5: $r (its training failed)"
        fi
    done
done

# 4) Control: train on TSP-50 (phase 1 configuration otherwise).
data_ok=1
for spec in "tsp50_train 12000 6" "tsp50_val 500 7"; do
    read -r name num seed <<< "$spec"
    if [ -f "data/$name.npz" ]; then log "SKIP  generate data/$name.npz (already exists)"; continue; fi
    run "generate data/$name.npz ($num instances, seed $seed)" "$OUT/logs/make_$name.log" \
        python scripts/make_dataset.py --n 50 --num "$num" --seed "$seed" --out "data/$name.npz" || data_ok=0
done
for s in "${SEEDS[@]}"; do
    for m in "${MODELS[@]}"; do
        r=${m}_train50_seed${s}
        if [ "$data_ok" = 1 ]; then
            train "control train $r" "$r" --model "$m" --seed "$s" \
                --train data/tsp50_train.npz --val data/tsp50_val.npz \
                --test data/tsp50_test.npz data/tsp100_test.npz
        else
            log "SKIP  control train $r (TSP-50 data generation failed)"
        fi
    done
done

# 5) Report (always runs: missing results are marked as such in the report).
if run "report" "$OUT/logs/report.log" \
        python scripts/phase2_batch2_report.py --out results/phase2_batch2_report.md; then
    cp results/phase2_batch2_report.md docs/phase2_batch2_report.md && log "copied to docs/phase2_batch2_report.md"
fi

log "failed steps: $FAILURES"
log "ALL DONE"
