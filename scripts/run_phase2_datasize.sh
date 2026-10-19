#!/usr/bin/env bash
# Phase 2, step 5: more training data (phase 1 configuration: BatchNorm, k = 10, raw distances).
#
#   0) generate data/tsp20_train100k.npz (100000 TSP-20 instances, seed 8); 25k and 50k are its first
#      25000 / 50000 instances (--num-train), so the training sets are nested
#   1) train, in order of importance (20 epochs each, evaluated on TSP-20/50/100 with k = 10):
#        a) gated 25k, 50k, 100k (seed 0)   b) gat (same)   c) gcn (same)
#        d) gated 100k seeds 1, 2           e) gat and gcn 100k seeds 1, 2
#      -> results/runs/<model>_data<size>_seed<s>/
#      each training is followed by its evaluation with k = 0.5 n
#      -> results/phase2_datasize/kfrac05/<model>_data<size>_seed<s>/
#   2) report  results/phase2_datasize_report.md + results/figs/phase2_datasize_gap.{png,pdf},
#              copied to docs/ (not committed)
# Progress goes to results/phase2_datasize.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
# A failed step does not stop the script: only the steps that depend on it are skipped.
# Steps whose output already exists are skipped, so the script can be relaunched safely.
#
# Launch (survives closing SSH):
#   tmux new-session -d -s phase2_datasize "bash scripts/run_phase2_datasize.sh"

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
OUT=results/phase2_datasize
DATA=data/tsp20_train100k.npz
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$OUT/kfrac05" "$OUT/logs"

LOG=results/phase2_datasize.log
TESTS=(data/tsp20_test.npz data/tsp50_test.npz data/tsp100_test.npz)
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

# train_and_eval <model> <size: 25k|50k|100k> <seed>: train (skip if done), then evaluate with k = 0.5 n.
train_and_eval() {
    local m=$1 size=$2 s=$3
    local num=$(( ${size%k} * 1000 )) r=${1}_data${2}_seed${3}
    local dir=results/runs/$r
    if [ "$data_ok" != 1 ]; then log "SKIP  train $r (no training data)"; return; fi
    if [ -f "$dir/eval.json" ]; then
        log "SKIP  train $r (already done)"
    else
        mkdir -p "$dir"
        if ! run "train $r ($num instances)" "$dir/train.log" python scripts/train_model.py --run-name "$r" \
                --model "$m" --seed "$s" --train "$DATA" --num-train "$num" --test "${TESTS[@]}"; then
            log "SKIP  k_frac 0.5: $r (its training failed)"
            return
        fi
    fi
    if [ -f "$OUT/kfrac05/$r/eval.json" ]; then log "SKIP  k_frac 0.5: $r (already done)"; return; fi
    mkdir -p "$OUT/kfrac05/$r"
    run "k_frac 0.5: $r" "$OUT/kfrac05/$r/evaluate.log" python scripts/evaluate.py --run-dir "$dir" \
        --out-dir "$OUT/kfrac05/$r" --k-frac 0.5 --batch-size 64 --test "${TESTS[@]}"
}

log "=== phase 2, step 5 (training-set size) — commit $(git rev-parse --short HEAD), niceness $(nice) ==="

# 0) Training data.
data_ok=1
if [ -f "$DATA" ]; then
    log "SKIP  generate $DATA (already exists)"
else
    run "generate $DATA (100000 instances, seed 8)" "$OUT/logs/make_tsp20_train100k.log" \
        python scripts/make_dataset.py --n 20 --num 100000 --seed 8 --out "$DATA" || data_ok=0
fi

# 1) Training, from the most to the least important run.
for m in gated gat gcn; do                      # a), b), c)
    for size in 25k 50k 100k; do train_and_eval "$m" "$size" 0; done
done
for s in 1 2; do train_and_eval gated 100k "$s"; done                 # d)
for m in gat gcn; do for s in 1 2; do train_and_eval "$m" 100k "$s"; done; done   # e)

# 2) Report (always runs: missing results are marked as such in the report).
if run "report" "$OUT/logs/report.log" \
        python scripts/phase2_datasize_report.py --out results/phase2_datasize_report.md; then
    mkdir -p docs/figs
    cp results/phase2_datasize_report.md docs/phase2_datasize_report.md \
        && cp results/figs/phase2_datasize_gap.png docs/figs/phase2_datasize_gap.png \
        && log "copied to docs/phase2_datasize_report.md (+ docs/figs/phase2_datasize_gap.png)"
fi

log "failed steps: $FAILURES"
log "ALL DONE"
