#!/usr/bin/env bash
# Phase 2 checks (points 1 to 4), then step 5 (training-set size, scripts/run_phase2_datasize.sh).
#
#   1) LayerNorm reruns with three checkpoints (best val gap / best val loss / last epoch)
#        -> results/runs/<model>_layernorm_ckpt_seed<s>/   (best.pt evaluated with k = 10 at the end of training)
#        -> results/phase2_checks/ln/<gap|loss|last>_<k10|kfrac05>/<run>/   (other checkpoints and k = 0.5 n)
#   2) TSP-50 control trained with k = 0.5 n -> results/runs/<model>_train50kfrac_seed<s>/ (TSP-20/50/100)
#   3) batch 2 TSP-50 control evaluated on TSP-20 -> results/phase2_checks/train50_tsp20/<run>/
#   4) report  results/phase2_checks_report.md (+ results/figs/phase2_checks_ln_curves.png), copied to docs/
#   5) step 5: waits until data/tsp20_train100k.npz is no longer being generated, then runs
#      scripts/run_phase2_datasize.sh (its own log: results/phase2_datasize.log)
# Progress goes to results/phase2_checks.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
# A failed step does not stop the script: only the steps that depend on it are skipped.
# Steps whose output already exists are skipped, so the script can be relaunched safely.
#
# Launch (survives closing SSH):
#   tmux new-session -d -s phase2_checks "bash scripts/run_phase2_checks.sh"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

# Low CPU priority for the whole batch (the children inherit it). Re-exec only once.
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
# $HOME is on AFS, whose tokens may expire after logout: keep caches and git config away from it.
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
OUT=results/phase2_checks
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$OUT/logs"

LOG=results/phase2_checks.log
MODELS=(gcn gat gated)
SEEDS=(0 1 2)
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

# evaluate <description> <run dir> <checkpoint file> <out dir> <extra evaluate.py args...>: skip if already done.
evaluate() {
    local desc=$1 run_dir=$2 ckpt=$3 out=$4; shift 4
    if [ -f "$out/eval.json" ]; then log "SKIP  $desc (already done)"; return 0; fi
    if [ ! -f "$run_dir/$ckpt" ]; then
        FAILURES=$((FAILURES + 1)); log "FAILED $desc: $run_dir/$ckpt not found"; return 1
    fi
    mkdir -p "$out"
    run "$desc" "$out/evaluate.log" python scripts/evaluate.py --run-dir "$run_dir" --checkpoint "$ckpt" \
        --out-dir "$out" "$@"
}

# train <description> <run name> <train_model.py args...>: skip if already done.
train() {
    local desc=$1 name=$2; shift 2
    local dir=results/runs/$name
    if [ -f "$dir/eval.json" ]; then log "SKIP  $desc (already done)"; return 0; fi
    mkdir -p "$dir"
    run "$desc" "$dir/train.log" python scripts/train_model.py --run-name "$name" "$@"
}

log "=== phase 2 checks (points 1-4) + step 5 — commit $(git rev-parse --short HEAD), niceness $(nice) ==="

# 1) LayerNorm reruns with three checkpoints.
for s in "${SEEDS[@]}"; do
    for m in "${MODELS[@]}"; do
        r=${m}_layernorm_ckpt_seed${s}
        if ! train "check1 train $r" "$r" --model "$m" --seed "$s" --norm layernorm --extra-checkpoints \
                --test "${TESTS[@]}"; then
            log "SKIP  check1 evaluations of $r (its training failed)"
            continue
        fi
        for sel in gap loss last; do
            case $sel in gap) ckpt=best.pt ;; loss) ckpt=best_loss.pt ;; last) ckpt=last.pt ;; esac
            if [ "$sel" != gap ]; then  # best.pt with k = 10 was evaluated at the end of training
                evaluate "check1 $sel k=10: $r" "results/runs/$r" "$ckpt" "$OUT/ln/${sel}_k10/$r" --test "${TESTS[@]}"
            fi
            evaluate "check1 $sel k_frac 0.5: $r" "results/runs/$r" "$ckpt" "$OUT/ln/${sel}_kfrac05/$r" \
                --k-frac 0.5 --batch-size 64 --test "${TESTS[@]}"
        done
    done
done

# 2) TSP-50 control trained (and tested) with k = 0.5 n.
for s in "${SEEDS[@]}"; do
    for m in "${MODELS[@]}"; do
        r=${m}_train50kfrac_seed${s}
        train "check2 train $r" "$r" --model "$m" --seed "$s" --k-frac 0.5 \
            --train data/tsp50_train.npz --val data/tsp50_val.npz --test "${TESTS[@]}"
    done
done

# 3) Batch 2 TSP-50 control on TSP-20.
for m in "${MODELS[@]}"; do
    for s in "${SEEDS[@]}"; do
        r=${m}_train50_seed${s}
        evaluate "check3 TSP-20: $r" "results/runs/$r" best.pt "$OUT/train50_tsp20/$r" --test data/tsp20_test.npz
    done
done

# 4) Report of points 1-4 (always runs: missing results are marked as such in the report).
if run "checks report" "$OUT/logs/report.log" \
        python scripts/phase2_checks_report.py --out results/phase2_checks_report.md; then
    mkdir -p docs/figs
    cp results/phase2_checks_report.md docs/phase2_checks_report.md \
        && cp results/figs/phase2_checks_ln_curves.png docs/figs/phase2_checks_ln_curves.png \
        && log "copied to docs/phase2_checks_report.md (+ docs/figs/phase2_checks_ln_curves.png)"
fi

# 5) Step 5 (training-set size). Wait while the 100k training set is still being generated elsewhere.
while pgrep -f "make_dataset.py .*tsp20_train100k" > /dev/null; do
    log "WAIT  data/tsp20_train100k.npz is still being generated"
    sleep 300
done
run "step 5 (scripts/run_phase2_datasize.sh, progress in results/phase2_datasize.log)" "$OUT/logs/step5.log" \
    bash scripts/run_phase2_datasize.sh

log "failed steps: $FAILURES (step 5 counts its own failures in results/phase2_datasize.log)"
log "ALL DONE"
