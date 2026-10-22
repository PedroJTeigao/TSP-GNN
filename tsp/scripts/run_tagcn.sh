#!/usr/bin/env bash
# TAGCN (Du et al. 2017) as the fourth encoder: phase 1 and phase 2 protocol, K = 2 and K = 3, seeds 0, 1, 2.
#
#   1) phase 1: train tagcn K=2 (3 layers) and K=3 (2 layers), best val gap, evaluated on TSP-20/50 (k = 10)
#        -> results/runs/tagcn_k<K>_seed<s>/
#      then best.pt on TSP-50/100 with k = 0.5 n -> results/runs/tagcn_k<K>_seed<s>/kfrac05/
#   2) beam search (CPU, three lanes in the background, one per seed, while step 3 trains on the GPU):
#      best.pt, greedy / B=10 / B=100, ± 2-opt, TSP-20/50/100, k = 0.5 n -> results/runs/tagcn_k<K>_seed<s>/beam/
#   3) LayerNorm retraining (--extra-checkpoints) -> results/runs/tagcn_k<K>_layernorm_seed<s>/
#      then last.pt on TSP-20/50/100 with k = 0.5 n -> results/runs/tagcn_k<K>_layernorm_seed<s>/last_kfrac05/
#   4) report results/tagcn_report.md (+ results/figs/tagcn_curves.png), copied to docs/
# Progress goes to results/tagcn.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
# A failed step does not stop the script: only the steps that depend on it are skipped.
# Steps whose output already exists are skipped, so the script can be relaunched safely.
#
# Launch (survives closing SSH):
#   tmux new-session -d -s tagcn "nohup bash scripts/run_tagcn.sh > results/tagcn.nohup 2>&1"

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
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" results/figs

LOG=results/tagcn.log
KS=(2 3)
SEEDS=(0 1 2)
TESTS=(data/tsp20_test.npz data/tsp50_test.npz data/tsp100_test.npz)
FAILDIR=results/runs/tagcn_logs/failures   # one file per failure (the beam lanes are separate processes)
mkdir -p "$FAILDIR"

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
    touch "$FAILDIR/$(echo "$desc" | tr -c 'A-Za-z0-9_=.-' '_')"
    log "FAILED $desc (exit code $code). Last lines of $out:"
    tail -n 15 "$out" | sed 's/^/    /' | tee -a "$LOG"
    return 1
}

# train <run name> <train_model.py args...>: skip if already done.
train() {
    local name=$1; shift
    local dir=results/runs/$name
    if [ -f "$dir/eval.json" ]; then log "SKIP  train $name (already done)"; return 0; fi
    mkdir -p "$dir"
    run "train $name" "$dir/train.log" python scripts/train_model.py --model tagcn --run-name "$name" "$@"
}

# evaluate <run name> <checkpoint> <subfolder> <evaluate.py args...>: skip if already done.
evaluate() {
    local name=$1 ckpt=$2 sub=$3; shift 3
    local dir=results/runs/$name out=results/runs/$name/$sub
    if [ -f "$out/eval.json" ]; then log "SKIP  evaluate $name/$sub (already done)"; return 0; fi
    if [ ! -f "$dir/$ckpt" ]; then
        touch "$FAILDIR/eval.$name.$sub"; log "FAILED evaluate $name/$sub: $dir/$ckpt not found"; return 1
    fi
    mkdir -p "$out"
    run "evaluate $name/$sub ($ckpt)" "$out/evaluate.log" python scripts/evaluate.py --run-dir "$dir" \
        --checkpoint "$ckpt" --out-dir "$out" "$@"
}

# beam_lane <seed>: beam search on both K of one seed (CPU; 2 PyTorch threads per lane).
beam_lane() {
    local s=$1 K name out
    for K in "${KS[@]}"; do
        name=tagcn_k${K}_seed${s}; out=results/runs/$name/beam
        if [ -f "$out/eval.json" ]; then log "SKIP  beam $name (already done)"; continue; fi
        if [ ! -f "results/runs/$name/best.pt" ]; then
            touch "$FAILDIR/beam.$name"; log "FAILED beam $name: best.pt not found (training failed)"; continue
        fi
        mkdir -p "$out"
        OMP_NUM_THREADS=2 run "beam $name" "$out/beam_eval.log" python scripts/beam_eval.py \
            --run-dir "results/runs/$name" --k-frac 0.5 --beam-widths 10 100 --batch-size 64 \
            --test "${TESTS[@]}" --out-dir "$out"
    done
}

log "=== TAGCN — commit $(git rev-parse --short HEAD), niceness $(nice) ==="

# 1) Phase 1 training + k = 0.5 n evaluation.
for s in "${SEEDS[@]}"; do
    for K in "${KS[@]}"; do
        r=tagcn_k${K}_seed${s}
        if train "$r" --tag-k "$K" --seed "$s"; then
            evaluate "$r" best.pt kfrac05 --k-frac 0.5 --batch-size 64 --test data/tsp50_test.npz data/tsp100_test.npz
        else
            log "SKIP  evaluations of $r (its training failed)"
        fi
    done
done

# 2) Beam search in the background (three lanes), while step 3 uses the GPU.
LANES=()
for s in "${SEEDS[@]}"; do
    beam_lane "$s" &
    LANES+=($!)
done

# 3) LayerNorm retraining + last.pt with k = 0.5 n.
for s in "${SEEDS[@]}"; do
    for K in "${KS[@]}"; do
        r=tagcn_k${K}_layernorm_seed${s}
        if train "$r" --tag-k "$K" --seed "$s" --norm layernorm --extra-checkpoints --test "${TESTS[@]}"; then
            evaluate "$r" last.pt last_kfrac05 --k-frac 0.5 --batch-size 64 --test "${TESTS[@]}"
        else
            log "SKIP  evaluations of $r (its training failed)"
        fi
    done
done

log "waiting for the beam search lanes"
wait "${LANES[@]}"

# 4) Report (always runs: missing results are marked as such in the report).
if run "TAGCN report" results/runs/tagcn_logs/report.log python scripts/tagcn_report.py; then
    mkdir -p docs/figs
    cp results/tagcn_report.md docs/tagcn_report.md \
        && cp results/figs/tagcn_curves.png docs/figs/tagcn_curves.png \
        && log "copied to docs/tagcn_report.md (+ docs/figs/tagcn_curves.png)"
fi

FAILURES=$(find "$FAILDIR" -type f | wc -l)
log "failed steps: $FAILURES"
log "ALL DONE ($FAILURES failures)"
