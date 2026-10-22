#!/usr/bin/env bash
# TAGCN BatchNorm reruns with --extra-checkpoints, to evaluate the last epoch (last.pt) of the phase 1 protocol.
#
# The first TAGCN batch (scripts/run_tagcn.sh) trained the BatchNorm runs without --extra-checkpoints, so
# only best.pt exists there. Here the same 6 runs (K = 2 and 3, seeds 0, 1, 2) are trained again into new
# folders; the scatter sums on the GPU are not deterministic, so these runs are close to, but not bit-for-bit
# identical with, the first ones (the report compares their best.pt on the same test instances).
#   1) train -> results/runs/tagcn_k<K>_ckpt_seed<s>/  (best.pt evaluated on TSP-20/50 with k = 10, as in phase 1)
#   2) last.pt: greedy and beam 100, ± 2-opt, TSP-20/50/100, k = 0.5 n
#        -> results/runs/tagcn_k<K>_ckpt_seed<s>/last_beam/   (one background lane per seed, CPU)
#   3) report results/tagcn_report.md regenerated and copied to docs/
# Progress goes to results/tagcn_ckpt.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
#
# Launch (survives closing SSH):
#   tmux new-session -d -s tagcn_ckpt "nohup bash scripts/run_tagcn_ckpt.sh > results/tagcn_ckpt.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" results/figs

LOG=results/tagcn_ckpt.log
KS=(2 3)
SEEDS=(0 1 2)
TESTS=(data/tsp20_test.npz data/tsp50_test.npz data/tsp100_test.npz)
FAILDIR=results/runs/tagcn_logs/ckpt_failures
mkdir -p "$FAILDIR"

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

# beam_lane <seed>: last.pt of both K of one seed, greedy + beam 100 (CPU, 2 PyTorch threads).
beam_lane() {
    local s=$1 K name out
    for K in "${KS[@]}"; do
        name=tagcn_k${K}_ckpt_seed${s}; out=results/runs/$name/last_beam
        if [ -f "$out/eval.json" ]; then log "SKIP  last.pt beam $name (already done)"; continue; fi
        if [ ! -f "results/runs/$name/last.pt" ]; then
            touch "$FAILDIR/beam.$name"; log "FAILED last.pt beam $name: last.pt not found (training failed)"; continue
        fi
        mkdir -p "$out"
        OMP_NUM_THREADS=2 run "last.pt greedy/beam100 $name" "$out/beam_eval.log" python scripts/beam_eval.py \
            --run-dir "results/runs/$name" --checkpoint last.pt --k-frac 0.5 --beam-widths 100 --batch-size 64 \
            --test "${TESTS[@]}" --out-dir "$out"
    done
}

log "=== TAGCN BatchNorm reruns with last.pt — commit $(git rev-parse --short HEAD), niceness $(nice) ==="

LANES=()
for s in "${SEEDS[@]}"; do
    for K in "${KS[@]}"; do
        r=tagcn_k${K}_ckpt_seed${s}
        if [ -f "results/runs/$r/eval.json" ]; then log "SKIP  train $r (already done)"; continue; fi
        mkdir -p "results/runs/$r"
        run "train $r" "results/runs/$r/train.log" python scripts/train_model.py --model tagcn --tag-k "$K" \
            --seed "$s" --extra-checkpoints --run-name "$r"
    done
    beam_lane "$s" &      # this seed's beam search runs on the CPU while the next seed trains
    LANES+=($!)
done
log "waiting for the beam search lanes"
wait "${LANES[@]}"

if run "TAGCN report" results/runs/tagcn_logs/ckpt_report.log python scripts/tagcn_report.py; then
    cp results/tagcn_report.md docs/tagcn_report.md && cp results/figs/tagcn_curves.png docs/figs/tagcn_curves.png \
        && log "copied to docs/tagcn_report.md (+ docs/figs/tagcn_curves.png)"
fi

FAILURES=$(find "$FAILDIR" -type f | wc -l)
log "failed steps: $FAILURES"
log "ALL DONE ($FAILURES failures)"
