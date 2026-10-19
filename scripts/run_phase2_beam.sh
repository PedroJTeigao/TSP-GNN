#!/usr/bin/env bash
# Phase 2, beam search decoding: no training, only new evaluations of existing checkpoints.
#
#   For each checkpoint (3 seeds each), scripts/beam_eval.py decodes TSP-20/50/100 (k = 0.5 n) with greedy,
#   beam 10 and beam 100, each without and with 2-opt:
#     phase1    results/runs/<gcn|gat|gated>_seed<s>/best.pt
#     ln_last   results/runs/<gcn|gated>_layernorm_ckpt_seed<s>/last.pt
#     data100k  results/runs/gated_data100k_seed<s>/best.pt
#   -> results/phase2_beam/<config>/<run>/   (eval.json + eval_per_instance.npz)
#   Then the report results/phase2_beam_report.md, copied to docs/.
# Three lanes run in parallel, one per seed (6 checkpoints each); decoding is CPU-bound (NumPy).
# Progress goes to results/phase2_beam.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
# A failed evaluation does not stop the script. Evaluations whose eval.json exists are skipped,
# so the script can be relaunched safely.
#
# Launch (survives closing SSH):
#   nohup bash scripts/run_phase2_beam.sh > results/phase2_beam.nohup 2>&1 &

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
# Three lanes in parallel: each one uses a few CPU threads for PyTorch (decoding itself is single-threaded).
export OMP_NUM_THREADS=2
OUT=results/phase2_beam
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$OUT/logs"

LOG=results/phase2_beam.log
SEEDS=(0 1 2)
TESTS=(data/tsp20_test.npz data/tsp50_test.npz data/tsp100_test.npz)
FAILDIR=$OUT/logs/failures   # one file per failure (the lanes are separate processes)
mkdir -p "$FAILDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# evaluate <config> <run name> <checkpoint file>: skip if already done.
evaluate() {
    local config=$1 run=$2 ckpt=$3
    local desc="$config: $run/$ckpt" out=$OUT/$config/$run
    if [ -f "$out/eval.json" ]; then log "SKIP  $desc (already done)"; return 0; fi
    if [ ! -f "results/runs/$run/$ckpt" ]; then
        log "FAILED $desc: results/runs/$run/$ckpt not found"; touch "$FAILDIR/$config.$run"; return 1
    fi
    mkdir -p "$out"
    log "START $desc"
    local start=$SECONDS
    python scripts/beam_eval.py --run-dir "results/runs/$run" --checkpoint "$ckpt" --k-frac 0.5 \
        --beam-widths 10 100 --batch-size 64 --test "${TESTS[@]}" --out-dir "$out" > "$out/beam_eval.log" 2>&1
    local code=$?
    if [ "$code" -eq 0 ]; then
        log "OK    $desc ($(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s)"
        return 0
    fi
    touch "$FAILDIR/$config.$run"
    log "FAILED $desc (exit code $code). Last lines of $out/beam_eval.log:"
    tail -n 15 "$out/beam_eval.log" | sed 's/^/    /' | tee -a "$LOG"
    return 1
}

# One lane = the 6 checkpoints of one seed.
lane() {
    local s=$1
    for m in gcn gat gated; do evaluate phase1 "${m}_seed${s}" best.pt; done
    for m in gcn gated; do evaluate ln_last "${m}_layernorm_ckpt_seed${s}" last.pt; done
    evaluate data100k "gated_data100k_seed${s}" best.pt
}

log "=== phase 2 beam search — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for s in "${SEEDS[@]}"; do
    lane "$s" &
done
wait

# Report (always runs: missing results are marked as such in the report).
log "START beam report"
if python scripts/phase2_beam_report.py > "$OUT/logs/report.log" 2>&1; then
    cp results/phase2_beam_report.md docs/phase2_beam_report.md
    log "OK    beam report (results/phase2_beam_report.md, copied to docs/)"
else
    touch "$FAILDIR/report"
    log "FAILED beam report. Last lines of $OUT/logs/report.log:"
    tail -n 15 "$OUT/logs/report.log" | sed 's/^/    /' | tee -a "$LOG"
fi

FAILURES=$(find "$FAILDIR" -type f | wc -l)
log "ALL DONE ($FAILURES failures)"
