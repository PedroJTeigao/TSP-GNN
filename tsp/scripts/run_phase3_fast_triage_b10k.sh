#!/usr/bin/env bash
# Phase 3, fast triage: rerun of the version-B learned policy on Fu-10k, one TAGCN seed at a time. In the main run
# (scripts/run_phase3_fast_triage.sh) the three seeds ran together and hit CUDA out of memory: two Ollama servers of
# another project held 12.6 GiB of the T4, and each whole-instance pass at n = 10000 needs ~1.6 GiB. Waits for the
# main run, then regenerates the report. Log: results/phase3_fast_triage/b10k.log.
#
# Launch from tsp/:
#   tmux new-session -d -s fast_b10k "nohup bash scripts/run_phase3_fast_triage_b10k.sh > results/phase3_fast_triage/b10k.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi
source ../env.sh
export OMP_NUM_THREADS=1 GIT_CONFIG_GLOBAL=/dev/null
OUT=results/phase3_fast_triage
LOG=$OUT/b10k.log
FAILS=0
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "ALL DONE fast triage" $OUT/run.log 2>/dev/null; do sleep 30; done
log "=== B learned on Fu-10k, one seed at a time — commit $(git rev-parse --short HEAD) ==="
for s in 0 1 2; do
    log "START global_learned seed $s"
    if python -u scripts/triage_ablation.py --data results/phase3_benchmark/data/tsp10000_test_concorde.npz \
        --score global_learned --seeds $s --model $OUT/global_models/tagcn_k3_seed$s/chosen.joblib \
        --out-root $OUT/e2_b > $OUT/logs/e2_b_rerun_global_learned_seed$s.log 2>&1; then
        log "OK    global_learned seed $s"
    else
        log "FAILED global_learned seed $s"; FAILS=$((FAILS + 1))
    fi
done
python -u scripts/phase3_fast_triage_report.py > $OUT/logs/report2.log 2>&1 || FAILS=$((FAILS + 1))
log "ALL DONE B 10k rerun ($FAILS failures)"
