#!/usr/bin/env bash
# Phase 3, fast triage, follow-up:
#   1. the version-B learned policy on Fu-10k, one TAGCN seed at a time. In the main run the three seeds ran together
#      and hit CUDA out of memory (another project's Ollama held 12.6-13.2 GiB of the T4); the network head is now
#      applied in chunks of edges (fast_triage.edge_probs: peak 0.81 GiB instead of 1.56 at n = 10000).
#   2. diagnostic of the LKH time: in the main run the network policies spent ~22 % more LKH time than random for the
#      same number of windows. On the first 4 Fu-10k instances, three processes together: random (rng seed 0),
#      batched learned with torch threads 2 (as in every run) and with torch threads 1.
#   3. the report again.
# Log: results/phase3_fast_triage/b10k.log.
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
DATA=results/phase3_benchmark/data/tsp10000_test_concorde.npz
MODEL0=results/phase3_learned_triage/models/tagcn_k3_seed0/chosen.joblib
LOG=$OUT/b10k.log
FAILS=0
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
run() {
    local desc=$1 out=$2; shift 2
    log "START $desc"
    if "$@" > "$out" 2>&1; then log "OK    $desc"; else log "FAILED $desc (see $out)"; return 1; fi
}
log "=== follow-up: B learned on Fu-10k + LKH-time diagnostic — commit $(git rev-parse --short HEAD) ==="
for s in 0 1 2; do
    run "global_learned Fu-10k seed $s" $OUT/logs/e2_b_rerun2_global_learned_seed$s.log python -u \
        scripts/triage_ablation.py --data $DATA --score global_learned --seeds $s \
        --model $OUT/global_models/tagcn_k3_seed$s/chosen.joblib --out-root $OUT/e2_b || FAILS=$((FAILS + 1))
done
PIDS=()
run "diag random" $OUT/logs/diag_random.log python -u scripts/triage_ablation.py --data $DATA --score random \
    --seeds 0 --max-instances 4 --out-root $OUT/diag & PIDS+=($!)
for t in 1 2; do
    run "diag fast_learned threads $t" $OUT/logs/diag_learned_t$t.log python -u scripts/triage_ablation.py --data $DATA \
        --score fast_learned --seeds 0 --model $MODEL0 --max-instances 4 --torch-threads $t \
        --folder-suffix _threads$t --out-root $OUT/diag & PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p || FAILS=$((FAILS + 1)); done
run "report" $OUT/logs/report3.log python -u scripts/phase3_fast_triage_report.py || FAILS=$((FAILS + 1))
log "ALL DONE follow-up ($FAILS failures)"
