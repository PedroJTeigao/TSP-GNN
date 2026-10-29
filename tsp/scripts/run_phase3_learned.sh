#!/usr/bin/env bash
# Phase 3, learned triage, steps 2-4: train the window signal for the 3 TAGCN K=3 seeds (sklearn, CPU), then the
# ranked policies of E2 on the test sets (tsp1000, tsp10000, Fu-10k), with random / disagreement / learned / oracle
# all run together on each set (8 processes, one LKH call at a time each: at most 8 LKH processes; GPU ~0.25 GiB
# per network process), then the report.
# Outputs: results/phase3_learned_triage/{models,e2}/, results/phase3_learned_triage_report.md (+ docs/ copy).
# Log: results/phase3_learned_triage/run.log.
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s learned "nohup bash scripts/run_phase3_learned.sh > results/phase3_learned_triage/run.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
OUT=results/phase3_learned_triage
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
LOG=$OUT/run.log
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

log "=== learned triage, train + policies + report — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for s in 0 1 2; do
    OMP_NUM_THREADS=4 run "train seed $s" "$LOGDIR/train_seed$s.log" python -u scripts/learned_triage_train.py \
        --seed $s || FAILS=$((FAILS + 1))
done

export OMP_NUM_THREADS=1
for name in tsp1000_test tsp10000_test tsp10000_test_concorde; do
    data=results/phase3_benchmark/data/$name.npz
    PIDS=()
    run "e2 $name random" "$LOGDIR/e2_${name}_random.log" python -u scripts/triage_ablation.py --data $data \
        --score random --out-root $OUT/e2 & PIDS+=($!)
    run "e2 $name oracle" "$LOGDIR/e2_${name}_oracle.log" python -u scripts/triage_ablation.py --data $data \
        --score oracle --out-root $OUT/e2 & PIDS+=($!)
    for s in 0 1 2; do
        run "e2 $name disagreement seed $s" "$LOGDIR/e2_${name}_disagreement_seed$s.log" \
            python -u scripts/triage_ablation.py --data $data --score disagreement --seeds $s --out-root $OUT/e2 &
        PIDS+=($!)
        run "e2 $name learned seed $s" "$LOGDIR/e2_${name}_learned_seed$s.log" \
            python -u scripts/triage_ablation.py --data $data --score learned --seeds $s \
            --model $OUT/models/tagcn_k3_seed$s/chosen.joblib --out-root $OUT/e2 &
        PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait $p || FAILS=$((FAILS + 1)); done
done

run "report" "$LOGDIR/report.log" python -u scripts/phase3_learned_triage_report.py || FAILS=$((FAILS + 1))
log "ALL DONE learned triage ($FAILS failures)"
