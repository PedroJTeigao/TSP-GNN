#!/usr/bin/env bash
# Phase 3, cheap window scoring for the triage (tsp_gnn.fast_triage):
#   1. version B: window features from the global heatmap for the learned-triage logs (3 seeds in parallel), and the
#      learned signal retrained on them (same protocol as scripts/learned_triage_train.py);
#   2. equivalence and scoring time on a sample (2 instances of tsp1000 and Fu-10k, 3 seeds): A and B against the
#      per-window scores; B goes on only if its Spearman correlation is >= 0.5 (scripts/fast_triage_decide.py);
#   3. policies (E2 protocol) on tsp1000 and Fu-10k: random + batched disagreement + batched learned (3 seeds), run
#      together (7 processes, one LKH call at a time each); then the B scores that passed, with random again
#      (at most 7 processes);
#   4. the report.
# Outputs: results/phase3_fast_triage/, results/phase3_fast_triage_report.md (+ docs/ copy).
# Log: results/phase3_fast_triage/run.log.
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s fast_triage "nohup bash scripts/run_phase3_fast_triage.sh > results/phase3_fast_triage/run.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase3_fast_triage
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
LOG=$OUT/run.log
FAILS=0
DATA=results/phase3_benchmark/data
MODELS=results/phase3_learned_triage/models

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

waitall() { for p in "$@"; do wait $p || FAILS=$((FAILS + 1)); done; }

log "=== fast triage — commit $(git rev-parse --short HEAD), niceness $(nice) ==="

# 1. version B features and retraining
global_features() {
    local s=$1 f=0
    for name in tsp200_test tsp1000_test tsp10000_test; do
        run "global features $name seed $s" "$LOGDIR/gf_${name}_seed$s.log" python -u \
            scripts/fast_triage_global_features.py --data data/$name.npz --seed $s || f=$((f + 1))
    done
    for name in tsp1000_test_concorde tsp10000_test_concorde; do
        run "global features $name seed $s" "$LOGDIR/gf_${name}_seed$s.log" python -u \
            scripts/fast_triage_global_features.py --data $DATA/$name.npz --seed $s || f=$((f + 1))
    done
    return $f
}
PIDS=(); for s in 0 1 2; do global_features $s & PIDS+=($!); done; waitall "${PIDS[@]}"
for s in 0 1 2; do
    OMP_NUM_THREADS=4 run "train on global features seed $s" "$LOGDIR/train_global_seed$s.log" python -u \
        scripts/learned_triage_train.py --seed $s --features-root $OUT/global_features --out-root $OUT/global_models \
        || FAILS=$((FAILS + 1))
done

# 2. equivalence and scoring time (one process at a time: clean times)
for name in tsp1000_test tsp10000_test_concorde; do
    run "check $name" "$LOGDIR/check_$name.log" python -u scripts/fast_triage_check.py --data $DATA/$name.npz \
        || FAILS=$((FAILS + 1))
done
run "B decision" "$LOGDIR/b_decision.log" python -u scripts/fast_triage_decide.py || FAILS=$((FAILS + 1))
B_DIS=$(grep -c "^disagreement=yes" "$LOGDIR/b_decision.log")
B_LRN=$(grep -c "^learned=yes" "$LOGDIR/b_decision.log")
log "B goes on: disagreement $B_DIS, learned $B_LRN (1 = yes)"

# 3. policies
policy() {   # name score folder-root [seed model]
    local name=$1 score=$2 root=$3 s=${4:-} model=${5:-}
    local args=(--data $DATA/$name.npz --score $score --out-root $root)
    [ -n "$s" ] && args+=(--seeds $s)
    [ -n "$model" ] && args+=(--model $model)
    run "e2 $name $score${s:+ seed $s} ($root)" "$LOGDIR/e2_$(basename $root)_${name}_${score}${s:+_seed$s}.log" \
        python -u scripts/triage_ablation.py "${args[@]}"
}
for name in tsp1000_test tsp10000_test_concorde; do
    PIDS=()
    policy $name random $OUT/e2 & PIDS+=($!)
    for s in 0 1 2; do
        policy $name fast_disagreement $OUT/e2 $s & PIDS+=($!)
        policy $name fast_learned $OUT/e2 $s $MODELS/tagcn_k3_seed$s/chosen.joblib & PIDS+=($!)
    done
    waitall "${PIDS[@]}"
done
if [ "$B_DIS" = 1 ] || [ "$B_LRN" = 1 ]; then
    for name in tsp1000_test tsp10000_test_concorde; do
        PIDS=()
        policy $name random $OUT/e2_b & PIDS+=($!)
        for s in 0 1 2; do
            [ "$B_DIS" = 1 ] && { policy $name global_disagreement $OUT/e2_b $s & PIDS+=($!); }
            [ "$B_LRN" = 1 ] && { policy $name global_learned $OUT/e2_b $s $OUT/global_models/tagcn_k3_seed$s/chosen.joblib & PIDS+=($!); }
        done
        waitall "${PIDS[@]}"
    done
else
    log "SKIP  B policies (Spearman < 0.5)"
fi

# 4. report
run "report" "$LOGDIR/report.log" python -u scripts/phase3_fast_triage_report.py || FAILS=$((FAILS + 1))
log "ALL DONE fast triage ($FAILS failures)"
