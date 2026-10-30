#!/usr/bin/env bash
# Phase 3, triage with larger windows (w = 50, 100; w = 20 completed):
#   1. LKH window logs for w = 50 and 100 on tsp200, Fu-1k, tsp1000, Fu-10k (8 processes, CPU), then the features per
#      TAGCN seed (2 GPU processes at a time: another project holds most of the T4);
#   2. the learned signal retrained for each w (train tsp200 + Fu-1k, test tsp1000 and Fu-10k) and its AP;
#   3a. in parallel (gaps only): the oracle at w = 50 (and w = 20, top 10 %) on the first 8 Fu-10k instances, and the
#       w = 50 policies on tsp1000 (CPU jobs together, GPU jobs 2 at a time);
#   3b. one process at a time on the first 8 Fu-10k instances (machine load recorded before / after each run):
#       w = 50 random (3 rng seeds) / batched disagreement / batched learned (3 TAGCN seeds), top 10 / 20 / 50 %, and
#       LKH on every window; w = 20 top 10 % and LKH on every window. w = 100 is left out of the policies;
#   4. the report.
# Outputs: results/phase3_triage_wsize/{log,features,models,par,seq}/, results/phase3_triage_wsize_report.md (+docs).
# Log: results/phase3_triage_wsize/run.log.
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s wsize "nohup bash scripts/run_phase3_triage_wsize.sh > results/phase3_triage_wsize/run.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi
source ../env.sh
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase3_triage_wsize
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR" "$OUT/seq"
LOG=$OUT/run.log
LOAD=$OUT/seq/load.tsv
FAILS=0
BENCH=results/phase3_benchmark/data
FU10K=$BENCH/tsp10000_test_concorde.npz
T1K=$BENCH/tsp1000_test.npz
declare -A DATA=([tsp200_test]=data/tsp200_test.npz [tsp1000_test_concorde]=$BENCH/tsp1000_test_concorde.npz
                 [tsp1000_test]=$T1K [tsp10000_test_concorde]=$FU10K)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
run() {
    local desc=$1 out=$2; shift 2
    log "START $desc"
    local start=$SECONDS
    if "$@" > "$out" 2>&1; then
        log "OK    $desc ($(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s)"
    else
        log "FAILED $desc (see $out)"; return 1
    fi
}
waitall() { for p in "$@"; do wait $p || FAILS=$((FAILS + 1)); done; }
load() {
    local l g
    l=$(cut -d' ' -f1-3 /proc/loadavg)
    g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
    printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$(echo $l | tr ' ' '\t')" "$g" >> $LOAD
}
pol() {   # mode-root label args...
    local root=$1 label=$2; shift 2
    run "$label" "$LOGDIR/$(echo $label | tr ' /' '__').log" python -u scripts/triage_ablation.py --out-root $root "$@"
}
seq_pol() {
    local root=$1 label=$2; shift 2
    load before "$label"
    pol $root "$label" "$@" || FAILS=$((FAILS + 1))
    load after "$label"
}

log "=== triage with larger windows — commit $(git rev-parse --short HEAD), niceness $(nice) ==="

# 1. LKH logs (CPU), then features (GPU, 2 at a time)
PIDS=()
for w in 50 100; do
    for name in "${!DATA[@]}"; do
        run "lkh log $name w$w" "$LOGDIR/lkh_${name}_w$w.log" python -u scripts/wsize_log.py --part lkh \
            --data ${DATA[$name]} --width $w & PIDS+=($!)
    done
done
waitall "${PIDS[@]}"
for w in 50 100; do
    for name in "${!DATA[@]}"; do
        PIDS=()
        for s in 0 1; do
            run "features $name w$w seed $s" "$LOGDIR/feat_${name}_w${w}_seed$s.log" python -u scripts/wsize_log.py \
                --part features --data ${DATA[$name]} --width $w --seed $s & PIDS+=($!)
        done
        waitall "${PIDS[@]}"
        run "features $name w$w seed 2" "$LOGDIR/feat_${name}_w${w}_seed2.log" python -u scripts/wsize_log.py \
            --part features --data ${DATA[$name]} --width $w --seed 2 || FAILS=$((FAILS + 1))
    done
done

# 2. learned signal per w
for w in 50 100; do
    for s in 0 1 2; do
        OMP_NUM_THREADS=4 run "train w$w seed $s" "$LOGDIR/train_w${w}_seed$s.log" python -u \
            scripts/learned_triage_train.py --seed $s --features-root $OUT/features/w$w --out-root $OUT/models/w$w \
            --test-sets tsp1000_test tsp10000_test_concorde || FAILS=$((FAILS + 1))
    done
done

# 3a. parallel: oracle on Fu-10k (8 instances), tsp1000 at w = 50
PIDS=()
for x in 10 20 50; do
    pol $OUT/par/w50 "par oracle Fu-10k w50 top$x" --data $FU10K --score oracle --width 50 --fracs $x \
        --max-instances 8 --folder-suffix _top$x & PIDS+=($!)
done
pol $OUT/par/w20 "par oracle Fu-10k w20 top10" --data $FU10K --score oracle --width 20 --fracs 10 --max-instances 8 \
    --folder-suffix _top10 & PIDS+=($!)
pol $OUT/par/w50 "par oracle tsp1000 w50" --data $T1K --score oracle --width 50 --fracs 10 20 50 & PIDS+=($!)
pol $OUT/par/w50 "par random tsp1000 w50" --data $T1K --score random --seeds 0 1 2 --width 50 --fracs 10 20 50 & PIDS+=($!)
pol $OUT/par/w50 "par all tsp1000 w50" --data $T1K --score random --seeds 0 --width 50 --fracs 100 \
    --folder-suffix _all & PIDS+=($!)
waitall "${PIDS[@]}"
for s in 0 1 2; do
    PIDS=()
    pol $OUT/par/w50 "par disagreement tsp1000 w50 seed $s" --data $T1K --score fast_disagreement --seeds $s \
        --width 50 --fracs 10 20 50 & PIDS+=($!)
    pol $OUT/par/w50 "par learned tsp1000 w50 seed $s" --data $T1K --score fast_learned --seeds $s --width 50 \
        --fracs 10 20 50 --model $OUT/models/w50/tagcn_k3_seed$s/chosen.joblib & PIDS+=($!)
    waitall "${PIDS[@]}"
done

# 3b. one process at a time on the first 8 Fu-10k instances
[ -f $LOAD ] || printf "when\trun\tload1\tload5\tload15\tgpu_other_mib\n" > $LOAD
S50="--data $FU10K --width 50 --max-instances 8"
seq_pol $OUT/seq/w50 "seq w50 random" $S50 --score random --seeds 0 1 2 --fracs 10 20 50
for s in 0 1 2; do
    seq_pol $OUT/seq/w50 "seq w50 disagreement seed $s" $S50 --score fast_disagreement --seeds $s --fracs 10 20 50
    seq_pol $OUT/seq/w50 "seq w50 learned seed $s" $S50 --score fast_learned --seeds $s --fracs 10 20 50 \
        --model $OUT/models/w50/tagcn_k3_seed$s/chosen.joblib
done
seq_pol $OUT/seq/w50 "seq w50 all" $S50 --score random --seeds 0 --fracs 100 --folder-suffix _all
S20="--data $FU10K --width 20 --max-instances 8"
seq_pol $OUT/seq/w20 "seq w20 random top10" $S20 --score random --seeds 0 1 2 --fracs 10
for s in 0 1 2; do
    seq_pol $OUT/seq/w20 "seq w20 disagreement seed $s top10" $S20 --score fast_disagreement --seeds $s --fracs 10
    seq_pol $OUT/seq/w20 "seq w20 learned seed $s top10" $S20 --score fast_learned --seeds $s --fracs 10 \
        --model results/phase3_learned_triage/models/tagcn_k3_seed$s/chosen.joblib
done
seq_pol $OUT/seq/w20 "seq w20 all" $S20 --score random --seeds 0 --fracs 100 --folder-suffix _all

# 4. report
run "report" "$LOGDIR/report.log" python -u scripts/phase3_triage_wsize_report.py || FAILS=$((FAILS + 1))
log "ALL DONE triage wsize ($FAILS failures)"
