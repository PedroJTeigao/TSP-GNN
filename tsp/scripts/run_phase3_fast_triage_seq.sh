#!/usr/bin/env bash
# Phase 3, fast triage: the equal-time comparison on Fu-10k with ONE process at a time (no other process of ours
# running): random (5 rng seeds), batched disagreement and batched learned (3 TAGCN seeds each), top 20 % and 50 %.
# Before and after each run, records the load average and the GPU memory used by other processes
# (results/phase3_fast_triage/seq/load.tsv). Then regenerates the report.
# Log: results/phase3_fast_triage/seq.log.
#
# Launch from tsp/:
#   tmux new-session -d -s fast_seq "nohup bash scripts/run_phase3_fast_triage_seq.sh > results/phase3_fast_triage/seq.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi
source ../env.sh
export OMP_NUM_THREADS=1 GIT_CONFIG_GLOBAL=/dev/null
OUT=results/phase3_fast_triage
SEQ=$OUT/seq
DATA=results/phase3_benchmark/data/tsp10000_test_concorde.npz
MODELS=results/phase3_learned_triage/models
LOG=$OUT/seq.log
LOAD=$SEQ/load.tsv
FAILS=0
mkdir -p $SEQ
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
load() {   # when run load1 load5 load15 gpu_other_MiB
    local l g
    l=$(cut -d' ' -f1-3 /proc/loadavg)
    g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
    printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$(echo $l | tr ' ' '\t')" "$g" >> $LOAD
}
[ -f $LOAD ] || printf "when\trun\tload1\tload5\tload15\tgpu_other_mib\n" > $LOAD
one() {   # label args...
    local label=$1; shift
    load before "$label"
    log "START $label"
    if python -u scripts/triage_ablation.py --data $DATA --out-root $SEQ "$@" > $OUT/logs/seq_${label}.log 2>&1; then
        log "OK    $label"
    else
        log "FAILED $label (see $OUT/logs/seq_${label}.log)"; FAILS=$((FAILS + 1))
    fi
    load after "$label"
}
log "=== equal time, one process at a time — commit $(git rev-parse --short HEAD) ==="
one random --score random
for s in 0 1 2; do
    one fast_disagreement_seed$s --score fast_disagreement --seeds $s
    one fast_learned_seed$s --score fast_learned --seeds $s --model $MODELS/tagcn_k3_seed$s/chosen.joblib
done
python -u scripts/phase3_fast_triage_report.py > $OUT/logs/report_seq.log 2>&1 || FAILS=$((FAILS + 1))
log "ALL DONE sequential ($FAILS failures)"
