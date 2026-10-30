#!/usr/bin/env bash
# Phase 3, fast triage: random (5 rng seeds) on Fu-10k again, alone, right after the sequential run
# (scripts/run_phase3_fast_triage_seq.sh). During that run another project started CPU-heavy jobs (load average
# 0.4 -> 7), so random (run first) and the network policies (run later) may have seen different loads; this brackets
# them. Load recorded in results/phase3_fast_triage/seq_after/load.tsv. Log: results/phase3_fast_triage/seq_after.log.
#
# Launch from tsp/:
#   tmux new-session -d -s fast_seq_after "nohup bash scripts/run_phase3_fast_triage_seq_after.sh > results/phase3_fast_triage/seq_after.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi
source ../env.sh
export OMP_NUM_THREADS=1 GIT_CONFIG_GLOBAL=/dev/null
OUT=results/phase3_fast_triage
ROOT=$OUT/seq_after
LOG=$OUT/seq_after.log
LOAD=$ROOT/load.tsv
mkdir -p $ROOT
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
load() {
    local l g
    l=$(cut -d' ' -f1-3 /proc/loadavg)
    g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
    printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$(echo $l | tr ' ' '\t')" "$g" >> $LOAD
}
until grep -q "ALL DONE sequential" $OUT/seq.log 2>/dev/null; do sleep 30; done
[ -f $LOAD ] || printf "when\trun\tload1\tload5\tload15\tgpu_other_mib\n" > $LOAD
log "=== random again, alone — commit $(git rev-parse --short HEAD) ==="
load before random_after
if python -u scripts/triage_ablation.py --data results/phase3_benchmark/data/tsp10000_test_concorde.npz \
    --score random --out-root $ROOT > $OUT/logs/seq_random_after.log 2>&1; then F=0; else F=1; fi
load after random_after
python -u scripts/phase3_fast_triage_report.py > $OUT/logs/report_seq_after.log 2>&1 || F=$((F + 1))
log "ALL DONE random after ($F failures)"
