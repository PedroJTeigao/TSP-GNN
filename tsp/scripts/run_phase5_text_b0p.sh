#!/usr/bin/env bash
# Phase 5, step 2, part C for arm b0+ (keyword parser, added after the main run): waits until the main part C
# (scripts/run_phase5_text_llm.sh) has finished both sizes, so that at most 8 LKH processes run, then runs the
# full pipeline on the b0+ predictions (data/phase5/formalize_b0p_tsp<n>.jsonl, written by
#   python scripts/formalize_eval.py --n <n> --arms b0p --out data/phase5/formalize_b0p_tsp<n>.jsonl).
# Outputs: data/phase5/pipeline_b0p_tsp{200,1000}.jsonl. Log: results/phase5/text_b0p.log.
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s text_b0p "nohup bash scripts/run_phase5_text_b0p.sh > results/phase5/text_b0p.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export OMP_NUM_THREADS=1 GIT_CONFIG_GLOBAL=/dev/null
LOGDIR=results/phase5/logs
mkdir -p "$LOGDIR"
LOG=results/phase5/text_b0p.log
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
        FAILS=$((FAILS + 1))
    fi
}

until grep -qE "(OK|FAILED) +C pipeline n = 1000" results/phase5/text_llm.log 2>/dev/null; do sleep 60; done
log "=== b0+ pipeline — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for n in 200 1000; do
    run "C pipeline b0+ n = $n" "$LOGDIR/pipeline_b0p_tsp$n.log" python -u scripts/pipeline_eval.py --n $n \
        --workers 8 --preds data/phase5/formalize_b0p_tsp$n.jsonl --no-oracle --out data/phase5/pipeline_b0p_tsp$n.jsonl
done
log "ALL DONE b0+ pipeline ($FAILS failures)"
