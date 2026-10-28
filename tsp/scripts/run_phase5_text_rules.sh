#!/usr/bin/env bash
# Phase 5, step 1: label 200 text-rule TSP instances at n = 200 and n = 1000 with the LKH-3 reference solver
# (scripts/make_text_rules.py, 8 instances in parallel = at most 8 LKH processes), then write the report.
# Outputs: data/phase5/text_rules_tsp{200,1000}.{jsonl,npz}; report results/phase5/text_rules_data.md.
# Log: results/phase5/text_rules.log (START / OK / FAILED lines, then "ALL DONE").
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s text_rules "nohup bash scripts/run_phase5_text_rules.sh > results/phase5/text_rules.nohup 2>&1"

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
LOG=results/phase5/text_rules.log
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

log "=== text rules — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
run "label n = 200" "$LOGDIR/text_rules_tsp200.log" \
    python -u scripts/make_text_rules.py --n 200 --num 200 --workers 8 --time-limit 60
run "label n = 1000" "$LOGDIR/text_rules_tsp1000.log" \
    python -u scripts/make_text_rules.py --n 1000 --num 200 --workers 8 --time-limit 120
run "report" "$LOGDIR/text_rules_report.log" python -u scripts/phase5_text_rules_report.py
log "ALL DONE text rules ($FAILS failures)"
