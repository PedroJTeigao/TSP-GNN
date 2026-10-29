#!/usr/bin/env bash
# Phase 5, step 2: rerun of part C at n = 1000 on the predictions not yet in a pipeline file, after the completion
# run (scripts/run_phase5_text_rest.sh) failed there: one b2 answer had "t2": "inf", which the schema now rejects
# (formalize.validate, formalize.load_preds), and the pipeline wrote nothing because results are written at the end
# (a solver error in one job no longer stops the others). Then writes the report.
# Outputs: data/phase5/pipeline_rest_tsp1000.jsonl, results/phase5/text_rules_llm.md. Log: results/phase5/text_c1000.log.
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s text_c1000 "nohup bash scripts/run_phase5_text_c1000.sh > results/phase5/text_c1000.nohup 2>&1"

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
LOG=results/phase5/text_c1000.log
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

log "=== C n = 1000 rerun — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
run "C pipeline (rest, rerun) n = 1000" "$LOGDIR/pipeline_rest2_tsp1000.log" python -u scripts/pipeline_eval.py \
    --n 1000 --workers 8 --skip-done data/phase5/pipeline_tsp1000.jsonl --out data/phase5/pipeline_rest_tsp1000.jsonl
run "report" "$LOGDIR/text_llm_report2.log" python -u scripts/phase5_text_llm_report.py
log "ALL DONE C n = 1000 rerun ($FAILS failures)"
