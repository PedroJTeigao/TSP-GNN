#!/usr/bin/env bash
# Phase 5, step 2, completion of B-D: the main run (scripts/run_phase5_text_llm.sh) stopped part B at n = 200 when
# Ollama aborted one answer (HTTP 500 "token repeat limit reached"; now recorded as a failed answer). After the main
# run and the b0+ pipeline (scripts/run_phase5_text_b0p.sh) are done, this resumes B (records already written are
# skipped), runs C on the predictions not yet in a pipeline file (at most 8 LKH processes), resumes D and writes the
# report.
# Outputs: data/phase5/formalize_tsp<n>.jsonl (appended), data/phase5/pipeline_rest_tsp<n>.jsonl,
# data/phase5/llm_verify_tsp200.jsonl (appended), results/phase5/text_rules_llm.md. Log: results/phase5/text_rest.log.
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s text_rest "nohup bash scripts/run_phase5_text_rest.sh > results/phase5/text_rest.nohup 2>&1"

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
LOG=results/phase5/text_rest.log
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

until grep -q "ALL DONE text rules LLM" results/phase5/text_llm.log 2>/dev/null && \
      grep -q "ALL DONE b0+ pipeline" results/phase5/text_b0p.log 2>/dev/null; do sleep 60; done
log "=== B-D completion — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for n in 200 1000; do
    run "B formalize (resume) n = $n" "$LOGDIR/formalize_rest_tsp$n.log" python -u scripts/formalize_eval.py --n $n
done
for n in 200 1000; do
    run "C pipeline (rest) n = $n" "$LOGDIR/pipeline_rest_tsp$n.log" python -u scripts/pipeline_eval.py --n $n \
        --workers 8 --skip-done data/phase5/pipeline_tsp$n.jsonl --out data/phase5/pipeline_rest_tsp$n.jsonl
done
run "D LLM verifier (resume) n = 200" "$LOGDIR/llm_verify_rest_tsp200.log" python -u scripts/llm_verify_eval.py --n 200
run "report" "$LOGDIR/text_llm_report.log" python -u scripts/phase5_text_llm_report.py
log "ALL DONE B-D completion ($FAILS failures)"
