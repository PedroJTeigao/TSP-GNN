#!/usr/bin/env bash
# Phase 5, step 2, parts B-D, after text_v2b:
#   B  formalisation arms b0 / b1 / b2 on text_v1 and text_v2b, n = 200 and 1000 (scripts/formalize_eval.py; 7B);
#   C  full pipeline per arm, LKH-3 with at most 8 processes (scripts/pipeline_eval.py), in parallel with
#   D  the 7B as route verifier, n = 200 only (scripts/llm_verify_eval.py); D waits for the label-error list of the
#      manual check (data/phase5/text_v2b_label_errors.json).
# Outputs: data/phase5/{formalize,pipeline}_tsp{200,1000}.jsonl, data/phase5/llm_verify_tsp200.jsonl.
# Log: results/phase5/text_llm.log (START / OK / FAILED lines, then "ALL DONE").
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s text_llm "nohup bash scripts/run_phase5_text_llm.sh > results/phase5/text_llm.nohup 2>&1"

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
LOG=results/phase5/text_llm.log
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

# text_v2 must be finished first (its cached LLM calls are replayed)
until grep -q "ALL DONE text v2 " results/phase5/text_v2.log 2>/dev/null; do sleep 30; done
until grep -q "ALL DONE text v2b " results/phase5/text_v2b.log 2>/dev/null; do sleep 30; done
log "=== text rules LLM — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for n in 200 1000; do
    run "B formalize n = $n" "$LOGDIR/formalize_tsp$n.log" python -u scripts/formalize_eval.py --n $n
done
(
    for n in 200 1000; do
        run "C pipeline n = $n" "$LOGDIR/pipeline_tsp$n.log" python -u scripts/pipeline_eval.py --n $n --workers 8
    done
) &
until [ -f data/phase5/text_v2b_label_errors.json ]; do sleep 30; done
run "D LLM verifier n = 200" "$LOGDIR/llm_verify_tsp200.log" python -u scripts/llm_verify_eval.py --n 200
wait
FAILS=$(grep -c "FAILED" "$LOG")
log "ALL DONE text rules LLM ($FAILS failures)"
