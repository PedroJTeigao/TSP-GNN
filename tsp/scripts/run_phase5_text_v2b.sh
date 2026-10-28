#!/usr/bin/env bash
# Phase 5, step 2, part A (revised): build text_v2b = text_v2 with the meaning filters (--strict; paraphrases by the local qwen2.5:7b, names, distractors,
# merged sentences) for the step-1 instances at n = 200 and n = 1000 (scripts/make_text_v2.py; needs ollama_pc).
# Outputs: data/phase5/text_rules_v2b_tsp{200,1000}.jsonl; new LLM calls in data/phase5/text_v2b_calls_tsp<n>.jsonl.
# Log: results/phase5/text_v2b.log (START / OK / FAILED lines, then "ALL DONE").
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s text_v2b "nohup bash scripts/run_phase5_text_v2b.sh > results/phase5/text_v2b.nohup 2>&1"

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
LOG=results/phase5/text_v2b.log
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
log "=== text v2b — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for n in 200 1000; do
    run "text_v2b n = $n" "$LOGDIR/text_v2b_tsp$n.log" python -u scripts/make_text_v2.py --n $n --strict
done
log "ALL DONE text v2b ($FAILS failures)"
