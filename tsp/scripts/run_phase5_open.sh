#!/usr/bin/env bash
# Phase 5, step 3: open language with a larger local model. Every step resumes if the run is restarted.
#   1. L1 (scripts/make_open_language.py l1) must be finished first, for both n;
#   2. L2: gemma3:12b rewrites L1 (test + dev);
#   3. B: p0, p1, then q14, q7, q14t (thinking on, L2 only), g12 (scripts/open_formalize.py); the same first instances for
#      all arms, so the comparisons are paired;
#   4, 5 in parallel: step 3, qwen3:14b on the older tasks (scripts/open_step3.py), and C, code check + LKH-3
#      (scripts/open_pipeline.py); step-3 latencies are therefore marked as measured under load;
#   6. latency alone: 30 calls per LLM arm with no LKH of mine running (scripts/open_latency.py).
# Settings (environment): sample sizes below, cut to about 6 h because the T4 throttles itself to about 300 MHz.
# Log: results/phase5/open_language/run.log (START / OK / FAILED lines, then "ALL DONE").
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s p5open "nohup bash scripts/run_phase5_open.sh > results/phase5/open_language/run.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export OMP_NUM_THREADS=1 GIT_CONFIG_GLOBAL=/dev/null
OUT=results/phase5/open_language
LOGDIR=$OUT/logs
LOG=$OUT/run.log
THINK=${THINK:-off} L2NUM=${L2NUM:-60} NUM=${NUM:-60} NUML0=${NUML0:-30} NUMQ7=${NUMQ7:-40} NUMQ7L0=${NUMQ7L0:-20}
NUMG=${NUMG:-20} NUMT=${NUMT:-12} RULES3=${RULES3:-100} ROUTES3=${ROUTES3:-50} CLIMIT200=${CLIMIT200:-60} CLIMIT1K=${CLIMIT1K:-30}
FAILS=0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {
    local desc=$1 out=$2; shift 2
    log "START $desc"
    local start=$SECONDS
    if "$@" >> "$out" 2>&1; then
        log "OK    $desc ($(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s)"
    else
        log "FAILED $desc (see $out)"
        FAILS=$((FAILS + 1))
    fi
}

log "=== open language — commit $(git rev-parse --short HEAD), niceness $(nice), think $THINK, q14 $NUM ($NUML0 L0), q7 $NUMQ7 ($NUMQ7L0), g12 $NUMG, q14t $NUMT, step 3 $RULES3 rules $ROUTES3 routes, C limit $CLIMIT200 / $CLIMIT1K ==="
until grep -q "ALL DONE l1" $LOGDIR/l1_1000.log 2>/dev/null; do sleep 60; done
run "L2 n = 1000" $LOGDIR/l2_1000.log python -u scripts/make_open_language.py l2 --n 1000 --limit $L2NUM
for n in 200 1000; do
    run "B parsers n = $n" $LOGDIR/b_$n.log python -u scripts/open_formalize.py --n $n --arms p0 p1 --num $NUM --num-l0 $NUML0
done
for n in 200 1000; do
    run "B q14 n = $n" $LOGDIR/b_$n.log python -u scripts/open_formalize.py --n $n --arms q14 --num $NUM --num-l0 $NUML0 --think $THINK
done
for n in 200 1000; do
    run "B q7 n = $n" $LOGDIR/b_$n.log python -u scripts/open_formalize.py --n $n --arms q7 --num $NUMQ7 --num-l0 $NUMQ7L0
done
for n in 200 1000; do
    run "B q14t n = $n" $LOGDIR/b_$n.log python -u scripts/open_formalize.py --n $n --arms q14t --levels L2 --num $NUMT
done
for n in 200 1000; do
    run "B g12 n = $n" $LOGDIR/b_$n.log python -u scripts/open_formalize.py --n $n --arms g12 --num $NUMG
done
step3_log=$LOGDIR/step3.status
( run "step 3" $LOGDIR/step3.log python -u scripts/open_step3.py --think $THINK --rules $RULES3 --routes $ROUTES3; echo $FAILS > $step3_log ) &
for n in 200 1000; do
    lim=$CLIMIT200; [ $n = 1000 ] && lim=$CLIMIT1K
    run "C n = $n" $LOGDIR/c_$n.log python -u scripts/open_pipeline.py --n $n --workers 7 --limit $lim
done
wait
FAILS=$((FAILS + $(cat $step3_log)))
# last, with nothing of mine running: the latencies that go into the report
run "latency alone" $LOGDIR/latency.log python -u scripts/open_latency.py --think $THINK
log "ALL DONE open language ($FAILS failures)"
