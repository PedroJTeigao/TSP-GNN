#!/usr/bin/env bash
# Phase 5 vision probe (scripts/vision_probe.py):
#   0. waits for the triage-with-larger-windows run to finish (its sequential timings must not be disturbed) and for
#      the GPU memory used by other processes (not ollama_pc, not ours) to fall below 6 GiB, checking every 10 min
#      for at most 6 h; after 6 h it goes on with gemma4:e2b. Start time and load in results/phase5/vision_probe/wait.log;
#   1. pulls the model into the private Ollama (port 11435), draws the images, fits num_gpu to the budget (falls back
#      to gemma4:e2b if the e4b model cannot be loaded within it);
#   2. 300 calls (50 Fu-1k instances × images a / b × 3 repetitions), the cell policies, the report.
# Log: results/phase5/vision_probe/run.log.
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s vision "nohup bash scripts/run_phase5_vision.sh > results/phase5/vision_probe/run.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi
source ../env.sh
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase5/vision_probe
mkdir -p $OUT/logs
LOG=$OUT/run.log
WAIT=$OUT/wait.log
FAILS=0
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
run() {
    local desc=$1 out=$2; shift 2
    log "START $desc"
    if "$@" > "$out" 2>&1; then log "OK    $desc"; else log "FAILED $desc (see $out)"; return 1; fi
}
state() { echo "$(date '+%Y-%m-%d %H:%M:%S')  load $(cut -d' ' -f1-3 /proc/loadavg)  GPU used by others $(python scripts/vision_probe.py --step gpu) MiB  $*"; }

log "=== vision probe — commit $(git rev-parse --short HEAD) ==="
START=$SECONDS
MODEL=gemma4:e4b-it-qat
until grep -q "ALL DONE triage wsize" results/phase3_triage_wsize/run.log 2>/dev/null; do
    state "waiting for the larger-windows run" >> $WAIT; sleep 600
done
while true; do
    G=$(python scripts/vision_probe.py --step gpu)
    if [ "$G" -lt 6144 ]; then state "GPU below 6 GiB: start with $MODEL" >> $WAIT; break; fi
    if [ $((SECONDS - START)) -ge 21600 ]; then
        MODEL=gemma4:e2b; state "6 h without the GPU below 6 GiB: start anyway with $MODEL" >> $WAIT; break
    fi
    state "waiting for the GPU" >> $WAIT; sleep 600
done
log "start with $MODEL"

run "pull $MODEL" $OUT/logs/pull_${MODEL//:/_}.log scripts/ollama_pc.sh pull $MODEL || FAILS=$((FAILS + 1))
run "images" $OUT/logs/images.log python -u scripts/vision_probe.py --step images || FAILS=$((FAILS + 1))
if ! run "tune $MODEL" $OUT/logs/tune_${MODEL//:/_}.log python -u scripts/vision_probe.py --step tune --model $MODEL; then
    if [ "$MODEL" != "gemma4:e2b" ]; then
        MODEL=gemma4:e2b
        log "e4b could not be fitted: fall back to $MODEL"
        run "pull $MODEL" $OUT/logs/pull_${MODEL//:/_}.log scripts/ollama_pc.sh pull $MODEL || FAILS=$((FAILS + 1))
        run "tune $MODEL" $OUT/logs/tune_${MODEL//:/_}.log python -u scripts/vision_probe.py --step tune --model $MODEL \
            || FAILS=$((FAILS + 1))
    else
        FAILS=$((FAILS + 1))
    fi
fi
run "ask $MODEL" $OUT/logs/ask.log python -u scripts/vision_probe.py --step ask --model $MODEL || FAILS=$((FAILS + 1))
state "after the calls" >> $WAIT
run "policy" $OUT/logs/policy.log python -u scripts/vision_probe.py --step policy --model $MODEL || FAILS=$((FAILS + 1))
run "report" $OUT/logs/report.log python -u scripts/phase5_vision_probe_report.py --model $MODEL || FAILS=$((FAILS + 1))
log "ALL DONE vision probe ($FAILS failures) model $MODEL"
