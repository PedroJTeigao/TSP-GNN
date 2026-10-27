#!/usr/bin/env bash
# Phase 3 benchmark, E1: LKH-3 on the whole instance at the budgets of our policies (scripts/lkh3_equal_time.py).
#
# Waits until E0 has finished ("ALL DONE e0" in results/phase3_benchmark/e0.log), so the two never share the CPU.
#   1k  budgets 0.72 1.25 1.54 1.94 s:  tsp1000_test (20), tsp1000_test_concorde (128)
#   10k budgets 8.4 14.2 17.5 21.5 s:   tsp10000_test (3), tsp10000_test_concorde (16)
# Variants and their fixed costs: results/phase3_benchmark/e1_prep/tsp{1000,10000}{,_extra}.json.
# 8 LKH processes. Outputs: results/phase3_benchmark/e1/<set>.{npz,json}.
# Log: results/phase3_benchmark/e1.log (START / OK / FAILED / SKIP lines, then "ALL DONE e1").
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s bench_e1 "nohup bash scripts/run_phase3_bench_e1.sh > results/phase3_benchmark/e1.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase3_benchmark
PREP=$OUT/e1_prep
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures_e1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" "$OUT/e1"
LOG=$OUT/e1.log
PROCS=8

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {
    local desc=$1 out=$2; shift 2
    log "START $desc"
    local start=$SECONDS
    "$@" > "$out" 2>&1
    local code=$?
    if [ "$code" -eq 0 ]; then
        log "OK    $desc ($(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s)"
        return 0
    fi
    touch "$FAILDIR/$(echo "$desc" | tr -c 'A-Za-z0-9_=.-' '_')"
    log "FAILED $desc (exit code $code). Last lines of $out:"
    tail -n 15 "$out" | sed 's/^/    /' | tee -a "$LOG"
    return 1
}

e1() {        # data file, size (1000 / 10000), budgets...
    local data=$1 n=$2 name; shift 2
    name=$(basename "$data" .npz)
    if [ -f "$OUT/e1/$name.json" ]; then log "SKIP  e1 $name (already done)"; return 0; fi
    run "e1 $name" "$LOGDIR/e1_$name.log" python -u scripts/lkh3_equal_time.py --data "$data" --budgets "$@" \
        --prep "$PREP/tsp$n.json" "$PREP/tsp${n}_extra.json" --procs $PROCS --out-root "$OUT/e1"
}

log "commit $(git rev-parse --short HEAD), E1: waiting for E0 to finish"
until grep -q "ALL DONE e0" "$OUT/e0.log" 2>/dev/null; do sleep 60; done
log "E0 finished: starting E1"
e1 data/tsp1000_test.npz 1000 0.72 1.25 1.54 1.94
e1 data/benchmark/tsp1000_test_concorde.npz 1000 0.72 1.25 1.54 1.94
e1 data/tsp10000_test.npz 10000 8.4 14.2 17.5 21.5
e1 data/benchmark/tsp10000_test_concorde.npz 10000 8.4 14.2 17.5 21.5

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE e1 with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE e1 (0 failures)"
fi
