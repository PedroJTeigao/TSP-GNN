#!/usr/bin/env bash
# Phase 3 benchmark, E0: new LKH-3 reference (stand-alone LKH-3.0.6 binary, scripts/lkh3_reference.py).
#
#   1k  (tsp1000_test, 20 inst.; tsp1000_test_concorde, 128 inst.):   RUNS = 10, no time limit
#   10k (tsp10000_test, 3 inst.; tsp10000_test_concorde, 16 inst.):  RUNS = 5, TIME_LIMIT = 696 s PER RUN
#        (LKH-3.0.6 has no total limit; 5 × 696 s + ~120 s of preprocessing ≈ 1 h per instance)
# 8 processes. Outputs: results/phase3_benchmark/ref/<set>.npz, then results/phase3_benchmark/ref_lkh3.npz.
# Log: results/phase3_benchmark/e0.log (START / OK / FAILED / SKIP lines, then "ALL DONE e0").
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s bench_e0 "nohup bash scripts/run_phase3_bench_e0.sh > results/phase3_benchmark/e0.nohup 2>&1"

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
REF=$OUT/ref
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures_e0
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" "$REF"
LOG=$OUT/e0.log
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

ref() {       # data file, LKH parameters
    local data=$1 params=$2 name
    name=$(basename "$data" .npz)
    if [ -f "$REF/$name.npz" ]; then log "SKIP  ref $name (already done)"; return 0; fi
    run "ref $name ($params)" "$LOGDIR/e0_$name.log" python -u scripts/lkh3_reference.py --data "$data" \
        --params "$params" --procs $PROCS --out-dir "$REF"
}

log "commit $(git rev-parse --short HEAD), E0"
ref data/tsp1000_test.npz RUNS=10
ref data/benchmark/tsp1000_test_concorde.npz RUNS=10
ref data/tsp10000_test.npz RUNS=5,TIME_LIMIT=696
ref data/benchmark/tsp10000_test_concorde.npz RUNS=5,TIME_LIMIT=696
if [ ! -f "$OUT/ref_lkh3.npz" ]; then
    run "combine ref_lkh3.npz" "$LOGDIR/e0_combine.log" python -u scripts/lkh3_reference.py \
        --combine "$OUT/ref_lkh3.npz" --out-dir "$REF" \
        --sets tsp1000_test tsp1000_test_concorde tsp10000_test tsp10000_test_concorde
fi

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE e0 with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE e0 (0 failures)"
fi
