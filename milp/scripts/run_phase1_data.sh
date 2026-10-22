#!/usr/bin/env bash
# FCNF phase 1, steps 1-3: generate and label the data sets with the MILP (scripts/make_dataset.py), one
# set at a time (each uses 8 processes, one HiGHS thread each), then the generator statistics.
#
#   fcnf20_test   1000  seed 33  to optimality       data/fcnf20_test.npz
#   fcnf20_val    1000  seed 32  to optimality       data/fcnf20_val.npz
#   fcnf20_train 12000  seed 31  to optimality       data/fcnf20_train.npz
#   fcnf50_test    500  seed 34  120 s limit         data/fcnf50_test.npz
#   fcnf100_test   200  seed 35  120 s limit         data/fcnf100_test.npz
#   -> results/phase1_fcnf_data_stats.md
# Progress goes to results/phase1_data.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
#
# Launch from milp/ (survives closing SSH):
#   tmux new-session -d -s fcnfdata "nohup bash scripts/run_phase1_data.sh > results/phase1_data.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
LOGDIR=results/phase1_data_logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" data
LOG=results/phase1_data.log

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
    touch "$FAILDIR/$desc"
    log "FAILED $desc (exit code $code). Last lines of $out:"
    tail -n 15 "$out" | sed 's/^/    /' | tee -a "$LOG"
    return 1
}

dataset() {   # name n num seed [time limit]
    if [ -f "data/$1.npz" ]; then log "SKIP  $1 (already exists)"; return 0; fi
    local limit=()
    [ -n "${5:-}" ] && limit=(--time-limit "$5")
    run "$1" "$LOGDIR/$1.log" python -u scripts/make_dataset.py --n "$2" --num "$3" --seed "$4" "${limit[@]}" \
        --out "data/$1.npz"
}

log "commit $(git rev-parse --short HEAD)"
dataset fcnf20_test   20  1000 33
dataset fcnf20_val    20  1000 32
dataset fcnf20_train  20 12000 31
dataset fcnf50_test   50   500 34 120
dataset fcnf100_test 100   200 35 120

run "data stats" "$LOGDIR/data_stats.log" python scripts/data_stats.py \
    --data data/fcnf20_train.npz data/fcnf20_val.npz data/fcnf20_test.npz data/fcnf50_test.npz data/fcnf100_test.npz \
    --out results/phase1_fcnf_data_stats.md

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE (0 failures)"
fi
