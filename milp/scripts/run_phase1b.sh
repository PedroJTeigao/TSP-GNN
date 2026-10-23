#!/usr/bin/env bash
# FCNF phase 1b: equal-time baseline, trust region (predict-and-search) and feasible warm starts.
#
#   bash scripts/run_phase1b.sh small            fcnf20 and fcnf50 (everything), then fcnf100: full MILP trajectory
#                                                without warm start (200 instances) and with slope scaling (first 100)
#   bash scripts/run_phase1b.sh fcnf100 RUN...   fcnf100, first 100 instances: trust region and trust-region warm
#                                                start for the given runs, then the equal-time check sample
# Trust region and its warm start: TAGCN K=3 runs (the better encoder of phase 1). The equal-time baseline of
# the phase 1 decoders (both encoders, 3 seeds) is read from the trajectories, no extra runs.
# Time limit of every MILP: none on fcnf20, 120 s on fcnf50 / fcnf100. 8 processes, one HiGHS thread each.
# Outputs: results/phase1b/{decode,traj,equal_time_check}/...; log results/phase1b.log (START / OK / FAILED /
# SKIP lines, then "ALL DONE <stage>").
#
# Launch from milp/ (survives closing SSH):
#   tmux new-session -d -s fcnf1b "nohup bash scripts/run_phase1b.sh small > results/phase1b_small.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase1b
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR"
LOG=results/phase1b.log
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

limit_of() { [ "$1" = fcnf20_test ] && echo "" || echo "--time-limit 120"; }

traj() {      # set warm [source] [max instances]
    local set=$1 warm=$2 source=${3:-} max=${4:-}
    local tag=$warm extra=()
    [ "$warm" = trust ] && tag=trust_$source && extra+=(--source "$source")
    [ -n "$max" ] && extra+=(--max-instances "$max")
    if [ -f "$OUT/traj/$tag/$set.json" ]; then log "SKIP  traj $set $tag (already done)"; return 0; fi
    run "traj $set $tag" "$LOGDIR/traj_${set}_$tag.log" python -u scripts/milp_trajectory.py --data "data/$set.npz" \
        --warm "$warm" "${extra[@]}" $(limit_of "$set") --procs $PROCS --out-root "$OUT/traj"
}

trust() {     # set run [max instances]
    local set=$1 r=$2 extra=()
    [ -n "${3:-}" ] && extra+=(--max-instances "$3")
    if [ -f "$OUT/decode/$r/${set}__trust.json" ]; then log "SKIP  trust $set $r (already done)"; return 0; fi
    run "trust $set $r" "$LOGDIR/trust_${set}_$r.log" python -u scripts/decode_eval.py --data "data/$set.npz" \
        --source "$r" --group trust "${extra[@]}" $(limit_of "$set") --procs $PROCS --out-root "$OUT/decode"
}

check() {     # set run instances [largest fixed limit] [methods]
    local set=$1 r=$2 n=$3 fmax=${4:-50} extra=()
    [ -n "${5:-}" ] && extra+=(--methods "$5")
    if [ -f "$OUT/equal_time_check/$r/$set.json" ]; then log "SKIP  check $set $r (already done)"; return 0; fi
    run "check $set $r ($n instances)" "$LOGDIR/check_${set}_$r.log" python -u scripts/equal_time_check.py \
        --data "data/$set.npz" --source "$r" --max-instances "$n" --fixed-max "$fmax" "${extra[@]}" --procs $PROCS \
        --out-root "$OUT/equal_time_check"
}

stage=${1:?usage: run_phase1b.sh small | fcnf100 RUN...}
shift
log "commit $(git rev-parse --short HEAD), stage $stage $*"
if [ "$stage" = small ]; then
    TR_RUNS=(tagcn_k3_seed0 tagcn_k3_seed1 tagcn_k3_seed2)
    for set in fcnf20_test fcnf50_test; do
        traj "$set" none
        traj "$set" slope
        for r in "${TR_RUNS[@]}"; do trust "$set" "$r"; done
        for r in "${TR_RUNS[@]}"; do traj "$set" trust "$r"; done
        check "$set" tagcn_k3_seed0 100
    done
    traj fcnf100_test none
    traj fcnf100_test slope "" 100
elif [ "$stage" = fcnf100 ]; then
    for r in "$@"; do trust fcnf100_test "$r" 100; done
    for r in "$@"; do traj fcnf100_test trust "$r" 100; done
    check fcnf100_test "$1" 40 25 prune_q20,prune_q30,prune_q50,delegate,tr_q20_d5,tr_q50_d20
fi

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE $stage with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE $stage (0 failures)"
fi
