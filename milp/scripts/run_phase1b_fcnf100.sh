#!/usr/bin/env bash
# FCNF phase 1b on fcnf100 (the rest is scripts/run_phase1b.sh small, which also runs the fcnf100 full-MILP
# trajectories without warm start and with slope scaling). First 100 instances, tagcn_k3_seed0 only:
#   trust region, all 9 (q, delta), with a 25 s limit (about the time of pruning q50 on fcnf100)
#   q50 / delta 20 also with the usual 120 s limit, as a reference point (results/phase1b/decode_tr120)
#   full MILP with the 25 s q50 / delta 20 solution as warm start
#   equal-time check on 40 instances (pruning, delegation, two trust regions, fixed limits up to 25 s)
# Outputs: results/phase1b/{decode,traj,equal_time_check}/...; log results/phase1b.log (START / OK / FAILED /
# SKIP lines, then "ALL DONE <stage>").
#
# Launch from milp/ (survives closing SSH):
#   tmux new-session -d -s fcnf1b100 "nohup bash scripts/run_phase1b_fcnf100.sh > results/phase1b_fcnf100.nohup 2>&1"

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

trust() {     # set run [max instances] [time limit] [configs] [out dir]
    local set=$1 r=$2 extra=() limit dir=${6:-decode}
    [ -n "${3:-}" ] && extra+=(--max-instances "$3")
    limit=$(limit_of "$set")
    [ -n "${4:-}" ] && limit="--time-limit $4"
    [ -n "${5:-}" ] && extra+=(--configs "$5")
    if [ -f "$OUT/$dir/$r/${set}__trust.json" ]; then log "SKIP  trust $set $r $dir (already done)"; return 0; fi
    run "trust $set $r $dir ($limit)" "$LOGDIR/trust_${set}_${r}_$dir.log" python -u scripts/decode_eval.py \
        --data "data/$set.npz" --source "$r" --group trust "${extra[@]}" $limit --procs $PROCS --out-root "$OUT/$dir"
}

check() {     # set run instances [largest fixed limit] [methods]
    local set=$1 r=$2 n=$3 fmax=${4:-50} extra=()
    [ -n "${5:-}" ] && extra+=(--methods "$5")
    if [ -f "$OUT/equal_time_check/$r/$set.json" ]; then log "SKIP  check $set $r (already done)"; return 0; fi
    run "check $set $r ($n instances)" "$LOGDIR/check_${set}_$r.log" python -u scripts/equal_time_check.py \
        --data "data/$set.npz" --source "$r" --max-instances "$n" --fixed-max "$fmax" "${extra[@]}" --procs $PROCS \
        --out-root "$OUT/equal_time_check"
}

stage=fcnf100
log "commit $(git rev-parse --short HEAD), stage $stage"
trust fcnf100_test tagcn_k3_seed0 100 25
traj fcnf100_test trust tagcn_k3_seed0 100
trust fcnf100_test tagcn_k3_seed0 100 120 q50_d20 decode_tr120
check fcnf100_test tagcn_k3_seed0 40 25 prune_q20,prune_q30,prune_q50,delegate,tr_q20_d5,tr_q50_d20

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE $stage with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE $stage (0 failures)"
fi
