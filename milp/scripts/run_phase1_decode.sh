#!/usr/bin/env bash
# FCNF phase 1, steps 5-7: heatmap-guided decoders and baselines on fcnf20 / fcnf50 / fcnf100, then the report.
#
# Runs (results/runs): gcn_seed0-2, tagcn_k3_seed0-2. Time limit of every MILP: none on fcnf20, 120 s on
# fcnf50 / fcnf100 (as the reference MILP). One decode_eval.py at a time; it uses 2 processes while the
# fcnf100 600 s re-solve (run_phase1_train.sh) is still running, 8 after.
#   A. fcnf20, then fcnf50: baselines, guided (prune q20/30/50, delegate) and warm for every run
#   B. fcnf100: baselines, guided for every run, warm for seed 0 of each encoder
#   -> report results/phase1_fcnf_report.md (copied to docs/)
#   C. fcnf100: warm for seeds 1-2 (2 × 2 × 200 full MILPs of up to 120 s, ~3 h) -> report again
# Outputs: results/phase1_decode/<run | baselines>/<set>__<group>.{npz,json}.
# Progress goes to results/phase1_decode.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
#
# Launch from milp/ (survives closing SSH):
#   tmux new-session -d -s fcnfdecode "nohup bash scripts/run_phase1_decode.sh > results/phase1_decode.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase1_decode
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" results/figs
LOG=results/phase1_decode.log
RUNS=(gcn_seed0 gcn_seed1 gcn_seed2 tagcn_k3_seed0 tagcn_k3_seed1 tagcn_k3_seed2)

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

procs() {   # 2 while the 600 s re-solve runs (it uses 6), 8 after
    if pgrep -f "reference_rerun.py" > /dev/null; then echo 2; else echo 8; fi
}

evaluate() {   # set source group [time limit]
    local set=$1 source=$2 group=$3 limit=()
    local tag=$group
    [ "$source" = baselines ] && tag=baselines
    [ -n "${4:-}" ] && limit=(--time-limit "$4")
    if [ -f "$OUT/$source/${set}__$tag.json" ]; then log "SKIP  $set $source $tag (already done)"; return 0; fi
    local p
    p=$(procs)
    run "$set $source $tag (procs $p)" "$LOGDIR/${set}_${source}_$tag.log" python -u scripts/decode_eval.py \
        --data "data/$set.npz" --source "$source" --group "$group" "${limit[@]}" --procs "$p" --out-root "$OUT"
}

report() {
    if [ -f scripts/phase1_fcnf_report.py ]; then
        run "report" "$LOGDIR/report.log" python scripts/phase1_fcnf_report.py \
            && cp results/phase1_fcnf_report.md docs/phase1_fcnf_report.md && mkdir -p docs/figs \
            && { cp results/figs/phase1_fcnf_*.png docs/figs/ 2>/dev/null; log "copied to docs/phase1_fcnf_report.md"; }
    else
        log "SKIP  report (scripts/phase1_fcnf_report.py does not exist yet)"
    fi
}

log "commit $(git rev-parse --short HEAD)"
for set in fcnf20_test fcnf50_test; do
    limit=""
    [ "$set" = fcnf50_test ] && limit=120
    evaluate "$set" baselines guided $limit
    for r in "${RUNS[@]}"; do evaluate "$set" "$r" guided $limit; done
    for r in "${RUNS[@]}"; do evaluate "$set" "$r" warm $limit; done
done
evaluate fcnf100_test baselines guided 120
for r in "${RUNS[@]}"; do evaluate fcnf100_test "$r" guided 120; done
for r in gcn_seed0 tagcn_k3_seed0; do evaluate fcnf100_test "$r" warm 120; done
while pgrep -f "reference_rerun.py" > /dev/null; do sleep 60; done   # the report needs the 600 s references
report
for r in gcn_seed1 gcn_seed2 tagcn_k3_seed1 tagcn_k3_seed2; do evaluate fcnf100_test "$r" warm 120; done
report

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE (0 failures)"
fi
