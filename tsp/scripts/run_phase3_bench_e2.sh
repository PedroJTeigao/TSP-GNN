#!/usr/bin/env bash
# Phase 3 benchmark, E2: ablation of the triage ranking (scripts/triage_ablation.py), w = 20, x = 20 % and 50 %.
#
# Waits until E3 has finished ("ALL DONE e3" in results/phase3_benchmark/e3.log; E3 also writes the data files).
# Sets (gap against the E0 LKH-3 reference, results/phase3_benchmark/data/):
#   tsp1000_test (20), tsp1000_test_concorde (128), tsp10000_test_concorde (16)
# Per set, 6 processes in parallel: random (rng seeds 0-4), simple rule, oracle, disagreement TAGCN K=3 seeds 0-2.
# Outputs: results/phase3_benchmark/e2/<score>[_seed<s>]/<set>.{json,npz}.
# Log: results/phase3_benchmark/e2.log (START / OK / FAILED / SKIP lines, then "ALL DONE e2").
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s bench_e2 "nohup bash scripts/run_phase3_bench_e2.sh > results/phase3_benchmark/e2.nohup 2>&1"

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
E2=$OUT/e2
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures_e2
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" "$E2"
LOG=$OUT/e2.log
SETS=(tsp1000_test tsp1000_test_concorde tsp10000_test_concorde)

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

one() {       # set, folder, score arguments...
    local name=$1 folder=$2; shift 2
    if [ -f "$E2/$folder/$name.json" ]; then log "SKIP  e2 $name $folder (already done)"; return 0; fi
    run "e2 $name $folder" "$LOGDIR/e2_${name}_$folder.log" python -u scripts/triage_ablation.py \
        --data "$OUT/data/$name.npz" --out-root "$E2" "$@"
}

log "commit $(git rev-parse --short HEAD), E2: waiting for E3 to finish"
until grep -q "ALL DONE e3" "$OUT/e3.log" 2>/dev/null; do sleep 60; done
log "E3 finished: starting E2"
for name in "${SETS[@]}"; do
    one "$name" random --score random &
    one "$name" simple --score simple &
    one "$name" oracle --score oracle &
    for s in 0 1 2; do
        one "$name" disagreement_seed$s --score disagreement --seeds $s &
    done
    wait
done

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE e2 with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE e2 (0 failures)"
fi
