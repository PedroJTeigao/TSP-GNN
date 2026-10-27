#!/usr/bin/env bash
# Phase 2 ablation: test graphs with k = 0.2 n (Joshi et al. 2021) against our k = min(0.5 n, 100) and k = 10.
#
# No retraining: TAGCN K=3 and GCN (TSP-20, k = 10, best.pt, seeds 0-2), scripts/scale_eval.py on
# TSP-50/100/200/500/1000, greedy and beam B=10, each + 2-opt. The 2-opt lists always use k = min(0.5 n, 100)
# (--two-opt-k-mode khalf100), so only the network's graph changes; the k = min(0.5 n, 100) results are the
# existing ones in results/phase2_scale/<run>/khalf100/ (not re-run, not changed).
#   k02n (k = 0.2 n) and k10 (k = 10) at every size; khalf (uncapped 0.5 n) at 500 and 1000, where ours is capped.
# Outputs: results/phase2_k02n/<run>/<k-mode>/<set>.{json,npz}; report results/phase2_k02n_report.md (copied to docs/).
# Log: results/phase2_k02n.log (START / OK / FAILED / SKIP lines, then "ALL DONE").
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s k02n "nohup bash scripts/run_phase2_k02n.sh > results/phase2_k02n.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=2
OUT=results/phase2_k02n
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR"
LOG=results/phase2_k02n.log
RUNS=()
for m in tagcn_k3 gcn; do for s in 0 1 2; do RUNS+=("${m}_seed${s}"); done; done

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

log "=== phase 2 k = 0.2 n ablation — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
for n in 50 100 200 500 1000; do
    modes=(k02n k10)
    [ "$n" -ge 500 ] && modes+=(khalf)
    for mode in "${modes[@]}"; do
        run "tsp${n} $mode" "$LOGDIR/tsp${n}_$mode.log" python -u scripts/scale_eval.py --data "data/tsp${n}_test.npz" \
            --k-mode "$mode" --two-opt-k-mode khalf100 --runs "${RUNS[@]}" --out-root "$OUT" --beam100-n
    done
done

if run "report" "$LOGDIR/report.log" python scripts/phase2_k02n_report.py; then
    cp results/phase2_k02n_report.md docs/phase2_k02n_report.md && log "copied to docs/phase2_k02n_report.md"
fi
FAILURES=$(find "$FAILDIR" -type f | wc -l)
log "ALL DONE ($FAILURES failures)"
