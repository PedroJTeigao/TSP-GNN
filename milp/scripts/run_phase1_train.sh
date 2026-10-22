#!/usr/bin/env bash
# FCNF phase 1, step 4: train the arc classifiers, and in parallel re-solve the unproven fcnf100 instances.
#
#   GPU lane 1: GCN,       seeds 0-2, one after the other   -> results/runs/gcn_seed<s>/
#   GPU lane 2: TAGCN K=3, seeds 0-2, one after the other   -> results/runs/tagcn_k3_seed<s>/
#   CPU:        unproven fcnf100_test instances, 600 s, 6 processes -> data/fcnf100_test_ref600.npz
# Progress goes to results/phase1_train.log (START / OK / FAILED / SKIP lines, GPU memory every 5 min, "ALL DONE").
#
# Launch from milp/ (survives closing SSH):
#   tmux new-session -d -s fcnftrain "nohup bash scripts/run_phase1_train.sh > results/phase1_train.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
LOGDIR=results/phase1_train_logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" results/runs
LOG=results/phase1_train.log
SEEDS=(0 1 2)

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

lane() {   # model name prefix extra args...
    local model=$1 prefix=$2; shift 2
    for s in "${SEEDS[@]}"; do
        local name=${prefix}_seed$s
        if [ -f "results/runs/$name/eval.json" ]; then log "SKIP  train $name (already done)"; continue; fi
        mkdir -p "results/runs/$name"
        run "train $name" "results/runs/$name/train.log" python -u scripts/train_model.py --model "$model" "$@" \
            --seed "$s" --run-name "$name"
    done
}

gpu_monitor() {
    while true; do
        sleep 300
        log "GPU $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
    done
}

log "commit $(git rev-parse --short HEAD)"
gpu_monitor &
MONITOR=$!

lane gcn gcn &
P1=$!
lane tagcn tagcn_k3 --tag-k 3 &
P2=$!
if [ -f data/fcnf100_test_ref600.npz ]; then
    log "SKIP  ref600 (already exists)"
else
    run "ref600" "$LOGDIR/ref600.log" python -u scripts/reference_rerun.py --data data/fcnf100_test.npz \
        --time-limit 600 --procs 6 --out data/fcnf100_test_ref600.npz &
    P3=$!
fi
wait $P1 $P2 ${P3:-}
kill $MONITOR 2>/dev/null

if ls "$FAILDIR"/* > /dev/null 2>&1; then
    log "ALL DONE with failures: $(ls "$FAILDIR" | tr '\n' ' ')"
else
    log "ALL DONE (0 failures)"
fi
