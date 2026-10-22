#!/usr/bin/env bash
# Phase 3: window size vs. solver — where does the model pay off against LKH? (scripts/wsize_eval.py)
#
# 1. Data (CPU, 8 LKH processes): data/tsp100_train.npz (12000, seed 19) and data/tsp100_val.npz (500, seed 20);
#    at the same time on the GPU: TAGCN K=3 on TSP-50 (tsp50_train), seeds 0-2 -> results/runs/tagcn_k3_tsp50_seed<s>/
# 2. TAGCN K=3 on TSP-100, seeds 0-2, one after the other -> results/runs/tagcn_k3_tsp100_seed<s>/;
#    at the same time, window refinement on tsp1000_test and tsp10000_test (2 passes, beam B=10):
#      LKH, w = 20 50 100 200 300                      -> results/phase3_wsize/lkh/
#      TSP-20 model (tagcn_k3_seed<s>), all w          -> results/phase3_wsize/tagcn_k3_seed<s>/
#      TSP-50 model, w = 50                            -> results/phase3_wsize/tagcn_k3_tsp50_seed<s>/
#      TSP-100 model, w = 100 200 300 (once trained)   -> results/phase3_wsize/tagcn_k3_tsp100_seed<s>/
# Training follows phase 1 (20 epochs, Adam 1e-3 cosine, batch 64, best val gap) with k = 0.5 n (--k-frac 0.5),
# the k = round(0.5 m) the windows use (for TSP-20 this is the k = 10 of phase 1).
# At the end the report results/phase3_wsize_report.md is generated (if the report script exists) and copied to docs/.
# Progress goes to results/phase3_wsize.log (START / OK / FAILED / SKIP lines, GPU memory every 5 min, "ALL DONE").
#
# Launch (survives closing SSH):
#   tmux new-session -d -s wsize "nohup bash scripts/run_phase3_wsize.sh > results/phase3_wsize.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."

if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi

source ../env.sh   # env.sh and tmp/ live in the repository root
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1
OUT=results/phase3_wsize
LOGDIR=$OUT/logs
FAILDIR=$LOGDIR/failures
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAILDIR" results/figs

LOG=results/phase3_wsize.log
SEEDS=(0 1 2)
SETS=(tsp1000_test tsp10000_test)

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

make_data() {   # make_data <name> <num> <seed>
    if [ -f "data/$1.npz" ]; then log "SKIP  data $1 (already exists)"; return 0; fi
    run "data $1" "$LOGDIR/data_$1.log" python scripts/make_dataset.py --n 100 --num "$2" --seed "$3" --out "data/$1.npz"
}

train() {       # train <run name> <size> <seed>
    local name=$1 n=$2 s=$3
    if [ -f "results/runs/$name/eval.json" ]; then log "SKIP  train $name (already done)"; return 0; fi
    if [ ! -f "data/tsp${n}_train.npz" ] || [ ! -f "data/tsp${n}_val.npz" ]; then
        touch "$FAILDIR/train_$name"; log "FAILED train $name: data/tsp${n}_train/val missing"; return 1
    fi
    mkdir -p "results/runs/$name"
    run "train $name" "results/runs/$name/train.log" python scripts/train_model.py --model tagcn --tag-k 3 \
        --k-frac 0.5 --train "data/tsp${n}_train.npz" --val "data/tsp${n}_val.npz" \
        --test data/tsp50_test.npz data/tsp100_test.npz --seed "$s" --run-name "$name"
}

windows() {     # windows <folder> <solver args...> -- <widths...>: both test sets, one after the other
    local folder=$1; shift
    local args=() widths=()
    while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
    widths=("$@")
    for name in "${SETS[@]}"; do
        if [ -f "$OUT/$folder/$name.json" ]; then log "SKIP  $name $folder (already done)"; continue; fi
        if [ "${args[1]}" = model ] && [ ! -f "results/runs/${args[3]}/best.pt" ]; then
            touch "$FAILDIR/win_${name}_$folder"; log "FAILED $name $folder: results/runs/${args[3]}/best.pt missing"; continue
        fi
        run "$name $folder w=${widths[*]}" "$LOGDIR/${name}_$folder.log" python scripts/wsize_eval.py \
            --data "data/$name.npz" "${args[@]}" --widths "${widths[@]}" --out-root "$OUT"
    done
}

gpu_monitor() {
    while true; do
        log "GPU  $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used in total"
        sleep 300
    done
}

log "=== phase 3 window size vs. solver — commit $(git rev-parse --short HEAD), niceness $(nice) ==="
gpu_monitor &
MON=$!

# ---- 1. data for TSP-100 (CPU) || TSP-50 training (GPU)
( make_data tsp100_train 12000 19; make_data tsp100_val 500 20 ) &
DATA=$!
( for s in "${SEEDS[@]}"; do train "tagcn_k3_tsp50_seed$s" 50 "$s"; done ) &
T50=$!
wait "$DATA" "$T50"

# ---- 2. TSP-100 training (then its windows) || LKH, TSP-20 and TSP-50 windows
PIDS=()
( for s in "${SEEDS[@]}"; do
      train "tagcn_k3_tsp100_seed$s" 100 "$s"
      windows "tagcn_k3_tsp100_seed$s" --solver model --run "tagcn_k3_tsp100_seed$s" -- 100 200 300 &
  done
  wait ) &
PIDS+=($!)
windows lkh --solver lkh -- 20 50 100 200 300 &
PIDS+=($!)
for s in "${SEEDS[@]}"; do
    windows "tagcn_k3_seed$s" --solver model --run "tagcn_k3_seed$s" -- 20 50 100 200 300 &
    PIDS+=($!)
done
( for s in "${SEEDS[@]}"; do windows "tagcn_k3_tsp50_seed$s" --solver model --run "tagcn_k3_tsp50_seed$s" -- 50; done ) &
PIDS+=($!)
wait "${PIDS[@]}"
kill "$MON" 2>/dev/null

if [ -f scripts/phase3_wsize_report.py ]; then
    if run "wsize report" "$LOGDIR/report.log" python scripts/phase3_wsize_report.py; then
        cp results/phase3_wsize_report.md docs/phase3_wsize_report.md && mkdir -p docs/figs \
            && cp results/figs/phase3_wsize_*.png docs/figs/ && log "copied to docs/phase3_wsize_report.md (+ figures)"
    fi
fi

FAILURES=$(find "$FAILDIR" -type f | wc -l)
log "failed steps: $FAILURES"
log "ALL DONE ($FAILURES failures)"
