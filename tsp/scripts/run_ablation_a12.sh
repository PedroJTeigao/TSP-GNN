#!/usr/bin/env bash
# Ablations A1 (window memo) and A2 (repeated timings), 2026-11-03: see docs/ablations_2026-11-03.md.
#
# ONE process at a time and nothing else of ours running, because the times are the result. CPU only
# (CUDA_VISIBLE_DEVICES=""): the network of the triage scores runs on the CPU with 2 torch threads.
# Before every measurement (one call of triage_ablation.py or lkh3_equal_time.py) the 1-minute load is read; above
# LOAD_MAX the run waits, up to WAIT_MAX seconds, then runs anyway and is marked "loaded" in the load table.
#
# Blocks, in this order (a finished measurement is skipped, so a rerun resumes):
#   A  w = 20, Fu-1k (128) and Fu-10k (16), TAGCN / rng seed 0, memo none / cache / exclude:
#      LKH on every window, random, batched disagreement, batched learned, top 10 / 20 / 50 %   (rep 0)
#   D  anchors: LKH-3 QUADRANT 5 from LKH's own tour on the whole instance at the four E1 budgets, 1 process
#   B  w = 20, both sets, seeds 1 and 2: random, disagreement, learned, memo none / exclude
#   C  w = 50, Fu-10k, seed 0, memo none / cache / exclude, the same four policies
#   E  reps 1-4 of the decisive points: Fu-10k, w = 20, seed 0: LKH on every window (none, cache), random and
#      learned (none)
# Pause: touch results/ablations/PAUSE holds the run between measurements; rm it to resume.
# Outputs: results/ablations/a1/w<w>/<folder>/<set>.{json,npz}, results/ablations/a2/e1_rep<r>/<set>.{json,npz}.
# Load table: results/ablations/a12_load.tsv. Log: results/ablations/a12.log ("ALL DONE a12" at the end).
#
# Launch from tsp/ (survives closing SSH):
#   tmux new-session -d -s abl_a12 "nohup bash scripts/run_ablation_a12.sh > results/ablations/a12.nohup 2>&1"

set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")/.."
if [ "$(nice)" -lt 10 ]; then
    exec nice -n $((10 - $(nice))) bash "$SCRIPT" "$@"
fi
source ../env.sh
export MPLCONFIGDIR="$PWD/../tmp/mplconfig" XDG_CACHE_HOME="$PWD/../tmp/cache" GIT_CONFIG_GLOBAL=/dev/null
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
OUT=results/ablations
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
LOG=$OUT/a12.log
LOAD=$OUT/a12_load.tsv
LOAD_MAX=4
WAIT_MAX=1800
FAILS=0
BENCH=results/phase3_benchmark/data
declare -A SET=([fu1k]=$BENCH/tsp1000_test_concorde.npz [fu10k]=$BENCH/tsp10000_test_concorde.npz)
LEARNED20=results/phase3_learned_triage/models
LEARNED50=results/phase3_triage_wsize/models/w50

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
[ -f $LOAD ] || printf "time\tlabel\tload1_before\twaited_s\tloaded\tload1_after\tseconds\n" > $LOAD

# measure LABEL OUTFILE command...: wait for a quiet machine, run, record load and time. Skips if OUTFILE exists.
measure() {
    local label=$1 outfile=$2; shift 2
    if [ -f "$outfile" ]; then log "SKIP  $label"; return 0; fi
    # $OUT/PAUSE: hold between measurements while other work of ours runs (it is not counted as waiting for load)
    if [ -f "$OUT/PAUSE" ]; then
        log "PAUSE before $label"
        while [ -f "$OUT/PAUSE" ]; do sleep 30; done
        sleep 60                                       # let the 1-minute load forget our other work
        log "RESUME"
    fi
    local waited=0 l
    l=$(cut -d' ' -f1 /proc/loadavg)
    while awk "BEGIN {exit !($l > $LOAD_MAX)}" && [ $waited -lt $WAIT_MAX ]; do
        sleep 60; waited=$((waited + 60)); l=$(cut -d' ' -f1 /proc/loadavg)
    done
    local loaded=0
    awk "BEGIN {exit !($l > $LOAD_MAX)}" && loaded=1
    log "START $label (load $l, waited $waited s, loaded $loaded)"
    local start=$SECONDS
    if "$@" > "$LOGDIR/$(echo "$label" | tr ' /' '__').log" 2>&1; then
        log "OK    $label ($(( SECONDS - start )) s)"
    else
        log "FAILED $label (see $LOGDIR/$(echo "$label" | tr ' /' '__').log)"; FAILS=$((FAILS + 1))
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$label" "$l" "$waited" "$loaded" \
        "$(cut -d' ' -f1 /proc/loadavg)" "$(( SECONDS - start ))" >> $LOAD
}

# pol W SET MEMO REP POLICY SEED: one triage measurement. POLICY = all | random | dis | learned.
pol() {
    local w=$1 set=$2 memo=$3 rep=$4 policy=$5 seed=$6
    local data=${SET[$set]} name folder args
    name=$(basename "$data" .npz)
    local suffix=""; [ "$memo" != none ] && suffix="_memo$memo"; [ "$rep" != 0 ] && suffix="${suffix}_rep$rep"
    local repargs=(); [ "$rep" != 0 ] && repargs=(--rep "$rep")
    local model=$LEARNED20/tagcn_k3_seed$seed/chosen.joblib
    [ "$w" = 50 ] && model=$LEARNED50/tagcn_k3_seed$seed/chosen.joblib
    case $policy in
        all)     folder=random_all; args=(--score random --seeds 0 --fracs 100 --folder-suffix _all) ;;
        random)  folder=random_s$seed; args=(--score random --seeds $seed --fracs 10 20 50 --folder-suffix _s$seed) ;;
        dis)     folder=fast_disagreement_seed$seed; args=(--score fast_disagreement --seeds $seed --fracs 10 20 50) ;;
        learned) folder=fast_learned_seed$seed; args=(--score fast_learned --seeds $seed --fracs 10 20 50 --model $model) ;;
    esac
    measure "a1 w$w $set $policy s$seed memo $memo rep $rep" "$OUT/a1/w$w/$folder$suffix/$name.json" \
        python -u scripts/triage_ablation.py --data "$data" --width $w --out-root "$OUT/a1/w$w" --memo "$memo" \
        "${repargs[@]}" "${args[@]}"
}

anchor() {   # REP SET
    local rep=$1 set=$2 n=1000 budgets="0.72 1.25 1.54 1.94"
    [ "$set" = fu10k ] && n=10000 && budgets="8.4 14.2 17.5 21.5"
    local data=data/benchmark/$(basename "${SET[$set]}")
    measure "a2 anchor $set rep $rep" "$OUT/a2/e1_rep$rep/$(basename "$data" .npz).json" \
        python -u scripts/lkh3_equal_time.py --data "$data" --budgets $budgets --variants quadrant5 --starts lkh \
        --procs 1 --prep results/phase3_benchmark/e1_prep/tsp$n.json results/phase3_benchmark/e1_prep/tsp${n}_extra.json \
        --out-root "$OUT/a2/e1_rep$rep"
}

log "=== ablations A1/A2 — commit $(git rev-parse --short HEAD), niceness $(nice), CPU only ==="

# A
for set in fu10k fu1k; do
    for memo in none cache exclude; do
        [ $memo != exclude ] && pol 20 $set $memo 0 all 0        # exclude at 100 % = cache (tests/test_window_memo.py)
        for p in random dis learned; do pol 20 $set $memo 0 $p 0; done
    done
done
# D
for set in fu1k fu10k; do anchor 0 $set; done
# B
for seed in 1 2; do
    for set in fu10k fu1k; do
        for memo in none exclude; do
            for p in random dis learned; do pol 20 $set $memo 0 $p $seed; done
        done
    done
done
# C
for memo in none cache exclude; do
    [ $memo != exclude ] && pol 50 fu10k $memo 0 all 0
    for p in random dis learned; do pol 50 fu10k $memo 0 $p 0; done
done
# E
for rep in 1 2 3 4; do
    pol 20 fu10k none $rep all 0
    pol 20 fu10k cache $rep all 0
    pol 20 fu10k none $rep random 0
    pol 20 fu10k none $rep learned 0
done

log "ALL DONE a12 ($FAILS failures)"
