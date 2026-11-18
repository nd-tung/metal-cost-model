#!/usr/bin/env bash
# Does the GPU's driver path need a free core? Q14's router and static split at 2^16, 2^18 and 2^20 with
# the GPU's ring time (results/Apple-M3-Ultra_20261117-155421_ringrouter/t_gpu_ring.txt) and 24 P + 8 E
# workers (every core busy), 23 P + 8 E and 22 P + 8 E (one or two P-cores left free), 24 P + 0 E (the
# E-cores free), interleaved, two rounds; the GPU alone and the CPU alone (24 P + 8 E) once per size.
# The plan's per-worker times from results/Apple-M3-Ultra_20261117-111222_e2e/plan_q14_*.sh.
set -euo pipefail
cd ~/Documents/metal-cost-model
export PATH=/opt/homebrew/bin:$PATH
export MCM_FLUSH_MB=512
source scripts/lib_sweep.sh
sweep_begin freecore
cp "$0" "$dir/free_core.sh"
pmset -g batt | head -1 >> "$dir/source.txt"
PLANS=results/Apple-M3-Ultra_20261117-111222_e2e
RING=results/Apple-M3-Ultra_20261117-155421_ringrouter/t_gpu_ring.txt
cp "$RING" "$dir/t_gpu_ring.txt"
T_RING=()
while read -r s t; do T_RING[$s]=$t; done < "$RING"
sweep_settings "REPS=5 SPIN=20 INFLIGHT=16 COOLDOWN=3 plans=$PLANS ring=$RING CONFIGS='24/8 23/8 22/8 24/0'"
RQ=(--data data/tpch-sf10 --reps 5 --gpu-spin-ms 20 --inflight 16 --out "$dir/run_query.csv" --gpu-timeline "$dir/timeline.csv")
run() {
  echo "## $*" >> "$dir/log.txt"
  build/run_query --query q14 "$@" "${RQ[@]}" >> "$dir/log.txt" 2>&1 || { echo "run_query failed: $*"; exit 1; }
  echo "ok: $*"
  sleep 3
}
for s in 16 18 20; do
  cp "$PLANS/plan_q14_$s.sh" "$dir/"
  eval "$(cat "$PLANS/plan_q14_$s.sh")"
  t=${T_RING[$s]}
  run --policy gpu --morsel-log2 "$s" --build-device gpu --machine-json "$dir/machine.json"
  run --policy cpu --morsel-log2 "$s" --build-device cpu --p-workers 24 --e-workers 8 --pred-ms "$PRED_CPU_MS"
  for round in 1 2; do
    for cfg in "24 8" "23 8" "22 8" "24 0"; do
      set -- $cfg
      for pol in router split; do
        run --policy "$pol" --morsel-log2 "$s" --build-device gpu --p-workers "$1" --e-workers "$2" \
            --t-gpu "$t" --t-p "$T_P" --t-e "$T_E" --pred-ms 0
      done
    done
  done
done
echo "done -> $dir"
