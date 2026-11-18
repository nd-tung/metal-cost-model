#!/usr/bin/env bash
# The router and the static split at Q14 with the GPU's per-morsel time measured under run_query's ring
# against the plan's (results/Apple-M3-Ultra_20261117-111222_e2e/plan_q14_*.sh), interleaved, two rounds.
# The ring time: the GPU alone, 16 in flight, the span from its first start to its last end per command
# buffer, median over 5 reps, measured first in this session (t_gpu_ring.txt). The CPU alone for the best
# static placement. Every run writes the per-command-buffer timeline; run_query's output goes to log.txt.
set -euo pipefail
cd ~/Documents/metal-cost-model
export PATH=/opt/homebrew/bin:$PATH
export MCM_FLUSH_MB=512
source scripts/lib_sweep.sh
sweep_begin ringrouter
cp "$0" "$dir/ring_router.sh"
pmset -g batt | head -1 >> "$dir/source.txt"
PLANS=results/Apple-M3-Ultra_20261117-111222_e2e
SIZES="16 18 20 22"
RQ=(--data data/tpch-sf10 --reps 5 --gpu-spin-ms 20 --inflight 16 --out "$dir/run_query.csv" --gpu-timeline "$dir/timeline.csv")
W=(--p-workers 24 --e-workers 8)
run() {
  echo "## $*" >> "$dir/log.txt"
  build/run_query --query q14 "$@" "${RQ[@]}" >> "$dir/log.txt" 2>&1 || { echo "run_query failed: $*"; exit 1; }
  echo "ok: $*"
  sleep 3
}
for s in $SIZES; do cp "$PLANS/plan_q14_$s.sh" "$dir/"; done
# 1. the GPU alone under the ring
for s in $SIZES; do run --policy gpu --morsel-log2 "$s" --build-device gpu --machine-json "$dir/machine.json"; done
/usr/bin/python3 - "$dir/timeline.csv" > "$dir/t_gpu_ring.txt" <<'EOF'
import csv, statistics, sys
from collections import defaultdict
span = defaultdict(lambda: [float("inf"), float("-inf"), 0])
for r in csv.DictReader(open(sys.argv[1])):
    if r["policy"] != "gpu":
        continue
    v = span[(int(r["n"]), int(r["rep"]))]
    v[0] = min(v[0], float(r["gpu_start_us"]))
    v[1] = max(v[1], float(r["gpu_end_us"]))
    v[2] += 1
per = defaultdict(list)
for (n, rep), (s, e, k) in span.items():
    per[n].append((e - s) / k)
for n in sorted(per):
    print(n.bit_length() - 1, f"{statistics.median(per[n]):.3f}")
EOF
cat "$dir/t_gpu_ring.txt"
T_RING=()
while read -r s t; do T_RING[$s]=$t; done < "$dir/t_gpu_ring.txt"
sweep_settings "REPS=5 SPIN=20 INFLIGHT=16 P_WORKERS=24 E_WORKERS=8 COOLDOWN=3 plans=$PLANS" \
               "t_gpu_ring_us(2^16 2^18 2^20 2^22)='${T_RING[16]} ${T_RING[18]} ${T_RING[20]} ${T_RING[22]}'"
# 2. the CPU alone, then the router and the split with the plan's and the ring's GPU time, interleaved
for s in $SIZES; do
  eval "$(cat "$PLANS/plan_q14_$s.sh")"
  run --policy cpu --morsel-log2 "$s" --build-device cpu "${W[@]}" --pred-ms "$PRED_CPU_MS"
  for round in 1 2; do
    for v in plan ring; do
      if [[ $v == plan ]]; then t=$T_GPU; pred=$PRED_BOTH_GPUBUILD_MS; else t=${T_RING[$s]}; pred=0; fi
      run --policy router --morsel-log2 "$s" --build-device gpu "${W[@]}" --t-gpu "$t" --t-p "$T_P" --t-e "$T_E" --pred-ms "$pred"
      run --policy split --morsel-log2 "$s" --build-device gpu "${W[@]}" --t-gpu "$t" --t-p "$T_P" --t-e "$T_E" --pred-ms "$pred"
    done
  done
done
echo "done -> $dir"
