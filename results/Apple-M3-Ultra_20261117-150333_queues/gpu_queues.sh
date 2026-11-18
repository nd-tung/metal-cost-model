#!/usr/bin/env bash
# Does the GPU's idle gap between the driver's batches belong to the command queue? Q14 on the GPU alone
# with its command buffers in turn on 1, 2 and 4 command queues (run_query --queues), two rounds
# interleaved; then 64 in flight, Q6, and the router and the static split at Q14 2^18/2^20, on 1 and 2
# queues. Every run_query writes the per-command-buffer timeline.
set -euo pipefail
cd ~/Documents/metal-cost-model
export PATH=/opt/homebrew/bin:$PATH
export MCM_FLUSH_MB=512
source scripts/lib_sweep.sh
sweep_begin queues
cp "$0" "$dir/gpu_queues.sh"
sweep_settings "REPS=5 SPIN=20 INFLIGHT=16 (64 where noted) QUEUES='1 2 4' P_WORKERS=24 E_WORKERS=8 COOLDOWN=3"
pmset -g batt | head -1 >> "$dir/source.txt"
RQ=(--data data/tpch-sf10 --reps 5 --gpu-spin-ms 20 --out "$dir/run_query.csv" --gpu-timeline "$dir/timeline.csv")
W=(--p-workers 24 --e-workers 8)
run() { build/run_query "$@" "${RQ[@]}" 2>&1 | grep -E "GPU on|MISMATCH|error" | head -2; sleep 3; }
run --query q14 --policy gpu --morsel-log2 20 --inflight 16 --queues 1 --machine-json "$dir/machine.json"
for round in 1 2; do
  for s in 18 20 22; do
    for qn in 1 2 4; do
      run --query q14 --policy gpu --morsel-log2 "$s" --inflight 16 --queues "$qn" --build-device gpu
    done
  done
done
for qn in 1 2; do
  run --query q14 --policy gpu --morsel-log2 20 --inflight 64 --queues "$qn" --build-device gpu
  run --query q6 --policy gpu --morsel-log2 20 --inflight 16 --queues "$qn"
done
# the router and the static split with the plan's numbers (results/Apple-M3-Ultra_20261117-111222_e2e/plan_q14_*.sh)
for qn in 1 2; do
  run --query q14 --policy router --morsel-log2 20 --inflight 16 --queues "$qn" --build-device gpu "${W[@]}" \
      --t-gpu 16.896 --t-p 729.703 --t-e 4387.019 --pred-ms 1.401
  run --query q14 --policy split --morsel-log2 20 --inflight 16 --queues "$qn" --build-device gpu "${W[@]}" \
      --t-gpu 16.896 --t-p 729.703 --t-e 4387.019 --pred-ms 1.401
  run --query q14 --policy router --morsel-log2 18 --inflight 16 --queues "$qn" --build-device gpu "${W[@]}" \
      --t-gpu 16.023 --t-p 182.397 --t-e 1099.902 --pred-ms 1.848
done
echo "done -> $dir"
