#!/usr/bin/env bash
# Why a pipelined Q14 GPU morsel takes 30-33 us in run_query at 2^20 and ~17-25 us in val_q14: GPU-only Q14
# (and Q6 at 2^20 as the streaming reference) under run_query with a per-command-buffer timeline, varying
# the queue depth (--inflight) and who built the hash table, against val_q14's batch dispatch.
set -euo pipefail
cd ~/Documents/metal-cost-model
export MCM_FLUSH_MB=512
OUT="results/Apple-M3-Ultra_$(date +%Y%m%d-%H%M%S)_gpuiso"
mkdir -p "$OUT"
uptime > "$OUT/uptime.txt"
pmset -g batt | head -1 >> "$OUT/uptime.txt"
RQ=(--data data/tpch-sf10 --reps 5 --gpu-spin-ms 20 --out "$OUT/run_query.csv" --gpu-timeline "$OUT/timeline.csv")
T=(--t-gpu 16.9 --t-p 729.7 --t-e 4387 --p-workers 24 --e-workers 8)
for s in 18 20 22; do
  for inf in 4 16 64; do
    build/run_query --query q14 --policy gpu --morsel-log2 "$s" --inflight "$inf" --build-device gpu "${RQ[@]}" 2>&1 | tail -1
    sleep 3
  done
  # the GPU alone, the table built by the 24 P workers (the CPU side of the split gets no morsel)
  build/run_query --query q14 --policy split --gpu-share 1 --morsel-log2 "$s" --inflight 16 --build-device cpu "${T[@]}" "${RQ[@]}" 2>&1 | tail -1
  sleep 3
  build/run_query --query q14 --policy split --gpu-share 1 --morsel-log2 "$s" --inflight 16 --build-device gpu "${T[@]}" "${RQ[@]}" 2>&1 | tail -1
  sleep 3
done
build/run_query --query q6 --policy gpu --morsel-log2 20 --inflight 16 "${RQ[@]}" 2>&1 | tail -1
sleep 3
for b in 16 58; do
  build/val_q14 --device gpu --dispatch pipelined --batch "$b" --batch-mb 4096 --min-log2 20 --max-log2 20 --reps 5 --gpu-spin-ms 20 \
    --cache cold --out "$OUT/val_q14.csv" 2>&1 | tail -1
  sleep 3
done
build/val_q14 --device gpu --dispatch pipelined --batch 16 --min-log2 18 --max-log2 22 --reps 5 --gpu-spin-ms 20 --cache cold \
  --out "$OUT/val_q14.csv" 2>&1 | tail -1
echo "done -> $OUT"
