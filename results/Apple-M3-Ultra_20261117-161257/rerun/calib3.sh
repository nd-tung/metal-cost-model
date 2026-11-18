#!/usr/bin/env bash
# Three rounds of the M3 Ultra's calibration: every kind of run analysis/model_runs.json builds the
# M3 model from, with the settings those runs recorded in their source.txt. Per round:
#   - the full round, scripts/run_calibration.sh: filter, agg, contention (THREADS 1-24, MB=2048), sort, join
#   - register aggregation, scripts/run_calib_reg.sh, COLS=2 and COLS=7
#   - the concurrent CPU build: part, build only, 1-24 P threads (COOLDOWN=10)
#   - the cache-state probe: calib_filter on SF10 l_quantity in 8 cache states
# A failing sweep is reported and the rounds go on. Run under caffeinate (the machine sleeps after 1 min idle).
set -uo pipefail
cd ~/Documents/metal-cost-model
export PATH=/opt/homebrew/bin:$PATH
export MCM_FLUSH_MB=512 THREADS="1 2 4 8 16 24" MB=2048
DATA=data/tpch-sf10
status=0
run() {
  echo "== $(date '+%F %T') start: $*"
  "$@"
  local rc=$?
  echo "== $(date '+%F %T') exit $rc: $*"
  [[ $rc -eq 0 ]] || status=1
  sleep 30
}
echo "== $(date '+%F %T') machine: $(sysctl -n machdep.cpu.brand_string) | flush $MCM_FLUSH_MB MB | load $(uptime | sed 's/.*load average[s]*: //') | $(pmset -g batt | head -1)"
for round in 1 2 3; do
  echo "== $(date '+%F %T') round $round"
  run scripts/run_calibration.sh
  run scripts/run_calib_reg.sh
  run env COLS=7 scripts/run_calib_reg.sh
  run env JOINS=part PHASES=build DEVICES= BUILD_THREADS="1 2 4 8 12 16 24" COOLDOWN=10 scripts/run_calib_join.sh
  run env CACHE="cold hot cpu-write cpu-read gpu-write gpu-read cpu-touch1 gpu-touch1" COLUMN="$DATA/l_quantity.u32" \
      scripts/run_calib_filter.sh
done
echo "== $(date '+%F %T') all rounds done (status $status)"
exit $status
