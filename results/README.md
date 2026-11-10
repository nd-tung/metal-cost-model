# Results index

One directory per sweep: `<chip>_<date>-<time>[_<op>]/` holding the raw CSV, `machine.json`, and the
outputs of `analysis/fit.py` (`fit.json`, `medians.csv`, `crossover.pdf`). Sweeps started after the
audit of 2026-11-09 also hold `source.txt` (commit, dirty flag, settings) and, when the source
differed from the commit, `source.tar.gz` with the exact source the binaries were built from.
Earlier runs record no source state; the table says which code produced them.

`models/<machine>.json` (`analysis/model.py`) is built from the runs listed in
`analysis/model_runs.json`; list a run there only once this table marks it valid.

`fit.json` gives two crossover estimates per GPU/CPU pair. Quote `report` (measured from the medians);
the fitted closed form is only usable where its `fitted_status` is `ok`.

| Run | Operator, data | Code | Status |
|---|---|---|---|
| `Apple-M1_20261109-1140` | filter, SF1 `l_quantity < 24` | before the cache flush and the E-core fix | **Invalid.** CPU timings are cache-hot (CPU-E 97 GB/s, above the 68 GB/s DRAM peak), and the "E" runs ran on P-cores. |
| `Apple-M1_20261109-1145` | filter, SF10 | cache flush, before the E-core fix | **CPU-E invalid** (ran on P-cores). GPU and CPU-P usable, superseded by 1148 onwards. |
| `Apple-M1_20261109-1148`, `-1204`, `-1206`, `-1208` | filter, SF10 | cache flush + E-core fix, sync GPU dispatch only | Valid. GPU alpha is a synchronous round trip to an idle GPU (see README). |
| `Apple-M1_20261109-1223_agg`, `-1250_agg`, `-1259_agg` | SUM and GROUP BY `l_extendedprice` by 5 keys, SF10 | as above, sync GPU dispatch only | Valid. The CPU aggregation table is flushed before every rep (pessimistic for the CPU at G = 2M); linear fits are poor at G = 100k and 2M. |
| `Apple-M1_20261109-1317_contention` | CPU and GPU streaming filter at the same time, random data | before the start handshake | Valid. Rep 0 of a CPU+GPU config may start up to 6% off (fixed later); medians unaffected. `contention.json` from `analysis/contention.py`. |
| `Apple-M1_20261110-100812_contention`, `-101455_contention`, `-102138_contention` | as above | `source.tar.gz` (with the start handshake) | Valid. `contention.json` from `analysis/contention.py`. The GPU-solo config at the end of each sweep is 3–9% slower than the one at its start (thermal drift). |
| `Apple-M1_20261110-104311_q1`, `-104511_q1`, `-104713_q1` | Q1 per morsel (held out), SF10: GPU sync + pipelined, CPU P, CPU E | `source.tar.gz` (before `--gpu-spin-ms`) | Valid. The GPU starts each rep from idle: mid-size morsels run up to 2× slower than in the SPIN=20 runs (README, GPU clock state). |
| `Apple-M1_20261110-105332_q1`, `-105548_q1`, `-105807_q1` | as above, `SPIN=20` | `source.tar.gz` | Valid. GPU kept busy for 20 ms before each timed rep (CPU rows unaffected). `predict_q1.json` in `-104311_q1` covers all six Q1 runs. |
| `Apple-M1_20261109-134819` | filter, SF10, sync + pipelined GPU dispatch | `source.tar.gz` (first pipelined version) | Sync GPU and CPU rows valid (match 1148–1208 within run-to-run spread). **Pipelined rows superseded:** they used the spacing of GPU completion times, which understates the per-morsel cost when command buffers overlap. |
| `Apple-M1_20261109-135114_agg` | SUM and GROUP BY, SF10, sync + pipelined | `source.tar.gz` (first pipelined version) | Sync GPU and CPU-P rows valid; **pipelined rows superseded** (as above; two cells implied 73–75 GB/s, above the 68 GB/s peak). G = 2M sync GPU is 1.7x faster than in 1223–1259 (28.6 vs 48.8 ms at 2^25): unexplained, see notes below. CPU-E 1.1–1.7x slower than in 1223–1259 (E-core clock). |
| `Apple-M1_20261110-092608` | filter, SF10, pipelined GPU (warm, cold) + CPU | `source.tar.gz` (span / K metric) | Valid. Use for pipelined filter numbers. |
| `Apple-M1_20261110-092820_agg` | SUM and GROUP BY, SF10, pipelined GPU (warm, cold) + CPU | `source.tar.gz` (span / K metric) | Valid. Use for pipelined agg numbers. The sweep script was edited while it ran (exit 127 after the last config); all 20 configs are complete and the binaries were rebuilt only after the sweep. |
| `Apple-M1_20261110-093904` | filter, SF10, sync GPU: shared (warm, cold), private + blit (warm, cold, resident), CPU | `source.tar.gz` (first `--storage private` version) | Valid. Use for private-storage numbers (configs `-priv`). |
| `Apple-M1_20261110-094141_agg` | SUM and GROUP BY, SF10, same configurations | as above | Valid. At G = 2M the GPU never beats the P-core in this run (1.28x at 2^25); see the open question below. |

## Open questions

- **G = 2M (`l_partkey`) GPU timing is bimodal.** Sync, warm, at 2^25: ~48–54 ms (1223, 1250, 1259, a
  recheck) or ~29–31 ms (135114, a second recheck right after the first). The mode changes between
  consecutive invocations and even between sizes within one, so it likely depends on where the buffers
  land in memory. The P-core takes ~42–44 ms, so at G = 2M GPU and P-core are a tie with high variance.

Removed: `Apple-M1_20261109-1242_agg` and `-1243_agg` (two sweeps ran at the same time and
contaminated each other; moved to the Trash, never committed).
