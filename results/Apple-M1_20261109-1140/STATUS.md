# INVALID — do not use

Produced before the cache flush (`--cache cold`) and the E-core placement fix existed.

- CPU timings read a cache-resident buffer: CPU-E reaches 97 GB/s, above the M1's 68 GB/s DRAM peak.
- The "E" runs ran on P-cores (thread-level `BACKGROUND` QoS does not reach the E-cores).

Kept for the record. Valid filter runs: `Apple-M1_20261109-1148`, `-1204`, `-1206`, `-1208`.
