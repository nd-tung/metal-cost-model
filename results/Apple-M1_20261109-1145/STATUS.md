# CPU-E rows invalid — superseded

Produced with the cache flush but before the E-core placement fix: the "E" runs ran on P-cores
(thread-level `BACKGROUND` QoS does not reach the E-cores), so every `cpu,e` row is P-core data.
GPU and CPU-P rows are usable but superseded by `Apple-M1_20261109-1148`, `-1204`, `-1206`, `-1208`.
