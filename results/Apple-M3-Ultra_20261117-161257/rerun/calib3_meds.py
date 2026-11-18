"""Measured medians, old calibration rounds vs the rerun: per (op key | config, n), the median over each set's
runs of the per-run medians (analysis/model.py run_medians), their ratio new/old, and each set's own
run-to-run spread (max / min of its runs' medians).

  calib3_meds.py <old manifest> <new manifest> [csv out]

Run from the project root with the project's venv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd() / "analysis"))
import model as mdl  # noqa: E402

MACHINE = "Apple-M3-Ultra"


def pooled(meds):
    g = meds.groupby(["config", "n"])["t_med"]
    return pd.DataFrame({"t": g.median(), "spread": g.max() / g.min(), "runs": g.count()})


def family(cfg):
    op, dev = cfg.split("|", 1)
    fam = op.split("/", 1)[0]
    if dev.startswith("gpu"):
        d = "gpu-pipe" if "-pipe" in dev else "gpu-sync"
    else:
        d = "cpu-p" if dev.startswith("cpu-p") else "cpu-e"
    return fam, d


def compare(old_entries, new_entries, states, label, out=None):
    mo, *_ = mdl.run_medians(old_entries, states=states)
    mn, *_ = mdl.run_medians(new_entries, states=states)
    po, pn = pooled(mo), pooled(mn)
    j = po.join(pn, lsuffix="_old", rsuffix="_new", how="outer")
    missing = j[j["t_old"].isna() | j["t_new"].isna()]
    j = j.dropna(subset=["t_old", "t_new"]).copy()
    j["ratio"] = j["t_new"] / j["t_old"]
    j["dev"] = (j["ratio"] - 1).abs()
    # is the new set's median inside the old runs' range, and vice versa?
    rng_o = mo.groupby(["config", "n"])["t_med"].agg(["min", "max"])
    rng_n = mn.groupby(["config", "n"])["t_med"].agg(["min", "max"])
    j["new_in_old_range"] = (j["t_new"] >= rng_o["min"].reindex(j.index)) & (j["t_new"] <= rng_o["max"].reindex(j.index))
    j["old_in_new_range"] = (j["t_old"] >= rng_n["min"].reindex(j.index)) & (j["t_old"] <= rng_n["max"].reindex(j.index))
    fams = [family(c) for c, _ in j.index]
    j["fam"], j["devc"] = [f for f, _ in fams], [d for _, d in fams]

    print(f"\n=== {label}: {len(j)} (config, n) cells, {len(missing)} only in one set")
    q = j["dev"]
    print(f"|new/old - 1|: median {q.median():.1%}, p90 {q.quantile(.9):.1%}, p99 {q.quantile(.99):.1%}, max {q.max():.1%}; "
          f"within 5%: {(q <= .05).mean():.0%}, within 10%: {(q <= .10).mean():.0%}")
    print(f"run-to-run spread (max/min - 1) median: old {(j['spread_old'] - 1).median():.1%}, new {(j['spread_new'] - 1).median():.1%}; "
          f"p90: old {(j['spread_old'] - 1).quantile(.9):.1%}, new {(j['spread_new'] - 1).quantile(.9):.1%}")
    both = j["new_in_old_range"] | j["old_in_new_range"]
    print(f"new median inside the old runs' range or old inside the new: {both.mean():.0%} of cells")
    t = j.groupby(["fam", "devc"]).agg(cells=("dev", "size"), med=("dev", "median"), p90=("dev", lambda x: x.quantile(.9)),
                                       mx=("dev", "max"), ratio_med=("ratio", "median"),
                                       spread_old=("spread_old", lambda x: (x - 1).median()),
                                       spread_new=("spread_new", lambda x: (x - 1).median()))
    print("\nby op family and device (|new/old - 1|; ratio_med: median new/old; spread: median max/min - 1 within a set)")
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    worst = j[~both].sort_values("dev", ascending=False).head(20)
    print("\nlargest moves outside both sets' run ranges (t in us)")
    for (cfg, n), r in worst.iterrows():
        ro = mo[(mo.config == cfg) & (mo.n == n)].sort_values("run")["t_med"].round(1).tolist()
        rn = mn[(mn.config == cfg) & (mn.n == n)].sort_values("run")["t_med"].round(1).tolist()
        print(f"  {cfg:52s} 2^{int(np.log2(n)):<2d} old {r.t_old:10.1f} {ro}  new {r.t_new:10.1f} {rn}  x{r.ratio:.2f}")
    if out:
        j.to_csv(out)
    return j


def main():
    old = json.loads(Path(sys.argv[1]).read_text())[MACHINE]
    new = json.loads(Path(sys.argv[2]).read_text())[MACHINE]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    compare(old["calibration"], new["calibration"], False, "calibration (cold)", out)
    compare(old["cache_states"], new["cache_states"], True, "cache states",
            out.replace(".csv", "_states.csv") if out else None)


if __name__ == "__main__":
    main()
