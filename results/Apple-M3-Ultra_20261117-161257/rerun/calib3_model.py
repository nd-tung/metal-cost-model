"""Rerun calibration: build an M3 model from a chosen set of runs and compare it with the current model.

  calib3_model.py manifest <out manifest> <run dir>...   # classify the new runs (source.txt) into a manifest
  calib3_model.py build <manifest> <out model json>       # analysis/model.py on it; models/Apple-M3-Ultra.json kept
  calib3_model.py compare <old model json> <new model json>

Run from the project root with the project's venv.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
MACHINE = "Apple-M3-Ultra"


def settings(run):
    txt = (ROOT / "results" / run / "source.txt").read_text()
    m = re.search(r"^settings: (.*)$", txt, re.M)
    return m.group(1) if m else ""


def classify(run):
    s = settings(run)
    if run.endswith("_contention"):
        return "contention", "contention"
    if run.endswith("_cache"):
        return "cache_states", "cache-state probe; SF10 l_quantity"
    if run.endswith("_reg"):
        return "calibration", "register aggregation" + (", 7 columns (COLS=7)" if "COLS='7'" in s else "") + "; GPU only"
    if run.endswith("_join") and "BUILD_THREADS='1" in s:
        return "calibration", "concurrent CPU build: part, calib_join --threads 1 2 4 8 12 16 24 (cpu-p-t<k>)"
    return "calibration", ""


def cmd_manifest(out, runs):
    base = json.loads((ROOT / "analysis" / "model_runs.json").read_text())
    spec = {k: v for k, v in base[MACHINE].items() if k not in ("calibration", "contention", "cache_states")}
    spec.update(calibration=[], contention=[], cache_states=[])
    count = {}
    for run in sorted(runs):
        kind, what = classify(run)
        tag = (kind, what, run.split("_", 2)[2] if run.count("_") >= 2 else "filter")  # rounds counted per sweep kind
        count[tag] = count.get(tag, 0) + 1
        note = ", ".join(x for x in [what, f"rerun round {count[tag]}", "GPU kept busy (SPIN=20)", "flush 512 MB"] if x)
        if kind == "contention":
            spec["contention"].append(run)
        else:
            spec[kind].append({"run": run, "keep": "all", "note": note})
    Path(out).write_text(json.dumps({"_about": base["_about"], MACHINE: spec}, indent=1))
    for k in ("calibration", "contention", "cache_states"):
        print(k, len(spec[k]))


def cmd_build(manifest, out):
    cur = ROOT / "models" / f"{MACHINE}.json"
    keep = cur.with_suffix(".json.keep")
    shutil.copy2(cur, keep)
    try:
        subprocess.run([sys.executable, "analysis/model.py", "--manifest", manifest, "--machine", MACHINE], check=True)
        shutil.move(cur, out)
    finally:
        shutil.move(keep, cur)  # the current model, untouched
    print("->", out)


def coeffs(m):
    rows = {}
    for op, o in m["ops"].items():
        for cfg, c in o["configs"].items():
            r = {"alpha_us": c.get("alpha_us"), "beta_ns": c.get("beta_ns_per_row"), "bw_GBps": c.get("eff_bw_GBps")}
            if "large_n" in c:
                r["large_bw_GBps"] = c["large_n"].get("eff_bw_GBps")
            if "overlap" in c and isinstance(c["overlap"], dict):
                r["k"] = c["overlap"].get("k")
            rows[(op, cfg)] = r
    return rows


def cmd_compare(old_path, new_path):
    old, new = json.loads(Path(old_path).read_text()), json.loads(Path(new_path).read_text())
    co, cn = coeffs(old), coeffs(new)
    keys = sorted(set(co) | set(cn))
    moved = {"alpha_us": [], "beta_ns": [], "large_bw_GBps": [], "k": []}
    print(f"{'op':34s} {'config':24s} {'alpha old':>10s} {'new':>9s} {'beta ns old':>12s} {'new':>9s} {'large GB/s old':>15s} {'new':>8s} {'k old':>6s} {'new':>6s}")
    for k in keys:
        a, b = co.get(k), cn.get(k)
        if a is None or b is None:
            print(f"{k[0]:34s} {k[1]:24s} only in {'new' if a is None else 'old'}")
            continue

        def f(x, w=9, p=3):
            return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"
        print(f"{k[0]:34s} {k[1]:24s} {f(a['alpha_us'], 10)} {f(b['alpha_us'])} {f(a['beta_ns'], 12, 4)} {f(b['beta_ns'], 9, 4)} "
              f"{f(a.get('large_bw_GBps'), 15, 1)} {f(b.get('large_bw_GBps'), 8, 1)} {f(a.get('k'), 6, 3)} {f(b.get('k'), 6, 3)}")
        for q in moved:
            x, y = a.get(q), b.get(q)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and abs(x) > 1e-9:
                moved[q].append((abs(y / x - 1), k))
    print()
    for q, v in moved.items():
        if not v:
            continue
        v.sort(reverse=True)
        n5, n10 = sum(e > 0.05 for e, _ in v), sum(e > 0.10 for e, _ in v)
        print(f"{q}: {len(v)} coefficients, {n5} moved > 5%, {n10} > 10%; largest: "
              + "; ".join(f"{k[0]} {k[1]} {e:+.0%}" for e, k in v[:4]))
    oc, nc = old["contention"], new["contention"]
    print()
    print(f"contention: B_max {oc['B_max_GBps']:.1f} -> {nc['B_max_GBps']:.1f} GB/s, w_gpu {oc['w_gpu']:.3f} -> {nc['w_gpu']:.3f}, "
          f"GPU solo {oc['gpu_solo_GBps']:.1f} -> {nc['gpu_solo_GBps']:.1f} GB/s")
    for q in ("p", "e"):
        print(f"  CPU {q} solo GB/s by threads: " + ", ".join(
            f"{t}: {oc['cpu_solo_GBps'][q][t]:.1f} -> {nc['cpu_solo_GBps'][q].get(t, float('nan')):.1f}" for t in oc["cpu_solo_GBps"][q]))
    ro, rn = old.get("register_agg", {}).get("configs", {}), new.get("register_agg", {}).get("configs", {})
    for cfg in sorted(set(ro) & set(rn)):
        print(f"register_agg {cfg}: c_us_per_update {ro[cfg]['c_us_per_update']:.3e} -> {rn[cfg]['c_us_per_update']:.3e}")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"manifest": lambda: cmd_manifest(args[0], args[1:]), "build": lambda: cmd_build(*args),
     "compare": lambda: cmd_compare(*args)}[cmd]()
