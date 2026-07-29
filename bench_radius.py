"""
Compare radius shaping families on a fixed restart budget.

Usage:  python3 bench_radius.py ROW [RESTARTS]
"""

import ast
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import maqaoa_core as M
import radius_search as R

CSV = "MaxCutMAQAOAData.csv"
OUT = "bench_radius.json"

CONFIGS = [
    ("const", 0.0),
    ("linear", 0.5),
    ("linear", 2.0),
    ("power", 1.0),
    ("power", 3.0),
    ("exp", 0.5),
    ("exp", 1.5),
]

# Load one graph and return everything the benchmark needs, including the true
# lowest shell radius taken from the cached survey so hits can be scored.
def load_row(row):
    df = pd.read_csv(CSV)
    edges = ast.literal_eval(df.loc[row, "Edges"])
    n = int(df.loc[row, "Number of Nodes"])
    energy, _, grad, D = M.make_energy(n, edges, p=1)
    z = np.load("shell_row%d.npz" % row)
    floor = float(z["floor"])
    r_shell = min(R.radius(x) for x in z["shell"])
    return edges, n, energy, grad, D, floor, r_shell

# Run a fixed number of restarts under one shaping configuration and score the results.
def run_config(energy, grad, D, floor, r_shell, name, a, restarts, seed, frac_min=None, frac_a=None):
    cnt = R.Counter(energy, grad)
    rng = np.random.default_rng(seed)
    Rcap = R.full_radius(D)
    shape, dshape = R.shape_family(name, a)
    fs = None if frac_a is None else (lambda t: np.exp(frac_a * t))
    t0 = time.time()
    hits = []
    # Run the optimization for each restart.
    for _ in range(restarts):
        x0 = R.sample_start(rng, D, Rcap)
        if name == "const" and frac_min is None and frac_a is None:
            shaped = None
        else:
            shaped = R.make_shaped_energy(cnt.energy, cnt.grad, floor, Rcap, shape, dshape, frac_min=frac_min, frac_shape=fs)
        got = R.one_run(cnt, D, Rcap, floor, x0, shaped=shaped)
        if got is not None:
            hits.append(got[0])
    secs = time.time() - t0
    if not hits:
        return {"shape": name, "a": a, "frac_min": frac_min, "frac_a": frac_a,
                "hits": 0, "shell_hits": 0, "evals": cnt.total(), "secs": round(secs, 1)}
    # Score the hits against the known shell radius and report statistics.
    radii = np.array([R.radius(x) for x in hits])
    fracs = np.array([R.positive_fraction(x) for x in hits])
    shell = radii < r_shell * (1.0 + 1e-3)
    n_shell = int(shell.sum())
    distinct = len(R.dedupe([hits[i] for i in range(len(hits)) if shell[i]]))
    return {"shape": name, "a": a, "frac_min": frac_min, "frac_a": frac_a,
            "hits": len(hits), "shell_hits": n_shell, "distinct_shell": distinct,
            "min_r": round(float(radii.min()), 6), "mean_r": round(float(radii.mean()), 4),
            "rel_excess": round(float(radii.min() / r_shell - 1.0), 8),
            "max_frac": round(float(fracs.max()), 5),
            "evals": cnt.total(),
            "evals_per_shell": None if n_shell == 0 else int(cnt.total() / n_shell),
            "secs": round(secs, 1)}

def main():
    row = int(sys.argv[1])
    restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    edges, n, energy, grad, D, floor, r_shell = load_row(row)
    print("row %d  n=%d  D=%d  floor=%s  shell radius=%.6f  restarts=%d"
          % (row, n, D, floor, r_shell, restarts))
    print("%-8s %5s %5s %6s %6s %9s %9s %9s %8s %6s"
          % ("shape", "a", "hits", "shell", "distin", "min_r", "mean_r", "ev/shell", "evals", "secs"))
    results = []
    for name, a in CONFIGS:
        d = run_config(energy, grad, D, floor, r_shell, name, a, restarts, seed=100 + row)
        d["row"] = row
        d["restarts"] = restarts
        results.append(d)
        print("%-8s %5.1f %5d %6d %6s %9.4f %9.4f %9s %8d %6.1f"
              % (name, a, d["hits"], d["shell_hits"], d.get("distinct_shell", "-"),
                 d.get("min_r", float("nan")), d.get("mean_r", float("nan")),
                 d.get("evals_per_shell", "-"), d["evals"], d["secs"]))
    old = json.load(open(OUT)) if os.path.exists(OUT) else []
    old = [r for r in old if r.get("row") != row]
    json.dump(old + results, open(OUT, "w"), indent=1)
 
 
if __name__ == "__main__":
    main()