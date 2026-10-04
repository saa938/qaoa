"""
Usage:  python src/search_compare.py

Runs run_filter.run on the unweighted p=1 graphs with both searches and compares them:
  restarts  random restarts over the whole space (the search run_filter.py has used so far)
  radius    restarts inside a shrinking maximum radius (run_filter.harvest_radius)
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_filter as rf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "search_compare.json"

P = 1
ROWS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
SEED_SHIFTS = [0, 500, 900] # added to the seed run_filter.main uses for the row
MODES = [("restarts", None), ("radius", 100), ("radius", 60)] # (search, restarts per iteration of the radius search)
CALLS = {"g": 0}

MAKE_ENERGY = rf.make_energy

# run_filter.make_energy with every gradient call counted.
def counted_make_energy(n, edges, w=None, p=1):
    energy, energy_batch, grad, D = MAKE_ENERGY(n, edges, w=w, p=p)

    def g(x, with_energy=False):
        CALLS["g"] += 1
        return grad(x, with_energy)

    return energy, energy_batch, g, D

# Are two picks the same point, allowing for the symmetries of the graph?
def same_pick(a, b, images, per):
    if a is None or b is None:
        return False
    b = np.array(b)
    return bool(min(rf.geodesic_dist(y, b, per) for y in images(np.array(a))) <= rf.TOL_D)

def main():
    rf.make_energy = counted_make_energy
    df = pd.read_csv(rf.CSV)
    res = json.load(open(OUT)) if OUT.exists() else {}
    for shift in SEED_SHIFTS:
        for row in ROWS:
            edges = [tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])]
            n = int(df.loc[row, "Number of Nodes"])
            m = len(rf.canonical_edges(edges))
            w = np.ones(m)
            seed = 1000 * row + 10 * P + shift
            for search, per_iter in MODES:
                key = "%d,%d,%s,%s" % (row, shift, search, per_iter)
                if key in res:
                    continue
                rf.SEARCH = search
                if per_iter is not None:
                    rf.LOOP_RESTARTS = {P: per_iter}
                CALLS["g"] = 0
                rec = rf.run(row, edges, n, w, P, seed, False, None)
                rec.update({"grads": CALLS["g"], "per_iter": per_iter, "shift": shift})
                res[key] = rec
                json.dump(res, open(OUT, "w"), indent=1)

    print("\n%4s %5s | %-8s %4s %9s %8s %5s %6s %8s %9s %5s | %s"
          % ("row", "seed", "search", "per", "floor", "r_min", "shell", "filter", "restarts", "grads", "secs", "same floor / radius / pick as restarts"))
    for shift in SEED_SHIFTS:
        for row in ROWS:
            base = res.get("%d,%d,restarts,None" % (row, shift))
            if base is None:
                continue
            edges = [tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])]
            n = int(df.loc[row, "Number of Nodes"])
            m = len(rf.canonical_edges(edges))
            autos, images = rf.symmetry_group(edges, n, np.ones(m), P)
            per = rf.periods(m, n, P, np.ones(m))
            for search, per_iter in MODES:
                d = res.get("%d,%d,%s,%s" % (row, shift, search, per_iter))
                if d is None:
                    continue
                note = ""
                if search == "radius":
                    note = "%s / %s / %s   stop: %s" % (
                        abs(d["floor"] - base["floor"]) <= 1e-5,
                        d["r_min"] is not None and abs(d["r_min"] - base["r_min"]) <= rf.RAD_TOL,
                        same_pick(d["pick"], base["pick"], images, per), d["stop"])
                print("%4d %5d | %-8s %4s %9.4f %8s %5d %6s %8d %9d %5.0f | %s"
                      % (row, 1000 * row + 10 * P + shift, search, "-" if per_iter is None else per_iter,
                         d["floor"], d["r_min"], d["shell"], d["filter"], d["restarts"], d["grads"], d["secs"], note))

if __name__ == "__main__":
    main()