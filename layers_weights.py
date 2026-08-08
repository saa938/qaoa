"""
Does the shell radius grow when layers are added or when the edges get weights?
"""

import ast
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import maqaoa_core as M
import maqaoa_weighted as W
import radius_search as R

CSV = "MaxCutMAQAOAData.csv"
OUT = "layers_weights.json"
ROWS = [10, 13, 17, 19]
N_WEIGHT_TRIALS = 5
W_LO, W_HI = 0.5, 2.0

def radius_of(x):
    v = M.geodesic_vec(np.asarray(x, float), np.zeros(len(x)))
    return float(np.sqrt(v @ v))

def polish(energy, grad, x0):
    return minimize(energy, x0, jac=grad, method="L-BFGS-B",
                    options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})

# Two passes: plain restarts to find the floor, then  hunt inward for the smallest radius on it.
# Note the radius this returns is an upper bound on the shell, not guaranteed to be the lowest shell radius.
def floor_and_shell(energy, grad, D, restarts, seed, a=0.5):
    rng = np.random.default_rng(seed)
    floor = np.inf
    for _ in range(restarts):
        floor = min(floor, float(polish(energy, grad, rng.uniform(0, np.pi, D)).fun))
    floor = round(floor, 6)

    shape, dshape = R.shape_family("exp", a)
    f, fg = R.make_shaped_energy(energy, grad, floor, R.full_radius(D), shape, dshape)
    best = np.inf
    for _ in range(restarts):
        rr = minimize(f, rng.uniform(0, np.pi, D), jac=fg, method="L-BFGS-B",
                      options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})
        rp = polish(energy, grad, rr.x)
        if rp.fun <= floor + 1e-6:
            best = min(best, radius_of(M.wrap_pi(rp.x)))
    return floor, (None if best == np.inf else round(best, 6))

# Setting every weight to 1 must reproduce maqaoa_core exactly
def check_weighted_reduces(df, row=10, n=8):
    edges = [tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])]
    _, eb1, _, D1 = M.make_energy(n, edges, p=1)
    _, eb2, _, _ = W.make_energy_weighted(n, edges, np.ones(len(edges)), p=1)
    X = np.random.default_rng(0).uniform(0, np.pi, (200, D1))
    err = float(np.max(np.abs(eb1(X) - eb2(X))))
    print("check: weighted(w=1) vs core, max abs err = %.3e" % err)
    assert err < 1e-12, "weighted convention does not reduce to the unweighted one"

def main():
    df = pd.read_csv(CSV)
    check_weighted_reduces(df)
    rows = []
    print("%4s %10s %4s %4s %10s %11s %10s %10s"
          % ("row", "kind", "p", "D", "floor", "-maxcut", "radius", "r/sqrtD"))

    for row in ROWS:
        edges = [tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])]
        n = int(df.loc[row, "Number of Nodes"])
        m = len(edges)

        for p in (1, 2):
            energy, energy_batch, grad, D = M.make_energy(n, edges, p=p)
            fl, sh = floor_and_shell(energy, grad, D,
                                     restarts=100 if p == 1 else 80, seed=row * 10 + p)
            r = {"row": row, "m": m, "kind": "layers", "p": p, "D": D,
                 "floor": fl, "shell_radius": sh,
                 "in_quarter_pi_sq": None if sh is None else round((sh / (np.pi / 4)) ** 2, 4),
                 "r_over_sqrt_D": None if sh is None else round(sh / np.sqrt(D), 5)}
            rows.append(r)
            print("%4d %10s %4d %4d %10.4f %11s %10.6f %10.5f"
                  % (row, "layers", p, D, fl, "-", sh, r["r_over_sqrt_D"]))

        for trial in range(3):
            w = np.round(np.random.default_rng(100 * row + trial).uniform(W_LO, W_HI, m), 4)
            energy, energy_batch, grad, D = W.make_energy_weighted(n, edges, w, p=1)
            fl, sh = floor_and_shell(energy, grad, D, restarts=60, seed=row * 10 + trial)
            mc = round(W.brute_weighted_maxcut(n, edges, w), 6)
            r = {"row": row, "m": m, "kind": "weighted", "p": 1, "D": D, "trial": trial,
                 "w_min": float(w.min()), "w_max": float(w.max()),
                 "w_sum": round(float(w.sum()), 4),
                 "floor": fl, "neg_weighted_maxcut": mc, "shell_radius": sh,
                 "in_quarter_pi_sq": None if sh is None else round((sh / (np.pi / 4)) ** 2, 4)}
            rows.append(r)
            r_over_sqrtD = None if sh is None else sh / np.sqrt(D)
            print("%4d %10s %4d %4d %10.4f %11.4f %10s %10s"
                % (row, "weighted", 1, D, fl, mc,
                    "None" if sh is None else "%.6f" % sh,
                    "None" if r_over_sqrtD is None else "%.5f" % r_over_sqrtD))

    json.dump(rows, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()
