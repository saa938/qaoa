"""
Can the standard global minimum be pinned down without running the global
optimizer many times on the plain landscape?
"""

import ast
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
import maqaoa_core as M

CSV = "MaxCutMAQAOAData.csv"
OUT = "floor_cheap.json"

# Exact MaxCut by enumerating all 2^n bitstrings.
# This is the simple and obvious but bad way to check.
def brute_maxcut(n, edges):
    best = 0
    for mask in range(1 << n):
        c = sum(1 for (i, j) in edges if ((mask >> i) & 1) != ((mask >> j) & 1))
        best = max(best, c)
    return best

# Per-edge expectations <Z_u Z_v> for a single parameter vector, so the energy can
# be split into its edge contributions.
def make_edge_terms(n, edges, p=1):
    edges = list(nx.Graph(edges).edges())
    m = len(edges)
    dim = 1 << n
    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    spin = 1 - 2 * bits
    zz = np.stack([spin[:, i] * spin[:, j] for (i, j) in edges], 0)
    inv = 1.0 / np.sqrt(dim)
    zero_idx = [np.where(((np.arange(dim) >> j) & 1) == 0)[0] for j in range(n)]

    def terms(x):
        x = np.asarray(x, float)
        gam = x[:m]
        bet = x[m:]
        psi = np.full(dim, inv, dtype=complex)
        psi *= np.exp(1j * (gam[:, None] * zz).sum(0))
        for j in range(n):
            c = np.cos(bet[j])
            s = 1j * np.sin(bet[j])
            i0 = zero_idx[j]
            i1 = i0 ^ (1 << j)
            a0 = psi[i0].copy()
            a1 = psi[i1].copy()
            psi[i0] = c * a0 + s * a1
            psi[i1] = s * a0 + c * a1
        prob = np.abs(psi) ** 2
        return 0.5 * (zz @ prob) - 0.5

    return terms, edges, m

# Coordinates that <Z_u Z_v> is allowed to depend on: the two mixer angles at the
# endpoints and the cost angles on every edge touching either endpoint.
def local_coords(edges, m, n, e_idx):
    u, v = edges[e_idx]
    idx = [i for i, (a, b) in enumerate(edges) if a in (u, v) or b in (u, v)]
    return idx + [m + u, m + v]

# Perturb coordinates outside the local set and confirm the edge term does not move.
def test_locality(terms, edges, m, n, D, trials=6, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for e_idx in range(len(edges)):
        loc = set(local_coords(edges, m, n, e_idx))
        out = [i for i in range(D) if i not in loc]
        if not out:
            continue
        for _ in range(trials):
            x = rng.uniform(0, np.pi, D)
            base = terms(x)[e_idx]
            y = x.copy()
            for i in out:
                y[i] = rng.uniform(0, np.pi)
            worst = max(worst, abs(terms(y)[e_idx] - base))
    return worst

# Minimize one edge term over its local coordinates only.  Summing these gives a
# lower bound on the floor, since relaxing the shared coordinates can only help.
def min_edge_term(terms, edges, m, n, D, e_idx, restarts=12, seed=0):
    loc = local_coords(edges, m, n, e_idx)
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, np.pi, D)

    def f(z):
        x = base.copy()
        x[loc] = z
        return terms(x)[e_idx]

    best = np.inf
    for _ in range(restarts):
        r = minimize(f, rng.uniform(0, np.pi, len(loc)), method="Powell",
                     options={"xtol": 1e-8, "ftol": 1e-10, "maxiter": 4000})
        best = min(best, float(r.fun))
    return best

def main():
    df = pd.read_csv(CSV)
    rows = []
    print("%3s %3s %4s %8s %8s %10s %10s %9s"
          % ("row", "n", "m", "floor", "-maxcut", "certified", "edge_bound", "locality"))
    for row in range(10, 20):
        edges = ast.literal_eval(df.loc[row, "Edges"])
        n = int(df.loc[row, "Number of Nodes"])
        terms, E, m = make_edge_terms(n, edges, p=1)
        D = m + n
        floor = float(np.load("shell_row%d.npz" % row)["floor"])
        mc = brute_maxcut(n, E)
        loc_err = test_locality(terms, E, m, n, D, trials=4, seed=row)
        bound = sum(min_edge_term(terms, E, m, n, D, i, restarts=8, seed=row * 100 + i)
                    for i in range(m))
        cert = abs(floor + mc) < 1e-6
        rows.append({"row": row, "n": n, "m": m, "floor": floor, "neg_maxcut": -mc,
                     "certified_by_maxcut": bool(cert), "edge_bound": round(bound, 4),
                     "locality_err": float(loc_err)})
        print("%3d %3d %4d %8.2f %8d %10s %10.4f %9.1e"
              % (row, n, m, floor, -mc, cert, bound, loc_err))
    json.dump(rows, open(OUT, "w"), indent=1)
    print("\ncertified by the MaxCut bound: %d / %d"
          % (sum(r["certified_by_maxcut"] for r in rows), len(rows)))

if __name__ == "__main__":
    main()