"""
Can a filter of automorphism-invariant functions pick out one point of the
minimum-radius shell?

Reads the shell weighted_radius.py cached in results/shells/, groups it into
symmetry groups, then applies the invariants in order and keeps the argmax set at
each step.  Reports how many points survive and whether they form a single group.
"""

import ast
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import maqaoa_core as M
import weighted_radius as W

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "MaxCutMAQAOAData.csv"
RESULTS_DIR = ROOT / "results"
OUT = RESULTS_DIR / "invariant.json"
SHELLS = RESULTS_DIR / "shells"
WRAD = RESULTS_DIR / "weighted_radius.json"

ROWS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
RESTARTS = 600 # only used when a row has no cached shell
FILTER_TOL = 1e-7
CANON = 5 # digits a canonical representative is rounded to before hashing
BUDGET = 600000 # sign patterns we are willing to enumerate per row

# Odd, automorphism-invariant functions.  Even functions of x cannot separate
# a point from its negation, so every entry here has odd total degree.
def functions(edges, n, m):
    G = nx.Graph(edges)
    G.add_nodes_from(range(n)) # keep isolated vertices in the degree vector
    eo = list(G.edges())
    deg = np.array([G.degree(j) for j in range(n)], dtype=float)
    edeg = np.array([G.degree(u) + G.degree(v) for (u, v) in eo], dtype=float)
    eprod = np.array([G.degree(u) * G.degree(v) for (u, v) in eo], dtype=float)
    tri = np.array([len(list(nx.common_neighbors(G, u, v))) for (u, v) in eo], dtype=float)
    wg = lambda a: np.concatenate([a, np.zeros(n)])
    wb = lambda a: np.concatenate([np.zeros(m), a])
    lin = [("sum_gamma", wg(np.ones(m))), ("sum_beta", wb(np.ones(n))),
           ("deg_gamma", wg(edeg)), ("deg_beta", wb(deg)),
           ("prod_gamma", wg(eprod)), ("tri_gamma", wg(tri))]
    out = []
    for name, v in lin:
        out.append((name, (lambda v: (lambda x: float(v @ x)))(v)))
        out.append((name + "_cube", (lambda v: (lambda x: float(v @ (x ** 3))))(v)))
    return out

# Group points by their canonical image under graph automorphisms and negation.
def groups(shell, images):
    keys = [tuple(np.round(M.canonicalize(x, images), CANON)) for x in shell]
    uniq = {}
    for i, k in enumerate(keys):
        uniq.setdefault(k, []).append(i)
    return list(uniq.values())

# Apply functions in order, keeping the argmax set at each step.
def filter(shell, funcs, tol=FILTER_TOL):
    idx = np.arange(len(shell))
    used = []
    for name, f in funcs:
        if len(idx) == 1:
            break
        v = np.array([f(shell[i]) for i in idx])
        keep = np.where(v >= v.max() - tol * max(1.0, abs(v.max())))[0]
        if len(keep) < len(idx):
            used.append(name)
        idx = idx[keep]
    return idx, used

# The exact flags weighted_radius.py recorded alongside the cached shells.
def load_exact_flags():
    if not WRAD.exists():
        return {}
    return {d["row"]: bool(d["exact"]) for d in json.load(open(WRAD))}

# Reuse the shell weighted_radius.py already computed, if there is one.
def load_shell(row, flags):
    npz = SHELLS / ("wshell_row%d.npz" % row)
    if not npz.exists():
        return None
    d = np.load(npz)
    return np.asarray(d["shell"], float), float(d["floor"]), flags.get(row, False)

# If the shell sits on the pi/8 grid, replace it with the exact enumeration.
def refine_exact(energy, D, shell, floor):
    U = np.round(shell / W.UNIT).astype(int)
    if float(np.abs(shell / W.UNIT - U).max()) >= 1e-3:
        return shell, False
    budget, pool = BUDGET, []
    for mg in sorted({tuple(np.abs(u).tolist()) for u in U}):
        budget -= 2 ** int((np.array(mg) != 0).sum())
        if budget < 0:
            return shell, False
        h = W.enumerate_signs(energy, D, np.array(mg), floor)
        if h is None:
            return shell, False
        pool.extend(list(h))
    keep = []
    for x in pool:
        W.add_distinct(keep, x)
    return np.array(keep), True

def run(df, row, known, flags):
    t0 = time.time()
    edges = [tuple(e) for e in ast.literal_eval(df.loc[row, "Edges"])]
    n = int(df.loc[row, "Number of Nodes"])
    energy, _, grad, D = M.make_energy(n, edges, p=1)
    eo = list(nx.Graph(edges).edges()) # same canonical order make_energy uses
    m = len(eo)

    got, source = load_shell(row, flags), "cached"
    shell, floor, exact = got
    r_min = float(min(W.radius(x) for x in shell))

    autos, images = M.symmetry_group(eo, p=1)
    orb = groups(shell, images)
    sizes = sorted({len(o) for o in orb})
    idx, used = filter(shell, functions(eo, n, m))
    one_group = len({tuple(np.round(M.canonicalize(shell[i], images), CANON))
                     for i in idx}) == 1
    e_sel = float(energy(shell[idx[0]]))

    print("row %2d: shell %3d (%s, %s), |Aut| %2d, groups %3d (sizes %s)"
          % (row, len(shell), "exact" if exact else "lower bound", source,
             len(autos), len(orb), sizes), flush=True)
    print("        invariant filter survivors %d, single group %s, E %.10f, %.0f s"
          % (len(idx), one_group, e_sel, time.time() - t0), flush=True)
    print("        functions that cut: %s" % (", ".join(used) or "none"), flush=True)
    return {"row": row, "n": n, "m": m, "D": D, "floor": floor, "r_min": r_min,
            "shell": len(shell), "exact": exact, "source": source,
            "n_auto": len(autos), "groups": len(orb), "group_sizes": sizes,
            "filter": int(len(idx)), "one_group": bool(one_group),
            "cutters": used, "energy_at_pick": e_sel}

def main():
    rows = [int(a) for a in sys.argv[1:]] or ROWS
    df = pd.read_csv(CSV)
    known = W.load_known()
    flags = load_exact_flags()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res = []
    for row in rows:
        r = run(df, row, known, flags)
        if r is not None:
            res.append(r)
        json.dump(res, open(OUT, "w"), indent=1) # checkpoint after every row

    print("\n%4s %7s %7s %8s %8s %9s" % ("row", "shell", "|Aut|", "groups", "filter", "1 group"))
    for d in res:
        print("%4d %7d %7d %8d %8d %9s"
              % (d["row"], d["shell"], d["n_auto"], d["groups"], d["filter"], d["one_group"]))

if __name__ == "__main__":
    main()
