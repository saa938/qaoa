"""
Can placement-blind invariants (standard deviation, elementary symmetric
polynomials) pick out one point of the minimum-radius shell?
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

ROOT = Path(__file__).resolve().parent.parent
SHELLS = ROOT / "results" / "shells"
OUT = ROOT / "results" / "blind_invariants.json"

UNIT = np.pi / 8
# graphs 14 and 19 are omitted: their shells are flagged exact=False in
# weighted_radius.json because the minima do not lie on the pi/8 grid
GRAPHS = [10, 11, 12, 13, 15, 16, 17, 18]
KMAX = 4

# Exact elementary symmetric polynomials e_1..e_kmax of an integer vector
def esp(v, kmax=KMAX):
    v = [int(t) for t in v]
    e = [0] * (len(v) + 1)
    e[0] = 1
    for x in v:
        for k in range(min(len(v), kmax), 0, -1):
            e[k] += e[k - 1] * x
    return e[1:kmax + 1]

# n^2 * variance, as an exact integer
def var_scaled(v):
    v = [int(t) for t in v]
    return len(v) * sum(x * x for x in v) - sum(v) ** 2

# The placement-blind functions, applied to gamma and beta separately 
def functions(kmax=KMAX):
    out = []
    for param in ("gamma", "beta"):
        out.append(("sum_" + param, (param, "e1")))
        out.append(("std_" + param, (param, "var")))
        for k in range(2, kmax + 1):
            out.append(("e%d_%s" % (k, param), (param, "e%d" % k)))
        out.append(("posfrac_" + param, (param, "pos")))
    return out

# Score every point under one function
def score(points, m, spec):
    param, what = spec
    sub = points[:, :m] if param == "gamma" else points[:, m:]
    if what == "var":
        return [var_scaled(r) for r in sub]
    if what == "pos":
        return [int((r > 0).sum()) for r in sub]
    k = int(what[1:])
    return [esp(r, k)[k - 1] for r in sub]

# Apply functions in order, keeping the argmax set at each step.
def filter(shell, m, funcs):
    idx = np.arange(len(shell))
    used = []
    for name, spec in funcs:
        if len(idx) == 1:
            break
        v = score(shell[idx], m, spec)
        best = max(v)
        keep = np.array([j for j, t in enumerate(v) if t == best])
        if len(keep) < len(idx):
            used.append(name)
        idx = idx[keep]
    return idx, used

# Smallest multiset class = the hard floor described in fact 1 above
# "whole" combines the two, "split" keeps gamma and beta separate
def resolution_limit(shell, m):
    whole = Counter(tuple(sorted(r)) for r in shell.tolist())
    split = Counter((tuple(sorted(r[:m])), tuple(sorted(r[m:])))
                    for r in shell.tolist())
    return min(whole.values()), min(split.values())

# Used to verify if surviving points are actually the same solution using symmetries.
def images(G, m, x):
    eo = list(G.edges())
    ei = {frozenset(e): k for k, e in enumerate(eo)}
    n = G.number_of_nodes()
    out = []
    for sig in GraphMatcher(G, G).isomorphisms_iter():
        g = np.empty(m, dtype=int)
        b = np.empty(n, dtype=int)
        for (u, v) in eo:
            g[ei[frozenset((sig[u], sig[v]))]] = x[:m][ei[frozenset((u, v))]]
        for k in range(n):
            b[sig[k]] = x[m:][k]
        y = np.concatenate([g, b])
        out.append(tuple(y.tolist()))
        out.append(tuple((-y).tolist()))
    return out

# Load a cached exact shell and convert it to integer pi/8 units.
def load_shell(graph):
    d = np.load(SHELLS / ("wshell_row%d.npz" % graph))
    S = np.asarray(d["shell"], float)
    U = np.round(S / UNIT).astype(int)
    assert np.abs(S / UNIT - U).max() < 1e-6, "graph %d is not on the pi/8 grid" % graph
    return U, float(d["floor"]), np.asarray(d["edges"])

def main():
    funcs = functions()
    res = []
    print("%4s %7s %7s %7s %8s %9s  %s"
          % ("graph", "shell", "whole", "split", "filter", "1 group", "best"))
    for graph in GRAPHS:
        U, floor, edges = load_shell(graph)
        G = nx.Graph([tuple(e) for e in edges])
        m = G.number_of_edges()

        whole, split = resolution_limit(U, m)
        idx, used = filter(U, m, funcs)
        canon = {min(images(G, m, U[j])) for j in idx}

        print("%4d %7d %7d %7d %8d %9s  %s"
              % (graph, len(U), whole, split, len(idx), len(canon) == 1,
                 ", ".join(used) or "none"))
        res.append({"graph": graph, "shell": int(len(U)), "floor": floor,
                    "limit_whole": int(whole), "limit_split": int(split),
                    "filter": int(len(idx)), "groups": int(len(canon)),
                    "one_group": bool(len(canon) == 1), "best": used})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()