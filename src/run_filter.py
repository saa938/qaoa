"""
Does the filter of graph-invariant functionals still pick out one point of the
minimum-radius shell once the edges are canonically ordered, once they carry
random weights, and once the circuit has more than one layer?

Self-contained: does not import maqaoa_core or weighted_radius.  The energy is the
same statevector construction as maqaoa_core.make_energy, extended with per-edge
weights and p layers.

Edge order is fixed by canonical_edges: lowest node index first inside each edge,
then the list sorted by first index ascending and by second index ascending within
it.  Everything downstream (energy, functionals, symmetry) is built on that order.

Usage:  python src/run_filter.py <p> <weighted 0|1> <row> [row ...]
Example: python src/run_filter.py 1 0 10 11 12
Results are appended to results/ordered_filter.json after every case, so re-running
the same command skips whatever already finished.
"""

import ast
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "MaxCutMAQAOAData.csv"
RESULTS_DIR = ROOT / "results"
OUT = RESULTS_DIR / "ordered_filter.json"

ROWS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
W_LO, W_HI = 0.5, 2.0
LAMS = [0.0, 0.1, 0.2, 0.35]
RESTARTS = {1: 100, 2: 25}
N_TRIALS = 2
TOL_E = 1e-6 # energy tolerance for sitting on the floor
TOL_D = 1e-3 # period-aware distance below which two minima are the same point
RAD_TOL = 1e-4 # radius window that defines the inner shell
FILTER_TOL = 1e-7
CANON = 5 # digits a canonical representative is rounded to before hashing
BUDGET = 400000 # sign patterns we are willing to enumerate per magnitude pattern

# Canonical edge order: lowest node index first inside each edge, then sorted by
# first index and by second index within it.  Self loops and duplicates dropped.
def canonical_edges(edges):
    seen = set()
    for (u, v) in edges:
        u, v = int(u), int(v)
        if u == v:
            continue
        seen.add((min(u, v), max(u, v)))
    return sorted(seen)

# Per-coordinate parameter-shift step: pi/(4 w_e) on gammas, pi/4 on betas.
def shift_vector(m, n, p, w):
    g = np.tile(np.pi / (4.0 * np.asarray(w, float)), p)
    b = np.full(p * n, np.pi / 4.0)
    return np.concatenate([g, b])

# Chain-rule factor that goes with that shift: w_e on gammas, 1 on betas.
def shift_scale(m, n, p, w):
    return np.concatenate([np.tile(np.asarray(w, float), p), np.ones(p * n)])

# Per-coordinate period of the energy: pi/w_e on gammas, pi on betas.  Shifting
# gamma_e by pi/w_e multiplies every amplitude by -1, which is a global phase.
# With w = 1 this is the plain pi periodicity maqaoa_core assumes.
def periods(m, n, p, w):
    g = np.tile(np.pi / np.asarray(w, float), p)
    b = np.full(p * n, np.pi)
    return np.concatenate([g, b])

# Statevector MA-QAOA energy for p layers on a weighted graph.  Cost unitary is
# exp(i gamma_e w_e Z_u Z_v) and the objective is
# 0.5 * sum_e w_e <Z_u Z_v> - 0.5 * sum_e w_e.  w = 1 reduces to maqaoa_core exactly.
def make_energy(n, edges, w=None, p=1):
    edges = canonical_edges(edges)
    m = len(edges)
    w = np.ones(m) if w is None else np.asarray(w, float)
    if len(w) != m:
        raise ValueError("need one weight per edge, got %d for %d edges" % (len(w), m))
    D = p * (m + n)
    dim = 1 << n

    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    spin = 1 - 2 * bits
    zz = np.stack([spin[:, i] * spin[:, j] for (i, j) in edges], 0).astype(float)
    inv = 1.0 / np.sqrt(dim)
    zero_idx = [np.where(((np.arange(dim) >> j) & 1) == 0)[0] for j in range(n)]
    wzz = zz * w[:, None]

    def _evolve(X):
        B = X.shape[0]
        gammas = X[:, :p * m].reshape(B, p, m)
        betas = X[:, p * m:].reshape(B, p, n)
        psi = np.full((B, dim), inv, dtype=complex)
        for layer in range(p):
            phase = np.einsum('be,ek->bk', gammas[:, layer, :] * w, zz)
            psi *= np.exp(1j * phase)
            for j in range(n):
                b = betas[:, layer, j][:, None]
                c = np.cos(b)
                s = 1j * np.sin(b)
                i0 = zero_idx[j]
                i1 = i0 ^ (1 << j)
                a0 = psi[:, i0]
                a1 = psi[:, i1]
                psi[:, i0] = c * a0 + s * a1
                psi[:, i1] = s * a0 + c * a1
        return psi

    def energy_batch(X):
        X = np.atleast_2d(X)
        prob = np.abs(_evolve(X)) ** 2
        return 0.5 * np.einsum('bk,ek->b', prob, wzz) - 0.5 * w.sum()

    def energy(x):
        return float(energy_batch(np.asarray(x, float)[None, :])[0])

    S = np.diag(shift_vector(m, n, p, w))
    mu = shift_scale(m, n, p, w)

    def grad(x):
        x = np.asarray(x, float)
        vals = energy_batch(np.concatenate([x[None, :] + S, x[None, :] - S], 0))
        return mu * (vals[:D] - vals[D:])

    return energy, energy_batch, grad, D

# Fold into the centred fundamental domain [-per/2, per/2).
def fold(x, per):
    return np.mod(np.asarray(x, float) + per / 2.0, per) - per / 2.0

# Distance from a point to the origin on the torus.
def norm_origin(x, per):
    v = fold(x, per)
    return float(np.sqrt(v @ v))

# Geodesic distance between two points on the torus.
def geodesic_dist(a, b, per):
    d = fold(np.asarray(a, float) - np.asarray(b, float), per)
    return float(np.sqrt(d @ d))

# Optimize a point using L-BFGS-B.
def polish(energy, grad, x0):
    return minimize(energy, x0, jac=grad, method="L-BFGS-B",
                    options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})

# Symmetry group from weight-preserving automorphisms plus the global sign flip.
# Takes n so isolated vertices survive, and w so that with random weights the
# automorphisms collapse to the identity as they should.
def symmetry_group(edges, n, w, p=1):
    edges = canonical_edges(edges)
    m = len(edges)
    per = periods(m, n, p, w)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for k, (u, v) in enumerate(edges):
        G.add_edge(u, v, weight=float(w[k]))
    em = lambda a, b: abs(a["weight"] - b["weight"]) < 1e-12
    autos = list(GraphMatcher(G, G, edge_match=em).isomorphisms_iter())
    eidx = {frozenset(e): i for i, e in enumerate(edges)}

    def induced(sig, x):
        gam = x[:p * m].reshape(p, m)
        bet = x[p * m:].reshape(p, n)
        gb = np.empty((p, m))
        bb = np.empty((p, n))
        for k in range(n):
            bb[:, sig[k]] = bet[:, k]
        for (u, v) in edges:
            gb[:, eidx[frozenset((sig[u], sig[v]))]] = gam[:, eidx[frozenset((u, v))]]
        return np.concatenate([gb.ravel(), bb.ravel()])

    def images(x):
        out = []
        for sig in autos:
            y = induced(sig, x)
            out.append(fold(y, per))
            out.append(fold(-y, per))
        return out

    return autos, images

# Canonicalize a point by returning the lexicographically smallest of its images.
def canonicalize(x, images):
    return sorted(images(x), key=lambda y: tuple(np.round(y, CANON)))[0]

# Partition a set of points into symmetry orbits.
def groups(shell, images):
    uniq = {}
    for i, x in enumerate(shell):
        k = tuple(np.round(canonicalize(x, images), CANON))
        uniq.setdefault(k, []).append(i)
    return list(uniq.values())

# Odd, graph-invariant functionals.  Even functions of x cannot separate a point
# from its negation, so every entry has odd total degree.  Each base vector is
# applied to one layer at a time, since layers are not interchangeable.  The
# weight-based entries only appear when the graph is actually weighted.
def functionals(edges, n, w, p=1):
    edges = canonical_edges(edges)
    m = len(edges)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for k, (u, v) in enumerate(edges):
        G.add_edge(u, v, weight=float(w[k]))
    deg = np.array([G.degree(j) for j in range(n)], dtype=float)
    wdeg = np.array([G.degree(j, weight="weight") for j in range(n)], dtype=float)
    edeg = np.array([deg[u] + deg[v] for (u, v) in edges])
    eprod = np.array([deg[u] * deg[v] for (u, v) in edges])
    ewdeg = np.array([wdeg[u] + wdeg[v] for (u, v) in edges])
    tri = np.array([len(list(nx.common_neighbors(G, u, v))) for (u, v) in edges],
                   dtype=float)
    wv = np.asarray(w, float)

    base = [("sum_gamma", "g", np.ones(m)), ("sum_beta", "b", np.ones(n)),
            ("deg_gamma", "g", edeg), ("deg_beta", "b", deg),
            ("prod_gamma", "g", eprod), ("tri_gamma", "g", tri)]
    if not np.allclose(wv, 1.0):
        base += [("w_gamma", "g", wv), ("wdeg_gamma", "g", ewdeg),
                 ("wdeg_beta", "b", wdeg)]

    D = p * (m + n)
    out = []
    for name, side, vec in base:
        for layer in range(p):
            v = np.zeros(D)
            if side == "g":
                v[layer * m:(layer + 1) * m] = vec
            else:
                v[p * m + layer * n:p * m + (layer + 1) * n] = vec
            tag = name if p == 1 else "%s_L%d" % (name, layer)
            out.append((tag, (lambda v: (lambda x: float(v @ x)))(v)))
            out.append((tag + "_cube", (lambda v: (lambda x: float(v @ (x ** 3))))(v)))
    return out

# Apply the functionals in a fixed order, keeping the argmax set at each step.
def apply_filter(shell, funcs, tol=FILTER_TOL):
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

def add_distinct(pool, x, per):
    if all(geodesic_dist(x, q, per) > TOL_D for q in pool):
        pool.append(x)
        return True
    return False

# Plain restarts, then restarts on E + lam * r^2 to pull inward.  Every candidate
# is re-minimised on the untouched energy, so the penalty only acts as a sampler.
# Points from both passes are kept: dropping the first pass would throw away the
# points that reached the floor whenever the penalised pass never gets back down.
def harvest(energy, grad, D, per, restarts, seed):
    rng = np.random.default_rng(seed)

    def r2(x):
        v = fold(x, per)
        return float(v @ v)

    def r2_jac(x):
        return 2.0 * fold(x, per)

    cand = []
    for _ in range(restarts):
        rp = polish(energy, grad, rng.uniform(0, np.pi, D))
        cand.append((float(rp.fun), fold(rp.x, per)))

    for lam in LAMS:
        obj = lambda x, l=lam: energy(x) + l * r2(x)
        jac = lambda x, l=lam: grad(x) + l * r2_jac(x)
        for _ in range(restarts):
            r = minimize(obj, rng.uniform(0, np.pi, D), jac=jac, method="L-BFGS-B",
                         options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 3000})
            rp = polish(energy, grad, r.x)
            cand.append((float(rp.fun), fold(rp.x, per)))

    # The floor is decided once every candidate is in, otherwise a late improvement
    # would discard everything collected against the older floor.
    floor = min(f for f, _ in cand)
    pool = []
    for f, x in sorted(cand, key=lambda t: t[0]):
        if f <= floor + TOL_E:
            add_distinct(pool, x, per)
    return floor, pool

# Keep the points at the smallest radius, then close that set under the symmetry
# group and under negation.
def inner_shell(pool, energy, images, per, floor):
    if not pool:
        return [], None
    r_min = min(norm_origin(x, per) for x in pool)
    shell = [x for x in pool if norm_origin(x, per) <= r_min + RAD_TOL]
    closed = []
    for x in shell:
        for y in images(x):
            if energy(y) <= floor + TOL_E and abs(norm_origin(y, per) - r_min) <= RAD_TOL:
                add_distinct(closed, y, per)
    return closed, r_min

# Where the shell sits on the pi/8 grid it can be made exact rather than
# restart-limited: hold each magnitude pattern fixed and try every sign pattern.
# Only applies when the period is pi on every coordinate, i.e. the unweighted case.
# A pattern that was not really on the grid fails the floor test and contributes
# nothing, so snapping is safe to attempt even for a point a little way off.
def refine_on_grid(energy_batch, shell, floor, per, budget=BUDGET):
    if not shell or not np.allclose(per, np.pi):
        return None
    unit = np.pi / 8.0
    S = np.array(shell)
    U = np.round(S / unit)
    on = np.abs(S / unit - U).max(axis=1) < 1e-3
    pool = []
    for mg in sorted({tuple(np.abs(u).tolist()) for u in U}):
        mag = np.array(mg) * unit
        nz = np.where(mag > 0)[0]
        if len(nz) > 20 or 2 ** len(nz) > budget:
            continue
        k = len(nz)
        signs = 1 - 2 * ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1)
        X = np.tile(mag, (1 << k, 1))
        X[:, nz] *= signs
        for lo in range(0, len(X), 4096):
            chunk = X[lo:lo + 4096]
            vals = energy_batch(chunk)
            for x in chunk[vals <= floor + TOL_E]:
                add_distinct(pool, fold(x, per), per)
    # Merge the restart points back with a looser tolerance, so a restart point
    # that is only an unsnapped copy of an enumerated one does not inflate the count.
    for x in S:
        y = fold(x, per)
        if all(geodesic_dist(y, q, per) > 1e-2 for q in pool):
            pool.append(y)
    return pool, bool(on.all())

def draw_weights(m, seed):
    return np.round(np.random.default_rng(seed).uniform(W_LO, W_HI, m), 4)

def load_results():
    if not OUT.exists():
        return []
    try:
        return json.load(open(OUT))
    except (ValueError, OSError):
        return []

def already_done(res, row, p, weighted, trial):
    return any(d["row"] == row and d["p"] == p and d["weighted"] == weighted
               and d.get("trial") == trial for d in res)

def run(row, edges, n, w, p, seed, weighted, trial):
    t0 = time.time()
    m = len(canonical_edges(edges))
    energy, energy_batch, grad, D = make_energy(n, edges, w=w, p=p)
    per = periods(m, n, p, w)
    autos, images = symmetry_group(edges, n, w, p)

    floor, pool = harvest(energy, grad, D, per, RESTARTS[p], seed)
    shell, r_min = inner_shell(pool, energy, images, per, floor)

    exact = False
    got = refine_on_grid(energy_batch, shell, floor, per)
    if got is not None and got[0]:
        ref, all_on = got
        r_ref = min(norm_origin(x, per) for x in ref)
        ref = [x for x in ref if norm_origin(x, per) <= r_ref + RAD_TOL]
        if len(ref) >= len(shell):
            shell, r_min, exact = ref, r_ref, all_on

    rec = {"row": row, "n": n, "m": m, "p": p, "D": D, "weighted": weighted,
           "trial": trial, "seed": seed, "floor": round(floor, 6),
           "r_min": None if r_min is None else round(r_min, 6),
           "pool": len(pool), "shell": len(shell), "exact": exact,
           "n_auto": len(autos), "restarts": RESTARTS[p],
           "w": None if not weighted else [float(a) for a in w]}

    if not shell:
        rec.update({"groups": None, "filter": None, "one_group": None,
                    "used": [], "energy_at_pick": None,
                    "secs": round(time.time() - t0, 1)})
        print("row %2d  p %d  no floor points found" % (row, p), flush=True)
        return rec

    idx, used = apply_filter(shell, functionals(edges, n, w, p))
    orb = groups([shell[i] for i in idx], images)
    rec.update({"groups": len(orb), "filter": int(len(idx)),
                "one_group": len(orb) == 1, "used": used,
                "energy_at_pick": round(float(energy(shell[idx[0]])), 8),
                "secs": round(time.time() - t0, 1)})

    print("row %2d  p %d  %-10s t%-2s floor %9.4f  r %8s  pool %4d  shell %4d %-11s "
          "|Aut| %2d  filter %3d  groups %2d  %5.0f s"
          % (row, p, "weighted" if weighted else "unweighted",
             "-" if trial is None else trial, floor,
             "-" if r_min is None else "%.5f" % r_min, len(pool), len(shell),
             "exact" if exact else "lower bound", len(autos), len(idx), len(orb),
             rec["secs"]), flush=True)
    print("        functionals used: %s" % (", ".join(used) or "none"), flush=True)
    return rec

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    p = int(sys.argv[1])
    if p not in RESTARTS:
        raise SystemExit("p must be one of %s" % sorted(RESTARTS))
    weighted = bool(int(sys.argv[2]))
    rows = [int(a) for a in sys.argv[3:]] or ROWS
    if not CSV.exists():
        raise SystemExit("cannot find %s" % CSV)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    res = load_results()
    for row in rows:
        if row not in df.index:
            print("row %d is not in the csv, skipping" % row, flush=True)
            continue
        edges = [tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])]
        n = int(df.loc[row, "Number of Nodes"])
        m = len(canonical_edges(edges))
        trials = list(range(N_TRIALS if p == 1 else 1)) if weighted else [None]
        for t in trials:
            if already_done(res, row, p, weighted, t):
                print("row %2d p %d weighted %s trial %s already done"
                      % (row, p, weighted, t), flush=True)
                continue
            w = np.ones(m) if not weighted else draw_weights(m, 100 * row + t)
            seed = 1000 * row + 10 * p + (0 if t is None else t + 1)
            res.append(run(row, edges, n, w, p, seed, weighted, t))
            json.dump(res, open(OUT, "w"), indent=1) # checkpoint after every case

    shown = [d for d in res if d["p"] == p and d["weighted"] == weighted
             and d["row"] in rows]
    if shown:
        print("\n%4s %2s %7s %7s %7s %8s %8s %9s"
              % ("row", "p", "trial", "shell", "|Aut|", "groups", "filter", "1 group"))
        for d in shown:
            print("%4d %2d %7s %7d %7d %8s %8s %9s"
                  % (d["row"], d["p"], "-" if d["trial"] is None else d["trial"],
                     d["shell"], d["n_auto"], d["groups"], d["filter"],
                     d["one_group"]))

if __name__ == "__main__":
    main()