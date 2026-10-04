"""
Usage:  python src/run_filter.py <p> <weighted 0|1> <row> [row ...]
Example: python src/run_filter.py 1 0 10 11 12
With no arguments it runs DEFAULT_P, DEFAULT_WEIGHTED on every row in ROWS.
p = 1 picks are reproducible across seeds (row 14 too, since settle is followed by slide).
Row 11 weight draw 0 is the slow case (~1,800 restarts).
p >= 2: several different minima can share the smallest radius, and the pick repeats once a seed has
found the one the filter picks (row 16 weight draw 0: about 1 walk in 7, two seeds agree to 5e-5).
Run two seeds and compare the saved picks.
Results are appended to results/ordered_filter_v2.json after every case
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
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "MaxCutMAQAOAData.csv"
RESULTS_DIR = ROOT / "results"
SEARCH = "restarts" # "restarts" = random restarts over the whole space, "radius" = restarts inside a shrinking maximum radius
OUT = RESULTS_DIR / ("ordered_filter_v2.json" if SEARCH == "restarts" else "ordered_filter_v2_radius.json")

ROWS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
DEFAULT_P = 2
DEFAULT_WEIGHTED = 1
W_LO, W_HI = 0.5, 2.0
LAMS = [0.0, 0.1, 0.35] # restart i uses LAMS[i % len(LAMS)]; 0 is a plain restart
RESTARTS = {1: 100, 2: 25} # restarts per batch; the pick is re-checked after every batch
STALL = {1: 400, 2: 200} # stop once the pick has not changed for this many restarts
MIN_HITS = 3 # and at least this many restarts have landed on the pick
MAX_RESTARTS = {1: 3000, 2: 1000}
LOOP_RESTARTS = {1: 100} # radius search: restarts per iteration
LOOP_ITERS = 12 # radius search: most iterations
PATIENCE = 2 # radius search: stop once more than this many iterations in a row find nothing closer
BIG = 1e6 # radius search: energy returned outside the maximum radius
CAP_EPS = 1e-3 # radius search: the smallish factor added to the maximum radius
N_TRIALS = 2
TOL_E = 1e-6 # energy tolerance for sitting on the floor
TOL_D = 1e-3 # period-aware distance below which two minima are the same point
RAD_TOL = 1e-4 # radius window that defines the inner shell
FILTER_TOL = 1e-7
CANON = 5 # digits a canonical representative is rounded to before hashing
BUDGET = 400000 # sign patterns we are willing to enumerate per magnitude pattern
EV_ZERO = 1e-6 # Hessian eigenvalues below this (relative to the largest) are flat directions
TIE_TOL = 1e-6 # a reduced gamma this close to -h/2 is moved to +h/2; a beta this close to +-pi/4 sits on the edge
SPELL_CAP = 4096 # most same-radius versions of one point we will list
H_STEP = 1e-5 # finite-difference step for the Hessian
SLIDE_P = 1 # from this p on, settle is followed by slide and one more settle (settle alone stops short in curved valleys)
SLIDE_MUS = [1e1, 1e3, 1e5, 1e7] # energy penalty weights for one slide round, loose to tight
SLIDE_ROUNDS = 5 # most slide rounds; stops early once the radius stops dropping

# Edge ordering
def canonical_edges(edges):
    seen = set()
    for (u, v) in edges:
        u, v = int(u), int(v)
        if u == v:
            continue
        seen.add((min(u, v), max(u, v)))
    return sorted(seen)

# Per-coordinate period of the energy: pi/w_e on gammas, pi on betas.
def periods(m, n, p, w):
    g = np.tile(np.pi / np.asarray(w, float), p)
    b = np.full(p * n, np.pi)
    return np.concatenate([g, b])

# Make the energy function, its batch version, and the adjoint gradient.
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
    half_c = 0.5 * wzz.sum(0)
    const = 0.5 * w.sum()
    shapes = [(dim >> (j + 1), 2, 1 << j) for j in range(n)]

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

    # Rotate qubit j in place: (a0, a1) -> (c a0 + s a1, s a0 + c a1).
    def _rot(psi, j, c, s):
        v = psi.reshape(shapes[j])
        a0 = v[:, 0, :].copy()
        a1 = v[:, 1, :]
        v[:, 0, :] = c * a0 + s * a1
        v[:, 1, :] = s * a0 + c * a1

    # Run the circuit on one point, keeping the phase of every cost layer for the backward pass.
    def _forward(x):
        gam = x[:p * m].reshape(p, m)
        bet = x[p * m:].reshape(p, n)
        psi = np.full(dim, inv, dtype=complex)
        phases = []
        for layer in range(p):
            ph = np.exp(1j * (gam[layer] @ wzz))
            phases.append(ph)
            psi *= ph
            for j in range(n):
                _rot(psi, j, np.cos(bet[layer, j]), 1j * np.sin(bet[layer, j]))
        return psi, phases, bet

    def energy(x):
        psi = _forward(np.asarray(x, float))[0]
        return float(half_c @ (psi.real ** 2 + psi.imag ** 2) - const)

    # Compute the adjoint gradient of the energy at x.  If with_energy is True, return (energy, gradient).
    def grad(x, with_energy=False):
        x = np.asarray(x, float)
        psi, phases, bet = _forward(x)
        e = float(half_c @ (psi.real ** 2 + psi.imag ** 2) - const)
        lam = half_c * psi
        g_gam = np.empty((p, m))
        g_bet = np.empty((p, n))
        for layer in range(p - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                vl = lam.reshape(shapes[j])
                vp = psi.reshape(shapes[j])
                g_bet[layer, j] = -2.0 * (np.vdot(vl[:, 0, :], vp[:, 1, :])
                                          + np.vdot(vl[:, 1, :], vp[:, 0, :])).imag
                c, s = np.cos(bet[layer, j]), -1j * np.sin(bet[layer, j])
                _rot(psi, j, c, s)
                _rot(lam, j, c, s)
            g_gam[layer] = -2.0 * w * (zz @ (lam.conj() * psi).imag)
            back = phases[layer].conj()
            psi *= back
            lam *= back
        g = np.concatenate([g_gam.ravel(), g_bet.ravel()])
        return (e, g) if with_energy else g

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

# Optimize a point using L-BFGS-B, taking the energy and gradient from one call.
def polish(energy, grad, x0):
    return minimize(lambda x: grad(x, True), x0, jac=True, method="L-BFGS-B",
                    options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})

# Reduce a point to the fundamental domain, flipping gammas and betas as needed to keep the point in the same radius shell.
def reduce_point(x, edges, n, w, p=1):
    edges = canonical_edges(edges)
    m = len(edges)
    h = np.pi / (2.0 * np.asarray(w, float))
    gam = np.asarray(x, float)[:p * m].reshape(p, m).copy()
    bet = np.asarray(x, float)[p * m:].reshape(p, n).copy()
    for layer in range(p):
        k = np.round(bet[layer] / (np.pi / 2)).astype(int)
        bet[layer] -= k * np.pi / 2
        for j in np.where(k % 2 != 0)[0]:
            for e, (u, v) in enumerate(edges):
                if j in (u, v):
                    gam[:layer + 1, e] *= -1
    flip = np.zeros(n, dtype=int)
    for layer in range(p):
        k = np.round(gam[layer] / h).astype(int)
        gam[layer] -= k * h
        tie = gam[layer] < -h / 2 + TIE_TOL
        gam[layer][tie] += h[tie]
        k[tie] -= 1
        for e, (u, v) in enumerate(edges):
            if k[e] % 2:
                flip[u] ^= 1
                flip[v] ^= 1
        bet[layer] = fold(np.where(flip == 1, -bet[layer], bet[layer]), np.pi)
    return np.concatenate([gam.ravel(), bet.ravel()])

# Enumerate all points that are the same radius as x, by flipping betas on the edge of the fundamental domain and flipping 
# the gammas on their edges in that layer and every earlier layer.     
def spellings(x, edges, n, w, p=1, cap=SPELL_CAP):
    edges = canonical_edges(edges)
    m = len(edges)
    q = np.pi / 4
    x = reduce_point(x, edges, n, w, p)
    b = x[p * m:]
    on = np.abs(np.abs(b) - q) < TIE_TOL
    b[on] = np.sign(b[on]) * q
    key = lambda y: tuple(np.round(y, CANON))
    found = {key(x): x}
    todo = [x]
    while todo and len(found) < cap:
        z = todo.pop()
        for k in np.where(np.abs(np.abs(z[p * m:]) - q) < TIE_TOL)[0]:
            layer, j = divmod(int(k), n)
            y = z.copy()
            y[p * m + k] = -y[p * m + k]
            for e, (u, v) in enumerate(edges):
                if j in (u, v):
                    for l in range(layer + 1):
                        y[l * m + e] = -y[l * m + e]
            y = reduce_point(y, edges, n, w, p)
            if key(y) not in found:
                found[key(y)] = y
                todo.append(y)
    return list(found.values())

# Hessian by central differences of the adjoint gradient.
def hessian(grad, x, h=H_STEP):
    D = len(x)
    H = np.empty((D, D))
    for k in range(D):
        e = np.zeros(D)
        e[k] = h
        H[k] = (grad(x + e) - grad(x - e)) / (2 * h)
    return 0.5 * (H + H.T)

# Iteratively move along the flat directions of the Hessian until the point settles into a local minimum.
def settle(energy, grad, x, reduce, max_it=30):
    x = reduce(x)
    e0 = energy(x)
    for _ in range(max_it):
        try:
            ev, V = eigh(hessian(grad, x))
        except np.linalg.LinAlgError:
            break
        N = V[:, ev < EV_ZERO * max(1.0, ev[-1])]
        t = N @ (N.T @ x)
        if N.shape[1] == 0 or np.linalg.norm(t) < 1e-10:
            break
        moved = False
        for step in (1.0, 0.5, 0.25, 0.125):
            y = reduce(polish(energy, grad, x - step * t).x)
            if y @ y < x @ x - 1e-12 and energy(y) <= e0 + TOL_E:
                x, moved = y, True
                break
        if not moved:
            break
    return x

# Iteratively slide along the gradient of the energy, with a penalty for moving away from the current radius.
def slide(energy, grad, x, reduce, rounds=SLIDE_ROUNDS):
    x = reduce(x)
    e0 = energy(x)
    for _ in range(rounds):
        y = x.copy()
        for mu in SLIDE_MUS:
            def obj(z, mu=mu):
                e, g = grad(z, True)
                return 0.5 * (z @ z) + mu * (e - e0), z + mu * g
            y = minimize(obj, y, jac=True, method="L-BFGS-B",
                         options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000}).x
        y = reduce(polish(energy, grad, y).x)
        if energy(y) > e0 + TOL_E or y @ y >= x @ x - 1e-10:
            break
        x = y
    return x

# One restart: plain L-BFGS-B when lam is 0, otherwise minimize energy + lam * r^2 first
# (pulls towards small radius) and then polish on the true energy.
def descend(energy, grad, per, x0, lam):
    if lam == 0:
        rp = polish(energy, grad, x0)
        return float(rp.fun), rp.x, int(rp.nfev)

    def obj(x):
        e, g = grad(x, True)
        v = fold(x, per)
        return e + lam * (v @ v), g + 2 * lam * v

    r = minimize(obj, x0, jac=True, method="L-BFGS-B",
                 options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 3000})
    rp = polish(energy, grad, r.x)
    return float(rp.fun), rp.x, int(r.nfev + rp.nfev)

# Radius of the closest copy of x to the origin (each coordinate folded onto its half period).
def reduced_radius(x, hper):
    v = x - hper * np.round(x / hper)
    return float(np.sqrt(v @ v))

# Uniform random point with radius below R inside the reduced box.
def ball_start(rng, D, R, hper):
    X = (rng.random((20000, D)) - 0.5) * hper
    ok = np.where((X * X).sum(1) <= R * R)[0]
    if len(ok):
        return X[ok[0]]
    while True:
        U = rng.normal(size=(20000, D))
        U /= np.linalg.norm(U, axis=1)[:, None]
        X = R * rng.random(20000)[:, None] ** (1.0 / D) * U
        ok = np.where((np.abs(X) <= hper / 2).all(1))[0]
        if len(ok):
            return X[ok[0]]

# One restart of the radius search: L-BFGS-B on the energy with BIG and zero gradient outside
# radius R, then a polish on the true energy.
def descend_capped(energy, grad, hper, x0, R):
    lim = R * (1 + CAP_EPS)

    def obj(x):
        if reduced_radius(x, hper) > lim:
            return BIG, np.zeros(len(x))
        return grad(x, True)

    r = minimize(obj, x0, jac=True, method="L-BFGS-B",
                 options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})
    rp = polish(energy, grad, r.x)
    return float(rp.fun), rp.x, int(r.nfev + rp.nfev)

# Compute the symmetry group of the graph and a function that returns all images of a point under the group.
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
            out += spellings(y, edges, n, w, p)
            out += spellings(-y, edges, n, w, p)
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

# Odd, graph-invariant functions.
def functions(edges, n, w, p=1):
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

# Apply the functions in a fixed order, keeping the argmax set at each step.
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

# Run a batch of restarts, keeping track of the best energy and the pick.  The pick is the point
def harvest(energy, grad, D, per, p, seed, settle_fn, pick_fn, images):
    rng = np.random.default_rng(seed)
    batch = RESTARTS[p]
    cap = MAX_RESTARTS.get(p, 30 * batch)
    stall = STALL.get(p, 4 * batch)
    floor = np.inf
    pool, counts, seen = [], [], []
    pick, since, total, nfev, hits = None, 0, 0, 0, 0
    while total < cap:
        old_floor = floor
        for _ in range(batch):
            lam = LAMS[total % len(LAMS)]
            f, x, nf = descend(energy, grad, per, (rng.random(D) - 0.5) * per, lam)
            total += 1
            nfev += nf
            if f < floor - TOL_E:
                pool, counts, seen = [], [], []
            floor = min(floor, f)
            if f > floor + TOL_E:
                continue
            xf = fold(x, per)
            j = next((k for k, (q, _) in enumerate(seen) if geodesic_dist(xf, q, per) <= TOL_D), None)
            if j is None:
                y = settle_fn(xf)
                if energy(y) > floor + TOL_E:
                    continue
                k = next((i for i, q in enumerate(pool) if geodesic_dist(y, q, per) <= TOL_D), None)
                if k is None:
                    pool.append(y)
                    counts.append(0)
                    k = len(pool) - 1
                seen.append((xf, k))
                j = len(seen) - 1
            counts[seen[j][1]] += 1

        new = pick_fn(floor, pool)
        same = (pick is not None and new is not None and abs(floor - old_floor) <= TOL_E
                and min(geodesic_dist(y, new, per) for y in images(pick)) <= TOL_D)
        since = since + batch if same else 0
        pick = new
        if pick is not None:
            imgs = images(pick)
            hits = sum(c for q, c in zip(pool, counts)
                       if min(geodesic_dist(y, q, per) for y in imgs) <= TOL_D)
        if since >= stall and hits >= MIN_HITS:
            break
    return floor, pool, {"hits": int(hits), "restarts": int(total), "nfev": int(nfev),
                         "stalled": int(since), "converged": bool(since >= stall and hits >= MIN_HITS)}

# Run a batch of restarts, keeping track of the best energy and the pick.  The pick is the point
# with the smallest radius, and the radius is shrunk whenever a smaller one is found.
def harvest_radius(energy, grad, D, per, p, seed, settle_fn, pick_fn, images):
    rng = np.random.default_rng(seed)
    batch = LOOP_RESTARTS[p]
    hper = per / 2
    full = float(np.sqrt(((hper / 2) ** 2).sum()))
    R, best, floor = full, np.inf, np.inf
    pool, counts, seen = [], [], []
    total, nfev, stalls, stop = 0, 0, 0, "hit LOOP_ITERS"
    for _ in range(LOOP_ITERS):
        radii = []
        for _ in range(batch):
            f, x, nf = descend_capped(energy, grad, hper, ball_start(rng, D, R, hper), R)
            total += 1
            nfev += nf
            if f < floor - TOL_E:
                pool, counts, seen, radii = [], [], [], []
                R, best, stalls = full, np.inf, 0
            floor = min(floor, f)
            if f > floor + TOL_E:
                continue
            xf = fold(x, per)
            j = next((k for k, (q, _) in enumerate(seen) if geodesic_dist(xf, q, per) <= TOL_D), None)
            if j is None:
                y = settle_fn(xf)
                if energy(y) > floor + TOL_E:
                    continue
                k = next((i for i, q in enumerate(pool) if geodesic_dist(y, q, per) <= TOL_D), None)
                if k is None:
                    pool.append(y)
                    counts.append(0)
                    k = len(pool) - 1
                seen.append((xf, k))
                j = len(seen) - 1
            ry = norm_origin(pool[seen[j][1]], per)
            if ry > R + RAD_TOL:
                continue
            radii.append(ry)
            counts[seen[j][1]] += 1
        if not radii:
            stop = "no hits"
            break
        best = min(best, min(radii))
        if min(radii) >= R - RAD_TOL:
            stalls += 1
            if stalls > PATIENCE:
                stop = "converged"
                break
        else:
            stalls = 0
        R = min(R, best)

    pick, hits = pick_fn(floor, pool), 0
    if pick is not None:
        imgs = images(pick)
        hits = sum(c for q, c in zip(pool, counts)
                   if min(geodesic_dist(y, q, per) for y in imgs) <= TOL_D)
    return floor, pool, {"hits": int(hits), "restarts": int(total), "nfev": int(nfev),
                         "stalled": int(stalls * batch), "converged": stop == "converged", "stop": stop}

# Keep the points at the smallest radius and remove ones using symmetry
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

# Do the points lie on the pi/8 grid?  If so, we can enumerate the exact shell.
def refine_on_grid(energy_batch, shell, floor, per, reduce, budget=BUDGET):
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
                add_distinct(pool, reduce(x), per)
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
    funcs = functions(edges, n, w, p)
    reduce = lambda x: reduce_point(x, edges, n, w, p)
    if p >= SLIDE_P:
        settle_fn = lambda x: settle(energy, grad, slide(energy, grad, settle(energy, grad, x, reduce), reduce), reduce)
    else:
        settle_fn = lambda x: settle(energy, grad, x, reduce)

    def pick_fn(floor, pool):
        shell, _ = inner_shell(pool, energy, images, per, floor)
        if not shell:
            return None
        idx, _ = apply_filter(shell, funcs)
        return shell[idx[0]]

    search = harvest_radius if SEARCH == "radius" else harvest
    floor, pool, info = search(energy, grad, D, per, p, seed, settle_fn, pick_fn, images)
    shell, r_min = inner_shell(pool, energy, images, per, floor)

    exact = False
    got = refine_on_grid(energy_batch, shell, floor, per, reduce)
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
           "n_auto": len(autos), "restarts": info["restarts"], "nfev": info["nfev"],
           "hits": info["hits"], "stalled": info["stalled"], "converged": info["converged"],
           "search": SEARCH, "stop": info.get("stop"),
           "w": None if not weighted else [float(a) for a in w]}

    if not shell:
        rec.update({"groups": None, "filter": None, "one_group": None,
                    "used": [], "energy_at_pick": None, "pick": None,
                    "secs": round(time.time() - t0, 1)})
        print("row %2d  p %d  no floor points found" % (row, p), flush=True)
        return rec

    idx, used = apply_filter(shell, functions(edges, n, w, p))
    orb = groups([shell[i] for i in idx], images)
    rec.update({"groups": len(orb), "filter": int(len(idx)),
                "one_group": len(orb) == 1, "used": used,
                "energy_at_pick": round(float(energy(shell[idx[0]])), 8),
                "pick": [round(float(a), 10) for a in shell[idx[0]]],
                "secs": round(time.time() - t0, 1)})

    print("row %2d  p %d  %-10s t%-2s floor %9.4f  r %8s  pool %4d  shell %4d %-11s "
          "|Aut| %2d  filter %3d  groups %2d  %5.0f s"
          % (row, p, "weighted" if weighted else "unweighted",
             "-" if trial is None else trial, floor,
             "-" if r_min is None else "%.5f" % r_min, len(pool), len(shell),
             "exact" if exact else "lower bound", len(autos), len(idx), len(orb),
             rec["secs"]), flush=True)
    print("        functions used: %s   restarts %d  hits on pick %d  %s"
          % (", ".join(used) or "none", info["restarts"], info["hits"],
             "converged" if info["converged"] else info.get("stop", "hit MAX_RESTARTS")), flush=True)
    return rec

def main():
    if len(sys.argv) >= 3:
        p = int(sys.argv[1])
        weighted = bool(int(sys.argv[2]))
        rows = [int(a) for a in sys.argv[3:]] or ROWS
    else:
        p, weighted, rows = DEFAULT_P, bool(DEFAULT_WEIGHTED), ROWS
    allowed = LOOP_RESTARTS if SEARCH == "radius" else RESTARTS
    if p not in allowed:
        raise SystemExit("p must be one of %s" % sorted(allowed))
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