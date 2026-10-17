"""
Usage:  python src/run_problems.py [problem] [row ...]
Example: python src/run_problems.py mwis 10 11
With no arguments it runs check(), then every problem and row in RUNS.

Problems:
  mwis               maximum weighted independent set
  max2sat            weighted MAX-2-SAT, one clause per edge
  spinglass          Ising spin glass with random couplings and random fields
  spinglass_nofield  the same couplings with no fields, so no one-qubit terms (like Max-Cut with signed weights)
"""

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from scipy.linalg import eigh
from scipy.optimize import minimize

import run_filter as rf

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

P = 1
RUNS = {"mwis": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "max2sat": [10, 11, 12, 13, 14],
        "spinglass": [10, 11, 12, 13, 14],
        "spinglass_nofield": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}
PENALTY = 1.5 # MWIS: P = PENALTY * max node weight (P > max weight makes the lowest C an independent set)
MIN_C = 0.125 # MAX-2-SAT: redraw the literal signs until every node coefficient is at least this big
CHECK = True # run check() before the runs when there are no command line arguments
CHECK_ROWS = [10, 13, 16]
CHECK_PICKS = [(10, 0), (13, 0), (15, 0), (17, 0)] # weighted Max-Cut cases run through both pipelines
TIE_THETA = 1e-5 # a gamma whose phase 2 c gamma is this close to +-pi/4 (or a beta this close to +-pi/4) is put exactly on it
REFINE_TOL = 1e-6 # flat valleys: refine every new floor point until the radius gradient along the valley is below this
REFINE_FINAL = 1e-9 # and the final floor points until it is below this
REFINE_ITERS = 400 # most refine steps for one point
PARITY = False # True turns snapping and refine off, which makes run() give exactly rf.run's results on Max-Cut

# Max-Cut written as Z terms: C = sum_(u,v) (w/2) Z_u Z_v - sum w/2.
def maxcut(edges, n, w):
    edges = rf.canonical_edges(edges)
    w = np.asarray(w, float)

    def value(X):
        return -np.array([sum(w[e] for e, (u, v) in enumerate(edges) if x[u] != x[v]) for x in X])

    return {"problem": "maxcut", "n": n, "edges": edges, "terms": list(edges), "c": w / 2,
            "const": -0.5 * w.sum(), "value": value, "w": w}

# Maximum weighted independent set: C = -sum_j w_j x_j + P sum_(u,v) x_u x_v.
def mwis(edges, n, w):
    edges = rf.canonical_edges(edges)
    m = len(edges)
    w = np.asarray(w, float)
    pen = PENALTY * w.max()
    deg = np.bincount(np.array(edges).ravel(), minlength=n)
    c = np.concatenate([np.full(m, pen / 4), w / 2 - pen * deg / 4])

    def value(X):
        X = np.asarray(X)
        return -(X @ w) + pen * sum(X[:, u] * X[:, v] for (u, v) in edges)

    return {"problem": "mwis", "n": n, "edges": edges, "terms": edges + [(j,) for j in range(n)],
            "c": c, "const": pen * m / 4 - w.sum() / 2, "value": value, "w": w, "pen": pen}

# Weighted MAX-2-SAT with one clause per edge.  Clause (u, v) is (l_u or l_v), where l_j is x_j when its
# sign is +1 and (not x_j) when it is -1.  A literal is false when (1 + s Z)/2 = 1.
def max2sat(edges, n, signs, a):
    edges = rf.canonical_edges(edges)
    m = len(edges)
    s = np.asarray(signs, int)
    a = np.asarray(a, float)
    cz = np.zeros(n)
    for e, (u, v) in enumerate(edges):
        cz[u] += a[e] * s[e, 0] / 4
        cz[v] += a[e] * s[e, 1] / 4
    c = np.concatenate([a * s[:, 0] * s[:, 1] / 4, cz])

    def value(X):
        X = np.asarray(X)
        sat = np.zeros(len(X))
        for e, (u, v) in enumerate(edges):
            lu = X[:, u] if s[e, 0] == 1 else 1 - X[:, u]
            lv = X[:, v] if s[e, 1] == 1 else 1 - X[:, v]
            sat += a[e] * ((lu + lv) > 0)
        return -sat

    return {"problem": "max2sat", "n": n, "edges": edges, "terms": edges + [(j,) for j in range(n)],
            "c": c, "const": -0.75 * a.sum(), "value": value, "signs": s, "a": a}

# Ising spin glass: C = sum_(u,v) J_uv Z_u Z_v + sum_j h_j Z_j.  With h = None there are no one-qubit terms.
def spinglass(edges, n, J, h=None):
    edges = rf.canonical_edges(edges)
    J = np.asarray(J, float)
    hv = np.zeros(n) if h is None else np.asarray(h, float)

    def value(X):
        Zv = 1 - 2 * np.asarray(X)
        return sum(J[e] * Zv[:, u] * Zv[:, v] for e, (u, v) in enumerate(edges)) + Zv @ hv

    if h is None:
        return {"problem": "spinglass_nofield", "n": n, "edges": edges, "terms": list(edges), "c": J,
                "const": 0.0, "value": value, "J": J, "h": None}
    return {"problem": "spinglass", "n": n, "edges": edges, "terms": edges + [(j,) for j in range(n)],
            "c": np.concatenate([J, hv]), "const": 0.0, "value": value, "J": J, "h": hv}

# Build one problem on the graph in row `row` of MaxCutMAQAOAData.csv.
def instance(problem, df, row):
    edges = rf.canonical_edges([tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])])
    n = int(df.loc[row, "Number of Nodes"])
    m = len(edges)
    if problem == "mwis":
        prob = mwis(edges, n, rf.draw_weights(n, 100 * row + 50))
    elif problem == "max2sat":
        a = rf.draw_weights(m, 100 * row + 60)
        rng = np.random.default_rng(100 * row + 70)
        for tries in range(1, 100001):
            prob = max2sat(edges, n, rng.choice([-1, 1], size=(m, 2)), a)
            if np.abs(prob["c"][m:]).min() >= MIN_C:
                prob["tries"] = tries
                break
        else:
            raise SystemExit("row %d: no MAX-2-SAT signs with every node coefficient >= %g" % (row, MIN_C))
    elif problem in ("spinglass", "spinglass_nofield"):
        rng = np.random.default_rng(100 * row + 90)
        J = rng.choice([-1, 1], size=m) * rf.draw_weights(m, 100 * row + 80)
        h = rng.choice([-1, 1], size=n) * rf.draw_weights(n, 100 * row + 85)
        prob = spinglass(edges, n, J, h if problem == "spinglass" else None)
    else:
        raise SystemExit("unknown problem %s" % problem)
    prob["row"] = row
    prob["type"] = df.loc[row, "Type"]
    return prob

# Value of every Z term on every bitstring (rows are terms, columns are bitstrings).
def z_table(n, terms):
    dim = 1 << n
    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    spin = 1 - 2 * bits
    return np.stack([np.prod(spin[:, list(S)], axis=1) for S in terms], 0).astype(float)

# Make an energy function for a given set of Z terms, coefficients, and a constant.
def make_energy(n, terms, c, const=0.0, p=1):
    T = len(terms)
    c = np.asarray(c, float)
    k = 2.0 * c
    D = p * (T + n)
    dim = 1 << n
    zs = z_table(n, terms)
    inv = 1.0 / np.sqrt(dim)
    zero_idx = [np.where(((np.arange(dim) >> j) & 1) == 0)[0] for j in range(n)]
    kzs = zs * k[:, None]
    diag = 0.5 * kzs.sum(0)
    shapes = [(dim >> (j + 1), 2, 1 << j) for j in range(n)]

    def _evolve(X):
        B = X.shape[0]
        gammas = X[:, :p * T].reshape(B, p, T)
        betas = X[:, p * T:].reshape(B, p, n)
        psi = np.full((B, dim), inv, dtype=complex)
        for layer in range(p):
            psi *= np.exp(1j * (gammas[:, layer, :] @ kzs))
            for j in range(n):
                b = betas[:, layer, j][:, None]
                cb = np.cos(b)
                sb = 1j * np.sin(b)
                i0 = zero_idx[j]
                i1 = i0 ^ (1 << j)
                a0 = psi[:, i0]
                a1 = psi[:, i1]
                psi[:, i0] = cb * a0 + sb * a1
                psi[:, i1] = sb * a0 + cb * a1
        return psi

    def energy_batch(X):
        X = np.atleast_2d(X)
        prob = np.abs(_evolve(X)) ** 2
        return 0.5 * np.einsum('bk,ek->b', prob, kzs) + const

    def _rot(psi, j, cb, sb):
        v = psi.reshape(shapes[j])
        a0 = v[:, 0, :].copy()
        a1 = v[:, 1, :]
        v[:, 0, :] = cb * a0 + sb * a1
        v[:, 1, :] = sb * a0 + cb * a1

    def _forward(x):
        gam = x[:p * T].reshape(p, T)
        bet = x[p * T:].reshape(p, n)
        psi = np.full(dim, inv, dtype=complex)
        phases = []
        for layer in range(p):
            ph = np.exp(1j * (gam[layer] @ kzs))
            phases.append(ph)
            psi *= ph
            for j in range(n):
                _rot(psi, j, np.cos(bet[layer, j]), 1j * np.sin(bet[layer, j]))
        return psi, phases, bet

    def energy(x):
        psi = _forward(np.asarray(x, float))[0]
        return float(diag @ (psi.real ** 2 + psi.imag ** 2) + const)

    def grad(x, with_energy=False):
        x = np.asarray(x, float)
        psi, phases, bet = _forward(x)
        e = float(diag @ (psi.real ** 2 + psi.imag ** 2) + const)
        lam = diag * psi
        g_gam = np.empty((p, T))
        g_bet = np.empty((p, n))
        for layer in range(p - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                vl = lam.reshape(shapes[j])
                vp = psi.reshape(shapes[j])
                g_bet[layer, j] = -2.0 * (np.vdot(vl[:, 0, :], vp[:, 1, :])
                                          + np.vdot(vl[:, 1, :], vp[:, 0, :])).imag
                cb, sb = np.cos(bet[layer, j]), -1j * np.sin(bet[layer, j])
                _rot(psi, j, cb, sb)
                _rot(lam, j, cb, sb)
            g_gam[layer] = -2.0 * k * (zs @ (lam.conj() * psi).imag)
            back = phases[layer].conj()
            psi *= back
            lam *= back
        g = np.concatenate([g_gam.ravel(), g_bet.ravel()])
        return (e, g) if with_energy else g

    def state(x):
        return _forward(np.asarray(x, float))[0]

    return energy, energy_batch, grad, D, state

# Per-coordinate period of the energy: pi/(2|c_a|) on gammas, pi on betas.
def periods(terms, n, p, c):
    g = np.tile(np.pi / np.abs(2.0 * np.asarray(c, float)), p)
    return np.concatenate([g, np.full(p * n, np.pi)])

# Reduce a point to the fundamental domain of the energy.
def reduce_point(x, terms, n, c, p=1):
    T = len(terms)
    k2 = np.abs(2.0 * np.asarray(c, float))
    h = np.pi / (2.0 * k2)
    tol = np.full(T, rf.TIE_TOL) if PARITY else TIE_THETA / k2
    gam = np.asarray(x, float)[:p * T].reshape(p, T).copy()
    bet = np.asarray(x, float)[p * T:].reshape(p, n).copy()
    for layer in range(p):
        k = np.round(bet[layer] / (np.pi / 2)).astype(int)
        bet[layer] -= k * np.pi / 2
        for j in np.where(k % 2 != 0)[0]:
            for a, S in enumerate(terms):
                if j in S:
                    gam[:layer + 1, a] *= -1
    flip = np.zeros(n, dtype=int)
    for layer in range(p):
        k = np.round(gam[layer] / h).astype(int)
        gam[layer] -= k * h
        tie = gam[layer] < -h / 2 + tol
        gam[layer][tie] += h[tie]
        k[tie] -= 1
        if not PARITY:
            top = gam[layer] > h / 2 - tol
            gam[layer][top] = h[top] / 2
        for a, S in enumerate(terms):
            if k[a] % 2:
                for j in S:
                    flip[j] ^= 1
        bet[layer] = rf.fold(np.where(flip == 1, -bet[layer], bet[layer]), np.pi)
    if not PARITY:
        on = np.abs(np.abs(bet) - np.pi / 4) < TIE_THETA
        bet[on] = np.sign(bet[on]) * np.pi / 4
    return np.concatenate([gam.ravel(), bet.ravel()])

# Same as rf.spellings for Z terms: every same-radius version of x, from betas sitting on +-pi/4.
def spellings(x, terms, n, c, p=1, cap=rf.SPELL_CAP):
    T = len(terms)
    q = np.pi / 4
    x = reduce_point(x, terms, n, c, p)
    b = x[p * T:]
    on = np.abs(np.abs(b) - q) < rf.TIE_TOL
    b[on] = np.sign(b[on]) * q
    key = lambda y: tuple(np.round(y, rf.CANON))
    found = {key(x): x}
    todo = [x]
    while todo and len(found) < cap:
        z = todo.pop()
        for kk in np.where(np.abs(np.abs(z[p * T:]) - q) < rf.TIE_TOL)[0]:
            layer, j = divmod(int(kk), n)
            y = z.copy()
            y[p * T + kk] = -y[p * T + kk]
            for a, S in enumerate(terms):
                if j in S:
                    for l in range(layer + 1):
                        y[l * T + a] = -y[l * T + a]
            y = reduce_point(y, terms, n, c, p)
            if key(y) not in found:
                found[key(y)] = y
                todo.append(y)
    return list(found.values())

# Find the symmetry group of a given problem.
def symmetry_group(n, terms, c, p=1):
    T = len(terms)
    H = nx.Graph()
    for j in range(n):
        H.add_node(j, kind=0, c=0.0)
    for a, S in enumerate(terms):
        H.add_node(n + a, kind=1, c=float(c[a]))
        for j in S:
            H.add_edge(j, n + a)
    nm = lambda u, v: u["kind"] == v["kind"] and abs(u["c"] - v["c"]) < 1e-12
    autos = list(GraphMatcher(H, H, node_match=nm).isomorphisms_iter())
    perms = [([s[j] for j in range(n)], [s[n + a] - n for a in range(T)]) for s in autos]

    def induced(perm, x):
        sig, tau = perm
        gam = x[:p * T].reshape(p, T)
        bet = x[p * T:].reshape(p, n)
        gb = np.empty((p, T))
        bb = np.empty((p, n))
        for j in range(n):
            bb[:, sig[j]] = bet[:, j]
        for a in range(T):
            gb[:, tau[a]] = gam[:, a]
        return np.concatenate([gb.ravel(), bb.ravel()])

    def images(x):
        out = []
        for perm in perms:
            y = induced(perm, x)
            out += spellings(y, terms, n, c, p)
            out += spellings(-y, terms, n, c, p)
        return out

    return autos, images

# Compute a list of functions for a given problem.
def functions(n, terms, c, p=1):
    T = len(terms)
    k = 2.0 * np.asarray(c, float)
    two = [a for a, S in enumerate(terms) if len(S) == 2]
    one = [a for a, S in enumerate(terms) if len(S) == 1]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a in two:
        G.add_edge(*terms[a], weight=float(k[a]))
    deg = np.array([G.degree(j) for j in range(n)], dtype=float)
    wdeg = np.array([G.degree(j, weight="weight") for j in range(n)], dtype=float)
    pairs = [terms[a] for a in two]
    edeg = np.array([deg[u] + deg[v] for (u, v) in pairs])
    eprod = np.array([deg[u] * deg[v] for (u, v) in pairs])
    ewdeg = np.array([wdeg[u] + wdeg[v] for (u, v) in pairs])
    tri = np.array([len(list(nx.common_neighbors(G, u, v))) for (u, v) in pairs], dtype=float)
    nodes = [terms[a][0] for a in one]
    kv = k[two]
    kh = k[one]

    base = [("sum_gamma", two, np.ones(len(two)))]
    if one:
        base += [("sum_hgamma", one, np.ones(len(one)))]
    base += [("sum_beta", "b", np.ones(n)), ("deg_gamma", two, edeg)]
    if one:
        base += [("deg_hgamma", one, deg[nodes])]
    base += [("deg_beta", "b", deg), ("prod_gamma", two, eprod), ("tri_gamma", two, tri)]
    if len(two) and not np.allclose(kv, kv[0]):
        base += [("w_gamma", two, kv), ("wdeg_gamma", two, ewdeg), ("wdeg_beta", "b", wdeg)]
    if one and not np.allclose(kh, kh[0]):
        base += [("k_hgamma", one, kh)]

    D = p * (T + n)
    out = []
    for name, side, vec in base:
        for layer in range(p):
            v = np.zeros(D)
            if side == "b":
                v[p * T + layer * n:p * T + (layer + 1) * n] = vec
            else:
                v[layer * T + np.array(side, dtype=int)] = vec
            tag = name if p == 1 else "%s_L%d" % (name, layer)
            out.append((tag, (lambda v: (lambda x: float(v @ x)))(v)))
            out.append((tag + "_cube", (lambda v: (lambda x: float(v @ (x ** 3))))(v)))
    return out

# Make an energy function for a given set of Z terms, coefficients, and a constant, counting every gradient call.
def refine(energy, grad, x, reduce, tol, iters=REFINE_ITERS):
    x = reduce(x)
    e0 = energy(x)
    for _ in range(iters):
        try:
            ev, V = eigh(rf.hessian(grad, x))
        except np.linalg.LinAlgError:
            break
        flat = ev < rf.EV_ZERO * max(1.0, ev[-1])
        if not flat.any():
            break
        N, M = V[:, flat], V[:, ~flat]
        t = N.T @ x
        if np.abs(t).max() < tol:
            break
        y0 = x - N @ t

        def obj(z):
            e, g = grad(y0 + M @ z, True)
            return e, M.T @ g

        z = minimize(obj, np.zeros(M.shape[1]), jac=True, method="L-BFGS-B",
                     options={"ftol": 1e-16, "gtol": 1e-13, "maxiter": 2000}).x
        y = y0 + M @ z
        if energy(y) > e0 + rf.TOL_E:
            break
        x = y
    return reduce(x)

# Exact answer by checking every bitstring with the problem's own definition (not the Z terms).
# Returns the lowest C and the bitstrings (as integers, bit j = node j) that reach it.
def exact(prob):
    n = prob["n"]
    X = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)
    v = np.asarray(prob["value"](X), float)
    best = float(v.min())
    return best, [int(i) for i in np.where(v <= best + 1e-9)[0]]

# MWIS weight by checking every independent set directly (no penalty), as a second check on exact().
def mwis_direct(prob):
    n, w = prob["n"], prob["w"]
    best, sets = -1.0, []
    for mask in range(1 << n):
        if any((mask >> u) & 1 and (mask >> v) & 1 for (u, v) in prob["edges"]):
            continue
        s = sum(w[j] for j in range(n) if (mask >> j) & 1)
        if s > best + 1e-9:
            best, sets = s, [mask]
        elif abs(s - best) <= 1e-9:
            sets.append(mask)
    return best, sets

# Same as rf.run for a problem written as Z terms, including the two-seed guard (rf.guarded) when rf.GUARD
# is True.  Returns one record; "confirmed" says whether two seeds confirmed the floor and the pick.
def run(prob, p, seed):
    t0 = time.time()
    n, terms, c, const = prob["n"], prob["terms"], prob["c"], prob["const"]
    T = len(terms)
    energy, energy_batch, grad, D, state = make_energy(n, terms, c, const, p)
    per = periods(terms, n, p, c)
    autos, images = symmetry_group(n, terms, c, p)
    funcs = functions(n, terms, c, p)
    reduce = lambda x: reduce_point(x, terms, n, c, p)
    if p >= rf.SLIDE_P:
        chain = lambda x: rf.settle(energy, grad, rf.slide(energy, grad, rf.settle(energy, grad, x, reduce), reduce), reduce)
    else:
        chain = lambda x: rf.settle(energy, grad, x, reduce)
    settle_fn = chain if PARITY else (lambda x: refine(energy, grad, chain(x), reduce, REFINE_TOL))

    def pick_fn(floor, pool):
        shell, _ = rf.inner_shell(pool, energy, images, per, floor)
        if not shell:
            return None
        idx, _ = rf.apply_filter(shell, funcs)
        return shell[idx[0]]

    # One seed: the search, then every floor point refined to the closest point of its flat valley.
    def one(s):
        floor, pool, info = rf.harvest(energy, grad, D, per, p, s, settle_fn, pick_fn, images)
        if not PARITY:
            tight = []
            for y in pool:
                rf.add_distinct(tight, refine(energy, grad, y, reduce, REFINE_FINAL), per)
            pool = tight
        return floor, pool, info

    floor, pool, info = rf.guarded(one, seed, pick_fn, images, per) if rf.GUARD else one(seed)
    shell, r_min = rf.inner_shell(pool, energy, images, per, floor)
    best, opt = exact(prob)
    rec = {"problem": prob["problem"], "row": prob.get("row"), "n": n, "T": T, "p": p, "D": D,
           "seed": seed, "floor": round(floor, 10), "best": round(best, 10),
           "ratio": round(floor / best, 6) if best != 0 else None, "opt": opt,
           "r_min": None if r_min is None else round(r_min, 6), "pool": len(pool), "shell": len(shell),
           "n_auto": len(autos), "restarts": info["restarts"], "nfev": info["nfev"], "hits": info["hits"],
           "stalled": info["stalled"], "converged": info["converged"], "seeds": info.get("seeds", [seed]),
           "seed_floors": info.get("seed_floors"), "confirmed": info.get("confirmed")}
    if not shell:
        rec.update({"filter": None, "groups": None, "used": [], "pick": None, "secs": round(time.time() - t0, 1)})
        print("%-9s row %2d  no floor points found" % (prob["problem"], prob.get("row", -1)), flush=True)
        return rec

    idx, used = rf.apply_filter(shell, funcs)
    orb = rf.groups([shell[i] for i in idx], images)
    pick = np.round(shell[idx[0]], 10)
    ev = np.linalg.eigvalsh(rf.hessian(grad, pick))
    psi = state(pick)
    rec.update({"filter": int(len(idx)), "groups": len(orb), "one_group": len(orb) == 1, "used": used,
                "energy_at_pick": float(energy(pick)), "grad_at_pick": float(np.abs(grad(pick)).max()),
                "hess_min": float(ev[0]), "flat": int((np.abs(ev) < rf.EV_ZERO * max(1.0, ev[-1])).sum()),
                "p_opt": float(sum(abs(psi[i]) ** 2 for i in opt)),
                "pick": [float(a) for a in pick], "secs": round(time.time() - t0, 1)})
    print("%-9s row %2d  seed %5d  floor %10.5f  best %10.5f  ratio %.4f  r %8.5f  pool %4d  shell %3d  "
          "|Aut| %d  filter %d  groups %d  %5.0f s"
          % (prob["problem"], prob.get("row", -1), seed, floor, best, rec["ratio"] or 0, r_min, len(pool),
             len(shell), len(autos), len(idx), len(orb), rec["secs"]), flush=True)
    print("          functions used: %s   restarts %d  hits on pick %d  %s   |grad| %.1e  min eig %.1e  "
          "P(optimum) %.4f"
          % (", ".join(used) or "none", info["restarts"], info["hits"],
             "converged" if info["converged"] else "hit MAX_RESTARTS", rec["grad_at_pick"], rec["hess_min"],
             rec["p_opt"]), flush=True)
    if rf.GUARD:
        print("          guard: seeds %s  floors %s  %s"
              % (info["seeds"], ["%.6f" % f for f in info["seed_floors"]],
                 "confirmed by %d seeds" % info["agree"] if info["confirmed"] else "NOT CONFIRMED, do not use"), flush=True)
    return rec

# Is b one of the images of a (same point up to the symmetries)?
def same_pick(a, b, images, per):
    if a is None or b is None:
        return False
    return bool(min(rf.geodesic_dist(y, np.array(b), per) for y in images(np.array(a))) <= rf.TOL_D)

# Energy from dense matrices and scipy expm, written separately from make_energy.
def dense_energy(prob, x, p):
    from scipy.linalg import expm
    n, terms, c = prob["n"], prob["terms"], np.asarray(prob["c"], float)
    T = len(terms)
    I2 = np.eye(2)
    X1 = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z1 = np.diag([1.0, -1.0])

    def op(single, q):
        out = np.array([[1.0]])
        for j in range(n - 1, -1, -1):
            out = np.kron(out, single if j == q else I2)
        return out

    Zs = [op(Z1, q) for q in range(n)]
    Xs = [op(X1, q) for q in range(n)]
    ZS = [np.linalg.multi_dot([Zs[j] for j in S] + [np.eye(1 << n)]) for S in terms]
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)
    Cm = np.diag(np.asarray(prob["value"](bits), float))
    psi = np.ones(1 << n, complex) / np.sqrt(1 << n)
    for layer in range(p):
        g = x[layer * T:(layer + 1) * T]
        b = x[p * T + layer * n:p * T + (layer + 1) * n]
        psi = expm(1j * sum(2 * g[a] * c[a] * ZS[a] for a in range(T))) @ psi
        psi = expm(1j * sum(b[j] * Xs[j] for j in range(n))) @ psi
    return float(np.real(psi.conj() @ Cm @ psi))

# Energy from a Qiskit circuit built with rzz, rz and rx, as in the docstring.
def qiskit_energy(prob, x, p):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    n, terms, c = prob["n"], prob["terms"], np.asarray(prob["c"], float)
    T = len(terms)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for a, S in enumerate(terms):
            g = x[layer * T + a]
            if len(S) == 2:
                qc.rzz(-4 * g * c[a], S[0], S[1])
            else:
                qc.rz(-4 * g * c[a], S[0])
        for j in range(n):
            qc.rx(-2 * x[p * T + layer * n + j], j)
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)
    return float(Statevector(qc).probabilities() @ np.asarray(prob["value"](bits), float))

# Add half a period to gamma_a in one layer and flip the betas of its qubits in that layer and every
# later one (flips=False skips the flips, which should change the energy).
def shift_gamma(x, prob, p, layer, a, flips=True):
    n, terms, c = prob["n"], prob["terms"], prob["c"]
    T = len(terms)
    y = np.array(x, float)
    y[layer * T + a] += np.pi / (4 * abs(c[a]))
    if flips:
        for l in range(layer, p):
            for j in terms[a]:
                y[p * T + l * n + j] *= -1
    return y

# Add pi/2 to beta_j in one layer and flip the gammas of every term with j in that layer and every
# earlier one.
def shift_beta(x, prob, p, layer, j, flips=True):
    n, terms = prob["n"], prob["terms"]
    T = len(terms)
    y = np.array(x, float)
    y[p * T + layer * n + j] += np.pi / 2
    if flips:
        for l in range(layer + 1):
            for a, S in enumerate(terms):
                if j in S:
                    y[l * T + a] *= -1
    return y

# Check that the Z-term code matches run_filter.py on a few problems, and then run each problem in RUNS on the given rows.
def check(rows=CHECK_ROWS):
    global PARITY
    df = pd.read_csv(rf.CSV)
    rng = np.random.default_rng(12345)
    keep, PARITY = PARITY, True
    print("1. Max-Cut through the Z-term code against run_filter.py (PARITY = True)", flush=True)
    worst = {"energy": 0.0, "grad": 0.0, "batch": 0.0, "fold": 0.0, "period": 0.0, "functions": 0.0}
    names_ok, autos_ok, images_ok = True, True, True
    for row in range(10, 20):
        edges = rf.canonical_edges([tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])])
        n = int(df.loc[row, "Number of Nodes"])
        m = len(edges)
        for w in (np.ones(m), rf.draw_weights(m, 100 * row)):
            prob = maxcut(edges, n, w)
            for p in (1, 2):
                E0, B0, G0, D0 = rf.make_energy(n, edges, w=w, p=p)
                E1, B1, G1, D1, _ = make_energy(n, prob["terms"], prob["c"], prob["const"], p)
                f0 = rf.functions(edges, n, w, p)
                f1 = functions(n, prob["terms"], prob["c"], p)
                names_ok &= [a for a, _ in f0] == [a for a, _ in f1] and D0 == D1
                worst["period"] = max(worst["period"], np.abs(rf.periods(m, n, p, w) - periods(prob["terms"], n, p, prob["c"])).max())
                X = rng.uniform(-np.pi, np.pi, (3, D0))
                worst["batch"] = max(worst["batch"], np.abs(B0(X) - B1(X)).max())
                for x in X:
                    worst["energy"] = max(worst["energy"], abs(E0(x) - E1(x)))
                    worst["grad"] = max(worst["grad"], np.abs(G0(x) - G1(x)).max())
                    worst["fold"] = max(worst["fold"], np.abs(rf.reduce_point(x, edges, n, w, p) - reduce_point(x, prob["terms"], n, prob["c"], p)).max())
                    worst["functions"] = max(worst["functions"], max(abs(a(x) - b(x)) for (_, a), (_, b) in zip(f0, f1)))
                if p == 1:
                    a0, im0 = rf.symmetry_group(edges, n, w, p)
                    a1, im1 = symmetry_group(n, prob["terms"], prob["c"], p)
                    autos_ok &= len(a0) == len(a1)
                    x = rng.uniform(-np.pi, np.pi, D0)
                    s0, s1 = im0(x), im1(x)
                    per = periods(prob["terms"], n, p, prob["c"])
                    images_ok &= len(s0) == len(s1) and all(min(rf.geodesic_dist(y, z, per) for z in s1) < 1e-12 for y in s0)
    print("   largest difference  energy %.1e  gradient %.1e  batch %.1e  folding %.1e  periods %.1e  functions %.1e"
          % tuple(worst[k] for k in ("energy", "grad", "batch", "fold", "period", "functions")), flush=True)
    print("   same function names %s   same |Aut| %s   same images %s   (rows 10-19, unweighted and weighted, p=1 and 2)"
          % (names_ok, autos_ok, images_ok), flush=True)
    PARITY = keep

    print("2-5. Each problem on rows %s" % rows, flush=True)
    print("   %-9s %4s %9s %9s %9s %9s %9s %9s %9s %9s %9s" % ("problem", "row", "terms", "exact", "dense",
          "qiskit", "fd grad", "gamma sym", "beta sym", "mirror", "no flips"))
    for problem in RUNS:
        for row in rows:
            prob = instance(problem, df, row)
            n, terms, c, const = prob["n"], prob["terms"], prob["c"], prob["const"]
            T = len(terms)
            bits = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)
            d_terms = np.abs(const + np.asarray(c) @ z_table(n, terms) - prob["value"](bits)).max()
            best, opt = exact(prob)
            d_exact = 0.0
            if problem == "mwis":
                wb, sets = mwis_direct(prob)
                d_exact = abs(best + wb) + (0.0 if sorted(sets) == sorted(opt) else 1.0)
            d_dense = d_qk = d_fd = d_g = d_b = d_m = 0.0
            no_flip = np.inf
            for p in (1, 2):
                E, B, G, D, _ = make_energy(n, terms, c, const, p)
                for x in rng.uniform(-np.pi, np.pi, (2, D)):
                    e = E(x)
                    d_dense = max(d_dense, abs(e - dense_energy(prob, x, p)))
                    d_qk = max(d_qk, abs(e - qiskit_energy(prob, x, p)))
                    fd = np.array([(E(x + 1e-6 * np.eye(D)[i]) - E(x - 1e-6 * np.eye(D)[i])) / 2e-6 for i in range(D)])
                    d_fd = max(d_fd, np.abs(fd - G(x)).max())
                    d_m = max(d_m, abs(E(-x) - e), abs(E(reduce_point(x, terms, n, c, p)) - e))
                    for layer in range(p):
                        for a in range(T):
                            d_g = max(d_g, abs(E(shift_gamma(x, prob, p, layer, a)) - e))
                            no_flip = min(no_flip, abs(E(shift_gamma(x, prob, p, layer, a, False)) - e))
                        for j in range(n):
                            d_b = max(d_b, abs(E(shift_beta(x, prob, p, layer, j)) - e))
                            no_flip = min(no_flip, abs(E(shift_beta(x, prob, p, layer, j, False)) - e))
            print("   %-9s %4d %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e"
                  % (problem, row, d_terms, d_exact, d_dense, d_qk, d_fd, d_g, d_b, d_m, no_flip), flush=True)

    print("5. Minimum weighted vertex cover against MWIS", flush=True)
    worst_vc = 0.0
    for row in rows:
        prob = instance("mwis", df, row)
        n, edges, w, pen = prob["n"], prob["edges"], prob["w"], prob["pen"]
        deg = np.bincount(np.array(edges).ravel(), minlength=n)
        vc = {"n": n, "terms": prob["terms"], "c": np.concatenate([np.full(len(edges), pen / 4), -w / 2 + pen * deg / 4]),
              "const": w.sum() / 2 + pen * len(edges) / 4,
              "value": lambda X: np.asarray(X) @ w + pen * sum((1 - np.asarray(X)[:, u]) * (1 - np.asarray(X)[:, v]) for (u, v) in edges)}
        bits = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)
        worst_vc = max(worst_vc, np.abs(vc["const"] + vc["c"] @ z_table(n, vc["terms"]) - vc["value"](bits)).max())
        for p in (1, 2):
            E1 = make_energy(n, prob["terms"], prob["c"], prob["const"], p)[0]
            E2 = make_energy(n, vc["terms"], vc["c"], vc["const"], p)[0]
            for x in rng.uniform(-np.pi, np.pi, (3, p * (len(prob["terms"]) + n))):
                worst_vc = max(worst_vc, abs(E2(x) - E1(x) - w.sum()))
    print("   largest |E_cover - E_mwis - sum w| %.1e" % worst_vc, flush=True)

    print("6. Folding near the boundary: the exact floor point with its boundary coordinates moved by +-3e-6 in phase", flush=True)
    worst_fold, worst_e = 0.0, 0.0
    for problem in ("mwis", "max2sat", "spinglass"):
        for row in rows:
            prob = instance(problem, df, row)
            n, terms, c, const = prob["n"], prob["terms"], np.asarray(prob["c"]), prob["const"]
            T, m = len(terms), len(prob["edges"])
            E = make_energy(n, terms, c, const, 1)[0]
            best, opt = exact(prob)
            z = 1 - 2 * ((opt[0] >> np.arange(n)) & 1)
            x = np.zeros(T + n)
            x[m:T] = np.pi / (8 * np.abs(c[m:T]))
            x[T:] = np.sign(c[m:T]) * z * np.pi / 4
            x0 = reduce_point(x, terms, n, c, 1)
            for _ in range(20):
                y = x.copy()
                y[m:T] += rng.choice([-1, 1], T - m) * 3e-6 / np.abs(2 * c[m:T])
                y[T:] += rng.choice([-1, 1], n) * 3e-6
                y[m:T] *= rng.choice([-1, 1], T - m)
                for a in range(m, T):
                    if y[a] * x[a] < 0:
                        y[T + terms[a][0]] *= -1
                worst_fold = max(worst_fold, np.abs(reduce_point(y, terms, n, c, 1) - x0).max())
                worst_e = max(worst_e, abs(E(y) - best))
    print("   largest difference after folding %.1e   (largest energy change from the moves %.1e)" % (worst_fold, worst_e), flush=True)

# Run weighted Max-Cut cases through rf.run and through run() above with the same seed, and compare.
# PARITY is on, so snapping and refine are off, and the guard is off, so both run one seed.
def check_picks(cases=CHECK_PICKS):
    global PARITY
    keep, PARITY = PARITY, True
    guard, rf.GUARD = rf.GUARD, False
    df = pd.read_csv(rf.CSV)
    for row, t in cases:
        edges = rf.canonical_edges([tuple(e) for e in ast.literal_eval(df.loc[row, "Edges"])])
        n = int(df.loc[row, "Number of Nodes"])
        m = len(edges)
        w = rf.draw_weights(m, 100 * row + t)
        seed = 1000 * row + 10 * P + t + 1
        a = rf.run(row, edges, n, w, P, seed, True, t)
        prob = maxcut(edges, n, w)
        prob["row"] = row
        b = run(prob, P, seed)
        same = [round(a["floor"], 6) == round(b["floor"], 6)]
        same += [a[k] == b[k] for k in ("r_min", "pool", "shell", "filter", "restarts", "hits", "nfev")]
        print("   row %d draw %d  same floor, r_min, pool, shell, filter, restarts, hits, nfev: %s   pick distance %.1e"
              % (row, t, same, np.abs(np.array(a["pick"]) - np.array(b["pick"])).max()), flush=True)
    PARITY = keep
    rf.GUARD = guard

def load(path):
    if not path.exists():
        return []
    try:
        return json.load(open(path))
    except (ValueError, OSError):
        return []

# Given a list of records for a problem, return the last record for a given row that has a pick, and whether the two seeds agreed on the pick and floor.
def label(res, row):
    recs = [d for d in res if d["row"] == row and d.get("pick") is not None]
    new = [d for d in recs if d.get("confirmed") is not None]
    if new:
        return new[-1], bool(new[-1]["confirmed"])
    first = [d for d in recs if d["seed"] % 1000 < 500]
    second = [d for d in recs if d["seed"] % 1000 >= 500]
    if not first:
        return None
    if not second:
        return first[0], None
    return first[0], bool(second[0].get("same_pick")) and bool(second[0].get("same_floor"))

# One csv row per graph, laid out like MaxCutMAQAOAData.csv, plus the instance.
def write_csv(problem, res, df):
    rows = []
    for row in sorted({d["row"] for d in res}):
        got = label(res, row)
        if got is None:
            continue
        d, confirmed = got
        prob = instance(problem, df, d["row"])
        n, T = prob["n"], len(prob["terms"])
        m = len(prob["edges"])
        x = np.array(d["pick"])
        E = make_energy(n, prob["terms"], prob["c"], prob["const"], P)[0]
        out = {"Row": d["row"], "Number of Nodes": n, "Type": prob["type"], "Edges": str(prob["edges"])}
        if problem == "mwis":
            out.update({"Node Weights": str([float(a) for a in prob["w"]]), "Penalty": float(prob["pen"])})
        elif problem == "max2sat":
            out.update({"Literal Signs": str([[int(a), int(b)] for a, b in prob["signs"]]),
                        "Clause Weights": str([float(a) for a in prob["a"]])})
        else:
            out["Couplings"] = str([float(a) for a in prob["J"]])
            if prob["h"] is not None:
                out["Fields"] = str([float(a) for a in prob["h"]])
        gam = x[:P * T].reshape(P, T)
        bet = x[P * T:].reshape(P, n)
        lists = lambda A: str([float(a) for a in A[0]]) if P == 1 else str([[float(a) for a in row] for row in A])
        out.update({"Betas": lists(bet), "Edge Gammas": lists(gam[:, :m]),
                    "Node Gammas": lists(gam[:, m:]), "Energy": E(x), "Radius": d["r_min"],
                    "Lowest Cost": d["best"],
                    "Optimal Bitstrings": str(["".join(str((i >> j) & 1) for j in range(n)) for i in d["opt"]])})
        if problem == "mwis":
            out["MWIS Nodes"] = str([[j for j in range(n) if (i >> j) & 1] for i in d["opt"]])
        out["Same Pick Both Seeds"] = confirmed
        rows.append(out)
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS_DIR / ("%s_p%d.csv" % (problem, P)), index=False)

def main():
    if len(sys.argv) >= 2:
        todo = {sys.argv[1]: [int(a) for a in sys.argv[2:]] or RUNS[sys.argv[1]]}
    else:
        todo = RUNS
        if CHECK:
            check()
            check_picks()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(rf.CSV)
    for problem, rows in todo.items():
        out = RESULTS_DIR / ("%s_p%d.json" % (problem, P))
        res = load(out)
        for row in rows:
            if label(res, row) is not None:
                print("%s row %d already done" % (problem, row), flush=True)
                continue
            prob = instance(problem, df, row)
            rec = run(prob, P, 1000 * row + 10 * P + 1)
            if problem == "max2sat":
                rec["tries"] = prob["tries"]
            res.append(rec)
            json.dump(res, open(out, "w"), indent=1) # checkpoint after every graph
        write_csv(problem, res, df)

        print("\n%-9s %4s %11s %11s %7s %8s %6s %7s %10s %10s"
              % ("problem", "row", "floor", "best", "ratio", "r_min", "shell", "filter", "confirmed", "P(opt)"))
        for row in sorted({d["row"] for d in res}):
            got = label(res, row)
            if got is None:
                continue
            d, confirmed = got
            print("%-9s %4d %11.5f %11.5f %7.4f %8s %6d %7s %10s %10.4f"
                  % (problem, row, d["floor"], d["best"], d["ratio"] or 0, d["r_min"], d["shell"],
                     d["filter"], confirmed, d.get("p_opt", 0)))

if __name__ == "__main__":
    main()