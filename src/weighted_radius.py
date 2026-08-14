"""
weighted radius, positive fraction then weighted radius, and weighted positive radius.
"""

import ast
import json
import itertools
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize, dual_annealing
import maqaoa_core as M

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "MaxCutMAQAOAData.csv"
RESULTS_DIR = ROOT / "results"
OUT = RESULTS_DIR / "weighted_radius.json"
SHELLS = RESULTS_DIR / "shells"
FLOORS = RESULTS_DIR / "floor_cheap.json"

ROWS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
RESTARTS = 3000
MODE = "primes"
MODES = ["linear", "power2", "primes"]
COORD_NOISE = 1e-4
LAMS = [0.0, 0.1, 0.2, 0.35]
TOL_E = 1e-6
TOL_D = 1e-3
RAD_TOL = 1e-4
FEAS = 1e-9
BIG = 1e6
SLACK = 1e-6
PI = np.pi
UNIT = PI / 8

# Fold a point into the geodesic fundamental domain [-pi/2, pi/2).
def fold(x):
    x = np.asarray(x, float)
    return M.geodesic_vec(x, np.zeros(x.shape[-1]))

# Which integer weight vector do we put on the coordinates?
# there are also different weights to try out
def weights(D, mode):
    if mode == "linear":
        return np.arange(1, D + 1, dtype=float)
    if mode == "power2":
        return 2.0 ** np.arange(D)
    if mode == "primes":
        p, k = [], 2
        while len(p) < D:
            if all(k % q for q in p):
                p.append(k)
            k += 1
        return np.array(p, dtype=float)
    raise ValueError(mode)

# Plain Euclidean radius from the origin
def radius(x):
    return M.norm_origin(np.asarray(x, float))

# Weighted radius: sqrt(w_0 * (p_0 - c)^2 + w_1 * (p_1 - c)^2 + ...).
def wradius(x, w):
    v = fold(x)
    return float(np.sqrt(w @ (v * v)))

# Same weights, but only the positive components contribute.
def wradius_pos(x, w):
    v = fold(x)
    p = np.clip(v, 0.0, None)
    return float(np.sqrt(w @ (p * p)))

# Fraction of the radius carried by the positive components.
def pos_frac(x):
    v = fold(x)
    r = np.linalg.norm(v)
    return float(np.linalg.norm(np.clip(v, 0.0, None)) / r) if r > 0 else 0.0

# Brute-force Max-Cut
def true_maxcut(edges, n):
    best = 0
    for b in range(1 << n):
        c = sum(1 for (u, v) in edges if ((b >> u) & 1) != ((b >> v) & 1))
        best = max(best, c)
    return best

# Known floors and Max-Cut values
def load_known():
    if not FLOORS.exists():
        return {}
    return {d["row"]: (float(d["floor"]), -int(d["neg_maxcut"]))
            for d in json.load(open(FLOORS))}

# Add a point only if nothing within TOL_D (mod pi) is already in the list.
def add_distinct(pool, x, tol=TOL_D):
    for q in pool:
        if M.geodesic_dist(x, q) <= tol:
            return False
    pool.append(x)
    return True

# Harvest floor-level minima for later analysis
def harvest(energy, grad, D, rng, restarts):
    E, X = [], []
    for i in range(restarts):
        lam = LAMS[i % len(LAMS)]
        x0 = rng.uniform(0, PI, D)
        if lam:
            def pen(x, l=lam):
                v = fold(x)
                return energy(x) + l * float(v @ v)
            def pgr(x, l=lam):
                v = fold(x)
                return grad(x) + 2 * l * v
            x0 = minimize(pen, x0, jac=pgr, method="L-BFGS-B",
                          options={"ftol": 1e-13, "gtol": 1e-10}).x
        r = M.polish(energy, grad, x0)
        E.append(float(r.fun)); X.append(fold(r.x))
    E = np.array(E)
    return round(float(E.min()), 6), E, X

# Slide a floor-level point downhill in radius while the energy stays at the floor.
def slide(energy, grad, D, x0, floor):
    cons = [{"type": "ineq",
             "fun": lambda x: floor + FEAS - energy(x),
             "jac": lambda x: -grad(x)}]
    r = minimize(lambda x: float(x @ x), fold(x0), jac=lambda x: 2 * x,
                 bounds=[(-PI / 2, PI / 2)] * D, constraints=cons,
                 method="SLSQP", options={"maxiter": 400, "ftol": 1e-14})
    x = fold(r.x)
    return (radius(x), x) if energy(x) <= floor + 1e-7 else (np.inf, None)

# The minimum-radius shell, using energy symmetry
def min_radius_shell(energy, grad, D, floor, at_floor):
    out = []
    for x in at_floor:
        r, xs = slide(energy, grad, D, x, floor)
        if xs is not None:
            out.append((r, xs))
    if not out:
        return np.inf, np.zeros((0, D))
    r_min = min(r for r, _ in out)
    shell = []
    for r, x in sorted(out, key=lambda t: t[0]):
        if r <= r_min + RAD_TOL:
            add_distinct(shell, x)
    for x in list(shell):
        add_distinct(shell, fold(-x))
    return r_min, np.array(shell)

# Enumerate every sign pattern at a fixed magnitude and see if it sits on the floor.
def enumerate_signs(energy, D, mag, floor, unit=UNIT):
    idx = np.nonzero(mag)[0]
    if len(idx) > 20:
        return None
    hits = []
    for signs in itertools.product([1, -1], repeat=len(idx)):
        x = np.zeros(D)
        x[idx] = unit * mag[idx] * np.array(signs)
        if energy(x) <= floor + 1e-9:
            hits.append(x)
    return np.array(hits)


# the true energy inside the caps, a large constant outside.
def make_barrier(energy, w, r_cap, k_cap, use_pos):
    wr = wradius_pos if use_pos else wradius
    def f(x):
        if radius(x) > r_cap + SLACK or wr(x, w) > k_cap + SLACK:
            return BIG
        return energy(x)
    return f

# How much can a weighted sum of squares move under the optimizer's residual coordinate error?
def tie_atol(shell, w):
    v = np.abs(np.array([fold(x) for x in shell]))
    return float(2 * COORD_NOISE * (v @ w).max())

# Count the winners and say how far the runner-up sits, in units of the noise.
# An exact integer key is never ambiguous, and a key with no runner-up at all is
# constant on the shell, which is a definite result rather than an uncertain one.
def winners(vals, atol, maximize=False):
    v = -np.asarray(vals, float) if maximize else np.asarray(vals, float)
    sel = np.where(v <= v.min() + atol)[0]
    rest = np.sort(v[v > v.min() + atol])
    if atol == 0.0 or len(rest) == 0:
        return len(sel), True, np.inf
    margin = (rest[0] - v.min()) / atol
    return len(sel), bool(margin > 10.0), margin

# Format a survivor count together with how trustworthy it is.
def fmt(k, ok, margin):
    if ok and not np.isfinite(margin):
        return "%d" % k
    if ok:
        return "%d (margin %.0fx)" % (k, margin)
    return "%d (unresolved, margin %.1fx)" % (k, margin)

# decrease radius then when it stops decrease weighted radius or weighted positive radius
def shrink(energy, D, w, floor, use_pos, seed, iters=6, maxiter=200):
    wr = wradius_pos if use_pos else wradius
    r_cap, k_cap = np.inf, np.inf
    best = None
    bounds = [(-PI / 2, PI / 2)] * D
    calls = 0
    for stage, which in enumerate(["radius", "second"]):
        for it in range(iters):
            f = make_barrier(energy, w, r_cap, k_cap, use_pos)
            res = dual_annealing(f, bounds, maxiter=maxiter, seed=seed + 17 * it)
            calls += int(res.nfev)
            x = fold(res.x)
            if energy(x) > floor + TOL_E:
                break
            best = x
            if which == "radius":
                new = radius(x)
                if new >= r_cap - 1e-6:
                    break
                r_cap = new
            else:
                new = wr(x, w)
                if new >= k_cap - 1e-6:
                    break
                k_cap = new
        if which == "radius" and best is not None:
            r_cap = radius(best) + RAD_TOL
    return best, calls

# How many Hessian eigenvalues are zero, and how small are the smallest ones?
# The count alone cannot tell a true flat direction from a false zero, so return
# the magnitudes as well.
def zero_modes(grad, x, D, tol=1e-6, h=1e-5, keep=5):
    H = np.zeros((D, D))
    for i in range(D):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        H[:, i] = (grad(xp) - grad(xm)) / (2 * h)
    H = 0.5 * (H + H.T)
    a = np.sort(np.abs(np.linalg.eigvalsh(H)))
    return int((a < tol).sum()), a[:keep]


def run(df, row, known, restarts=RESTARTS, mode=MODE, seed=0):
    edges = [tuple(e) for e in ast.literal_eval(df.loc[row, "Edges"])]
    n = int(df.loc[row, "Number of Nodes"])
    energy, _, grad, D = M.make_energy(n, edges, p=1)
    E = list(nx.Graph(edges).edges()) # same canonical order make_energy uses
    m = len(E)
    floor_known, mc = known.get(row, (None, None))
    if mc is None:
        mc = true_maxcut(E, n)
    w = weights(D, mode)
    rng = np.random.default_rng(31337 + 100 * row)
    t0 = time.time()

    # Cut at the certified floor when there is one
    floor_h, Eh, Xh = harvest(energy, grad, D, rng, restarts)
    floor = floor_h if floor_known is None else min(floor_h, floor_known)
    at_floor = [Xh[i] for i in range(len(Eh)) if Eh[i] <= floor + TOL_E]
    if not at_floor:
        print("row %d: harvest bottomed out at %.6f, above the known floor %.6f"
              % (row, floor_h, floor))
        floor = floor_h
        at_floor = [Xh[i] for i in range(len(Eh)) if Eh[i] <= floor + TOL_E]

    r_min, shell = min_radius_shell(energy, grad, D, floor, at_floor)
    if not len(shell):
        print("row %d: no point survived the slide, nothing to score" % row)
        return None

    # If the shell sits on the pi/8 grid, replace it with the exact enumeration
    U = np.round(shell / UNIT).astype(int)
    on_grid = float(np.abs(shell / UNIT - U).max()) < 1e-3
    exact = False
    if on_grid:
        mags = sorted({tuple(np.abs(u).tolist()) for u in U})
        budget = 600000
        pool = []
        for mg in mags:
            k = int((np.array(mg) != 0).sum())
            budget -= 2 ** k
            if budget < 0:
                pool = None
                break
            h = enumerate_signs(energy, D, np.array(mg), floor)
            if h is None:
                pool = None
                break
            pool.extend(list(h))
        if pool:
            keep = []
            for x in pool:
                add_distinct(keep, x)
            shell = np.array(keep)
            exact = True

    zm, small_eigs = zero_modes(grad, shell[0], D)

    # Compute the radius, weighted radius, weighted positive radius, and positive fraction of each shell point.
    R = np.array([radius(x) for x in shell])
    if exact:
        V = np.round(shell / UNIT).astype(np.int64)
        wi = w.astype(np.int64)
        RW = np.array([wi @ (v * v) for v in V])
        RP = np.array([wi @ (np.clip(v, 0, None) ** 2) for v in V])
        # The radius is constant on the shell, so ranking by the positive fraction
        # is the same as ranking by the sum of squares of the positive components.
        PF = np.array([np.clip(v, 0, None) @ np.clip(v, 0, None) for v in V])
    else:
        RW = np.array([wradius(x, w) ** 2 for x in shell])
        RP = np.array([wradius_pos(x, w) ** 2 for x in shell])
        PF = np.array([pos_frac(x) for x in shell])

    # On the grid every key is an exact integer, so nothing is tied by accident
    # and the tolerance is zero. Off the grid it has to come from the noise.
    if exact:
        atol_w = atol_pf = 0.0
    else:
        atol_w = tie_atol(shell, w)
        atol_pf = COORD_NOISE / max(r_min, 1e-12)

    # Experiment A: radius then weighted radius.
    a, a_ok, a_marg = winners(RW, atol_w)
    # Experiment B: radius, then largest positive fraction, then weighted radius.
    # B is only as trustworthy as the positive fraction step that feeds it.
    _, pf_ok, pf_marg = winners(PF, atol_pf, maximize=True)
    keep = np.where(PF >= PF.max() - atol_pf)[0]
    b, b_step_ok, b_marg = winners(RW[keep], atol_w)
    b_ok = bool(pf_ok and b_step_ok)
    b_marg = min(pf_marg, b_marg)
    # Experiment C: radius then weighted positive radius.
    c, c_ok, c_marg = winners(RP, atol_w, maximize=True)
    pick = shell[int(np.argmax(RP))]

    print("\n=== row %d  (n=%d, m=%d, D=%d, weights=%s) ===" % (row, n, m, D, mode))
    print("  Max-Cut %d, floor %.6f, gap %+.3f, %.1f s"
          % (mc, floor, mc + floor, time.time() - t0))
    print("  min radius %.6f = %.6f pi,  R^2 in (pi/4)^2 units %.4f"
          % (r_min, r_min / PI, (r_min / (PI / 4)) ** 2))
    print("  shell size %d  (%s)"
          % (len(shell), "exact by enumeration" if exact
             else "restart-limited, lower bound"))
    print("  flat directions at a shell point %d of %d, smallest |eig| %s"
          % (zm, D, " ".join("%.1e" % v for v in small_eigs)))
    print("  distinct values on the shell: radius %d, weighted radius %d, "
          "positive fraction %d, weighted positive radius %d"
          % (len(set(np.round(R, 9))), len(set(np.round(RW, 9))),
             len(set(np.round(PF, 9))), len(set(np.round(RP, 9)))))
    print("  survivors  A (r, r_w) %s   B (r, posfrac, r_w) %s   C (r, r_w+) %s"
          % (fmt(a, a_ok, a_marg), fmt(b, b_ok, b_marg), fmt(c, c_ok, c_marg)))
    print("  C picks gamma/(pi/8) %s" % np.round(pick[:m] / UNIT, 3))
    print("          beta /(pi/8) %s" % np.round(pick[m:] / UNIT, 3))

    np.savez(SHELLS / ("wshell_row%d_%s.npz" % (row, mode)), shell=shell, pick=pick,
             radii=R, floor=floor, weights=w, edges=np.array(E))
    return {"row": row, "mode": mode, "n": n, "m": m, "D": D, "maxcut": mc, "floor": floor,
            "r_min": r_min, "r2_units": (r_min / (PI / 4)) ** 2,
            "shell": len(shell), "exact": exact, "zero_modes": zm,
            "small_eigs": [float(v) for v in small_eigs],
            "A": a, "B": b, "C": c,
            "A_ok": bool(a_ok), "B_ok": b_ok, "C_ok": bool(c_ok)}

# Each worker process loads its own copy of the CSV/floors once, instead of
# the parent pickling a DataFrame into every one of the row x mode tasks.
def _init_worker():
    global _DF, _KNOWN
    _DF = pd.read_csv(CSV)
    _KNOWN = load_known()

def _run_task(row_mode):
    row, mode = row_mode
    return run(_DF, row, _KNOWN, mode=mode)

def main():
    rows = [int(a) for a in sys.argv[1:]] or ROWS
    SHELLS.mkdir(parents=True, exist_ok=True)
    tasks = [(row, mode) for mode in MODES for row in rows]
    nproc = min(len(tasks), mp.cpu_count())
    with mp.Pool(nproc, initializer=_init_worker) as pool:
        out = [r for r in pool.map(_run_task, tasks) if r is not None]
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n%4s %8s %4s %4s %8s %8s %10s %6s %6s %10s %10s %10s"
          % ("row", "mode", "n", "m", "MaxCut", "floor", "R^2 units",
             "shell", "exact", "A", "B", "C"))
    for r in out:
        print("%4d %8s %4d %4d %8d %8.3f %10.4f %6d %6s %10s %10s %10s"
              % (r["row"], r["mode"], r["n"], r["m"], r["maxcut"], r["floor"],
                 r["r2_units"], r["shell"], r["exact"],
                 "%d" % r["A"] if r["A_ok"] else "%d?" % r["A"],
                 "%d" % r["B"] if r["B_ok"] else "%d?" % r["B"],
                 "%d" % r["C"] if r["C_ok"] else "%d?" % r["C"]))

if __name__ == "__main__":
    main()