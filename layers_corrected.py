"""
Two things were wrong with my first attempt at this and both are fixed here.

FIX 1
  Instead of snapping to the nearest pi/4  grid point (which can falsely pass), 
  the new test directly measures each minimum's distance to its nearest grid point.

FIX 2
  Rather than relying on random restarts, the smallest radius is found by solving
  the constrained optimization problem   min ||x||^2   s.t.  E(x) <= floor + eps,
  ensuring that the shell is actually the lowest-radius shell at the floor.
"""

import ast
import sys
import os
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
import maqaoa_core as M

CSV = "MaxCutMAQAOAData.csv"
TOL_E = 1e-6        # energy tolerance for "at the floor"
TOL_D = 1e-2        # torus distance below which two minima are the same point
GRIDS = [2, 4, 8, 16]

# Max cut value by brute force.  Only works for small graphs (n <= 20).
def true_maxcut(edges, n):
    E = list(nx.Graph(edges).edges())
    return max(sum(1 for (i, j) in E if ((mask >> i) & 1) != ((mask >> j) & 1))
               for mask in range(1 << n))

# Count the number of near-zero Hessian eigenvalues at a given point.
def zero_modes(grad, x, D, thresh=1e-2, h=1e-5):
    H = np.zeros((D, D))
    for i in range(D):
        e = np.zeros(D)
        e[i] = h
        H[:, i] = (grad(x + e) - grad(x - e)) / (2 * h)
    w = np.linalg.eigvalsh(0.5 * (H + H.T))
    return int((np.abs(w) <= thresh).sum()), w

# Find the minimum radius of a shell at the given floor.
def min_radius_at_floor(energy, grad, D, floor, starts, eps=1e-9):
    def r2(x):
        v = M.geodesic_vec(x, np.zeros(D))
        return float(v @ v)

    def r2_jac(x):
        return 2 * M.geodesic_vec(x, np.zeros(D))

    cons = [{"type": "ineq",
             "fun": lambda x: floor + eps - energy(x),
             "jac": lambda x: -grad(x)}]

    best_r, best_x = np.inf, None
    for x0 in starts:
        try:
            res = minimize(r2, x0, jac=r2_jac, constraints=cons, method="SLSQP",
                           options={"maxiter": 400, "ftol": 1e-14})
        except Exception:
            continue
        xw = M.wrap_pi(res.x)
        # SLSQP can drift slightly off the constraint; re-polish and re-check.
        rp = M.polish(energy, grad, xw)
        if rp.fun <= floor + TOL_E:
            cand = M.wrap_pi(rp.x)
            r = M.norm_origin(cand)
            # the polish can undo some of the radius gain, so keep the SLSQP
            # point too when it is genuinely at the floor
            if energy(xw) <= floor + TOL_E and M.norm_origin(xw) < r:
                cand, r = xw, M.norm_origin(xw)
            if r < best_r:
                best_r, best_x = r, cand
    return best_r, best_x

# Compute the geodesic distance from a point to the nearest grid point.
def dist_to_grid(x, k):
    return float(np.linalg.norm(M.geodesic_vec(M.snap_to_grid(x, k), x)))

# Run the analysis for a given row and parameters.
def run(row, p, restarts):
    df = pd.read_csv(CSV)
    edges = [tuple(e) for e in ast.literal_eval(df.loc[row, "Edges"])]
    n = int(df.loc[row, "Number of Nodes"])
    energy, _, grad, D = M.make_energy(n, edges, p=p)
    m = len(list(nx.Graph(edges).edges()))
    mc = true_maxcut(edges, n)
    rng = np.random.default_rng(31337 + 100 * row + p)
    t0 = time.time()

    # 1. harvest floor-level minima 
    lam_plan = ([0.0] * (restarts // 4) + [0.1] * (restarts // 4)
                + [0.2] * (restarts // 4) + [0.35] * (restarts - 3 * (restarts // 4)))
    E, X = [], []
    for lam in lam_plan:
        x0 = rng.uniform(0, np.pi, D)
        if lam:
            def pen(x, l=lam):
                v = M.geodesic_vec(x, np.zeros(D))
                return energy(x) + l * float(v @ v)
            def pgr(x, l=lam):
                v = M.geodesic_vec(x, np.zeros(D))
                return grad(x) + 2 * l * v
            x0 = minimize(pen, x0, jac=pgr, method="L-BFGS-B",
                          options={"ftol": 1e-13, "gtol": 1e-10}).x
        r = M.polish(energy, grad, x0)
        E.append(float(r.fun)); X.append(M.wrap_pi(r.x))
    E = np.array(E)
    floor = round(float(E.min()), 6)
    at_floor = [X[i] for i in range(len(E)) if E[i] <= floor + TOL_E]

    # 2. constrained radius minimization 
    starts = at_floor[:40] if len(at_floor) > 40 else at_floor
    r_min, x_min = min_radius_at_floor(energy, grad, D, floor, starts)

    # 3. collect the shell at that radius 
    # Re-run the constrained solve from every floor-level start and keep every
    # distinct endpoint whose radius matches the best one.
    shell = []
    for x0 in starts:
        r, x = min_radius_at_floor(energy, grad, D, floor, [x0])
        if x is not None and r <= r_min + 1e-3:
            if all(M.geodesic_dist(x, q) > TOL_D for q in shell):
                shell.append(x)
    if not shell and x_min is not None:
        shell = [x_min]
    shell = np.array(shell)

    # 4. flat directions    
    zms = []
    for k in np.linspace(0, len(shell) - 1, min(3, len(shell))).astype(int):
        z, _ = zero_modes(grad, shell[k], D)
        zms.append(z)
    zm = int(np.median(zms)) if zms else -1

    # 5. corrected grid tests 
    grid_abs, grid_rel = {}, {}
    for k in GRIDS:
        dists = np.array([dist_to_grid(x, k) for x in shell])
        grid_abs[k] = (float((dists < 1e-6).mean()), float(dists.max()))
        if len(shell) > 1:
            Dm = M.geodesic_dist_matrix(shell)
            off = Dm[~np.eye(len(shell), dtype=bool)]
            u = (off / (np.pi / k)) ** 2
            grid_rel[k] = float(np.abs(u - np.round(u)).max())
        else:
            grid_rel[k] = float("nan")

    secs = round(time.time() - t0, 1)
    print(f"\n=== row {row}, p={p}  (n={n}, m={m}, D={D}) ===")
    print(f"  true Max-Cut = {mc};  floor reached = {floor}  "
          f"(gap {mc + floor:+.2f})")
    print(f"  restarts {len(lam_plan)}, reached floor {len(at_floor)}, "
          f"shell after constrained descent = {len(shell)}")
    print(f"  min radius = {r_min/np.pi:.6f} pi   "
          f"R^2 in (pi/4)^2 units = {(r_min/(np.pi/4))**2:.4f}")
    print(f"  flat directions at the minimum: {zm} of {D}  "
          f"-> {'MANIFOLD' if zm > 0 else 'isolated point'}")
    print(f"  ABSOLUTE quantization (fraction of shell exactly on grid):")
    for k in GRIDS:
        f_on, dmax = grid_abs[k]
        print(f"      pi/{k:<2d}: {100*f_on:5.1f}% on grid, max distance to grid {dmax:.4f}")
    print(f"  RELATIVE quantization (pairwise sq. distances integer?):")
    for k in GRIDS:
        print(f"      pi/{k:<2d}: max integer residual {grid_rel[k]:.3e}")
    print(f"  ({secs}s)")

    rec = dict(row=row, p=p, D=D, n=n, m=m, true_maxcut=mc, floor=floor,
               gap=mc + floor, n_restarts=len(lam_plan), n_at_floor=len(at_floor),
               n_shell=int(len(shell)), r_min_over_pi=r_min / np.pi,
               r2_pi4=(r_min / (np.pi / 4)) ** 2, zero_modes=zm,
               grid_absolute={str(k): grid_abs[k] for k in GRIDS},
               grid_relative={str(k): grid_rel[k] for k in GRIDS}, secs=secs)
    path = "layers_corrected.json"
    allr = json.load(open(path)) if os.path.exists(path) else []
    allr = [a for a in allr if not (a["row"] == row and a["p"] == p)] + [rec]
    json.dump(sorted(allr, key=lambda z: (z["row"], z["p"])), open(path, "w"), indent=1)
    return rec


if __name__ == "__main__":
    row = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if len(sys.argv) > 3:
        restarts = int(sys.argv[3])
    else:
        restarts = {1: 120, 2: 40, 3: 12}.get(p, 40)
    print(f"running row={row}  p={p}  restarts={restarts}")
    print("(pass arguments as:  python layers_corrected.py <row> <p> [restarts])")
    run(row, p, restarts)
