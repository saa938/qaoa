"""
How small can the radius get while staying exactly on the floor?
"""

import ast
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
import maqaoa_core as M
import radius_search as R

CSV = "MaxCutMAQAOAData.csv"
OUT = "shell_radius.json"
TOL_E = 1e-7
N_SLIDE = 25 # sliding is an SLSQP solve each, so only spend it on the best candidates

# Squared geodesic radius and its gradient.  wrap folds each coordinate into
# [-pi/2, pi/2), and away from the fold boundary d(r^2)/dx is just 2v.
def r2_and_grad(x):
    v = M.geodesic_vec(np.asarray(x, float), np.zeros(len(x)))
    return float(v @ v), 2.0 * v

def radius_of(x):
    return float(np.sqrt(r2_and_grad(x)[0]))

# Slide along the floor from x0 to reduce the radius.  E >= floor everywhere, so the
# equality constraint just means "stay on the floor".  On an isolated minimum there is
# no feasible direction and this correctly does nothing, which is why sliding alone is
# not enough and the harvest below has to come first.
def slide(energy, grad, floor, x0, tol_e=TOL_E):
    con = NonlinearConstraint(lambda x: energy(x) - floor, 0.0, 0.0, jac=lambda x: grad(x))
    r = minimize(lambda x: r2_and_grad(x)[0], x0, jac=lambda x: r2_and_grad(x)[1],
                 constraints=[con], method="SLSQP",
                 options={"ftol": 1e-14, "maxiter": 500})
    x = M.polish(energy, grad, r.x).x # SLSQP can drift, put it back on the true floor
    if abs(energy(x) - floor) > tol_e:
        return x0
    return x if radius_of(x) < radius_of(x0) else x0

# Harvest floor points with exp(a*r) shaping on the floor-zeroed energy.  The shaping
# multiplies something that vanishes at the floor, so every true minimum stays exactly
# where it was and this only acts as a sampler biased inward.
def harvest(energy, grad, D, floor, restarts, seed, a=0.5):
    rng = np.random.default_rng(seed)
    shape, dshape = R.shape_family("exp", a)
    f, fg = R.make_shaped_energy(energy, grad, floor, R.full_radius(D), shape, dshape)
    pts = []
    for _ in range(restarts):
        rr = minimize(f, rng.uniform(0.0, np.pi, D), jac=fg, method="L-BFGS-B",
                      options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})
        rp = M.polish(energy, grad, rr.x)
        if rp.fun <= floor + 1e-6:
            pts.append(M.wrap_pi(rp.x))
    return pts

def load_row(df, row):
    return (int(df.loc[row, "Number of Nodes"]),
            [tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])])

def main():
    df = pd.read_csv(CSV)
    floors = {d["row"]: d["floor"] for d in json.load(open("floor_cheap.json"))}
    rows = []
    print("%4s %4s %4s %8s %6s %10s %10s %9s %10s %6s"
          % ("row", "m", "D", "floor", "n_got", "r_raw", "r_slid", "gain",
             "r^2/(pi/4)^2", "nnz"))
    for row in range(10, 20):
        n, edges = load_row(df, row)
        energy, energy_batch, grad, D = M.make_energy(n, edges, p=1)
        floor = floors[row]

        pts = harvest(energy, grad, D, floor, restarts=150, seed=row)
        if not pts:
            print("row %d: no floor points harvested" % row)
            continue

        r_raw = np.array([radius_of(x) for x in pts])
        order = np.argsort(r_raw)[:N_SLIDE]
        refined = [slide(energy, grad, floor, pts[i]) for i in order]
        r_ref = np.array([radius_of(x) for x in refined])

        x = refined[int(r_ref.argmin())]
        v = M.geodesic_vec(x, np.zeros(D))
        rmin = float(r_ref.min())

        r = {"row": row, "n": n, "m": len(edges), "D": D, "floor": floor,
             "n_harvested": len(pts),
             "r_min_before_slide": round(float(r_raw.min()), 6),
             "r_min_after_slide": round(rmin, 6),
             "slide_gain": round(float(r_raw.min() - rmin), 6),
             "in_quarter_pi_sq": round((rmin / (np.pi / 4)) ** 2, 4),
             "r_over_sqrt_m": round(rmin / np.sqrt(len(edges)), 5),
             "r_over_sqrt_D": round(rmin / np.sqrt(D), 5),
             "energy_at_solution": round(float(energy(x)), 12),
             "pos_fraction": round(float(np.linalg.norm(v[v > 0]) / rmin), 6),
             "n_nonzero_coords": int((np.abs(v) > 1e-6).sum())}
        rows.append(r)
        print("%4d %4d %4d %8.2f %6d %10.6f %10.6f %9.5f %10.4f %6d"
              % (row, r["m"], D, floor, r["n_harvested"], r["r_min_before_slide"],
                 r["r_min_after_slide"], r["slide_gain"], r["in_quarter_pi_sq"],
                 r["n_nonzero_coords"]))
        np.savez("shellmin_row%d.npz" % row, x=x, refined=np.array(refined),
                 radii=r_ref, floor=floor, edges=np.array(edges))
    json.dump(rows, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()
