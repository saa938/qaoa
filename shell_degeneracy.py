"""
At the smallest radius, is the minimum unique, or are there other true minima at
the same radius with the same largest positive fraction?
"""

import ast
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import maqaoa_core as M
import radius_search as R

CSV = "MaxCutMAQAOAData.csv"
OUT = "shell_degeneracy.json"
TOL_D = 1e-2
BAND = 1e-4 # radius window counted as "at the shell"

def radii(P):
    G = M.geodesic_vec(P, np.zeros(P.shape[1]))
    return np.sqrt((G ** 2).sum(1))

# Fraction of the radius carried by the positive coordinates.  Note the sign flip does
# not preserve this: if x has fraction f then -x has sqrt(1 - f^2), because the positive
# and negative parts swap and the two squared parts add to r^2.  So the larger of a
# mirror pair is always at least 1/sqrt(2), and this criterion can only ever halve the
# candidate set, never pick out one point.
def pos_fracs(P):
    G = M.geodesic_vec(P, np.zeros(P.shape[1]))
    r = np.sqrt((G ** 2).sum(1))
    pos = np.sqrt((np.where(G > 0, G, 0.0) ** 2).sum(1))
    return np.where(r > 1e-12, pos / np.maximum(r, 1e-30), 0.0)

def dedupe(P, tol=TOL_D):
    keep = []
    for x in P:
        if not keep:
            keep.append(x)
            continue
        K = np.array(keep)
        d = np.mod(K - x + np.pi / 2, np.pi) - np.pi / 2
        if np.sqrt((d ** 2).sum(1)).min() > tol:
            keep.append(x)
    return np.array(keep) if keep else np.zeros((0, P.shape[1]))

# Brute force inside a ball of radius Rcap, ignoring the positive fraction entirely.
def harvest_in_ball(energy, grad, D, floor, Rcap, restarts, seed, a=1.5):
    rng = np.random.default_rng(seed)
    shape, dshape = R.shape_family("exp", a)
    f, fg = R.make_shaped_energy(energy, grad, floor, Rcap, shape, dshape)
    pts = []
    for _ in range(restarts):
        rr = minimize(f, R.sample_start(rng, D, Rcap), jac=fg, method="L-BFGS-B",
                      options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})
        rp = M.polish(energy, grad, rr.x)
        if rp.fun <= floor + 1e-6:
            pts.append(M.wrap_pi(rp.x))
    return np.array(pts) if pts else np.zeros((0, D))

def main():
    df = pd.read_csv(CSV)
    shells = {d["row"]: d for d in json.load(open("shell_radius.json"))}
    rows = []
    print("%4s %4s %6s %10s %10s %10s %8s"
          % ("row", "m", "n_aut", "radius", "at_shell", "max_frac", "tied"))
    for row in sorted(shells):
        s = shells[row]
        edges = [tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])]
        n = int(df.loc[row, "Number of Nodes"])
        energy, energy_batch, grad, D = M.make_energy(n, edges, p=1)
        floor, Rstar = s["floor"], s["r_min_after_slide"]

        autos, images = M.symmetry_group(edges, p=1)

        # seed with the known shell point plus a fresh capped harvest
        got = harvest_in_ball(energy, grad, D, floor, Rstar + 1e-3,
                              restarts=120, seed=1000 + row)
        P = np.array([np.load("shellmin_row%d.npz" % row)["x"]] + list(got))

        # close under the automorphisms and the sign flip, both of which preserve the
        # radius, so without this the count is just however many images the optimizer
        # happened to stumble on
        closed = np.array([y for x in P for y in images(x)])
        closed = closed[np.abs(energy_batch(closed) - floor) < 1e-8]

        shell = dedupe(closed[np.abs(radii(closed) - Rstar) < BAND])
        r = {"row": row, "m": s["m"], "D": D, "floor": floor,
             "shell_radius": Rstar, "n_aut": len(autos),
             "n_distinct_at_shell": int(len(shell))}
        if len(shell):
            fr = pos_fracs(shell)
            fmax = float(fr.max())
            r["max_pos_fraction"] = round(fmax, 6)
            r["n_tied_at_max_fraction"] = int((fr > fmax - 1e-6).sum())
            r["distinct_fractions"] = sorted({round(float(v), 6) for v in fr})
            np.savez("shellset_row%d.npz" % row, shell=shell, fracs=fr,
                     radius=Rstar, floor=floor)
        rows.append(r)
        print("%4d %4d %6d %10.6f %10d %10.6f %8s"
              % (row, r["m"], r["n_aut"], Rstar, r["n_distinct_at_shell"],
                 r.get("max_pos_fraction", float("nan")),
                 r.get("n_tied_at_max_fraction", "-")))
    json.dump(rows, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()
