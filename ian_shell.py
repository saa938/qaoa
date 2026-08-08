"""
Where does the inner shell sit in his harvest, once the points are folded into
the fundamental domain?
"""

import glob
import json
import os
import numpy as np
import maqaoa_core as M

DATA = "AshayMAQAOAData"
OUT = "ian_shell.json"
TOL_D = 1e-2
ROW_OF = {"AshayGlobalMinimums_0": 10, "AshayGlobalMinimums_1": 11,
          "AshayGlobalMinimums_2": 12, "AshayGlobalMinimums_3": 13}

# Deduplication. But I don't think it did much most of the minima were already unique.

# Wrap every point into [-pi/2, pi/2) and deduplicate (because of pi periodicity). 
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
    return np.array(keep)

# Sign symmetry too, since E(x) = E(-x) means x and -x are the same minimum.
def sign_canonical(P):
    A, B = M.wrap_pi(P), M.wrap_pi(-P)
    out = np.empty_like(A)
    for i in range(len(A)):
        out[i] = A[i] if tuple(np.round(A[i], 6)) <= tuple(np.round(B[i], 6)) else B[i]
    return out

# The fraction of that radius carried by positive coordinates.
def radii_and_fracs(P):
    G = M.geodesic_vec(P, np.zeros(P.shape[1]))
    r = np.sqrt((G ** 2).sum(1))
    pos = np.sqrt((np.where(G > 0, G, 0.0) ** 2).sum(1))
    return r, np.where(r > 1e-12, pos / np.maximum(r, 1e-30), 0.0)

def main():
    rows = []
    print("%4s %4s %6s %8s %8s %10s %8s %8s %8s %8s"
          % ("row", "m", "n_raw", "n_fold", "n_sign", "r_min", "r_mean", "r_max",
             "n_tied", "max_frac"))
    for f in sorted(glob.glob(os.path.join(DATA, "AshayGlobalMinimums_*", "er_graph_minima.npz"))):
        folder = os.path.basename(os.path.dirname(f))
        d = np.load(f, allow_pickle=True)
        X = d["minima"]
        edges = [tuple(int(a) for a in e) for e in d["edges"]]
        n = 1 + max(max(e) for e in edges)
        energy, energy_batch, grad, D = M.make_energy(n, edges, p=1)
 
        E = energy_batch(X)
        Xw = dedupe(M.wrap_pi(X))
        Xs = dedupe(sign_canonical(X))
        r_w, f_w = radii_and_fracs(Xw)
 
        rmin = float(r_w.min())
        at_min = r_w < rmin + 1e-6
        r = {"folder": folder, "row": ROW_OF[folder], "n": n, "m": len(edges), "D": D,
             "n_raw": int(X.shape[0]),
             "n_after_pi_fold": int(Xw.shape[0]),
             "n_after_pi_fold_and_sign": int(Xs.shape[0]),
             "energy_min": round(float(E.min()), 6),
             "energy_max": round(float(E.max()), 9),
             "r_min": round(rmin, 6),
             "r_mean": round(float(r_w.mean()), 4),
             "r_max": round(float(r_w.max()), 4),
             "r_min_over_sqrtD": round(rmin / np.sqrt(D), 5),
             "n_tied_at_r_min": int(at_min.sum()),
             "max_frac_at_r_min": round(float(f_w[at_min].max()), 6),
             "global_max_frac": round(float(f_w.max()), 6)}
        rows.append(r)
        print("%4d %4d %6d %8d %8d %10.6f %8.4f %8.4f %8d %8.4f"
              % (r["row"], r["m"], r["n_raw"], r["n_after_pi_fold"],
                 r["n_after_pi_fold_and_sign"], r["r_min"], r["r_mean"], r["r_max"],
                 r["n_tied_at_r_min"], r["max_frac_at_r_min"]))
    json.dump(rows, open(OUT, "w"), indent=1)
 
if __name__ == "__main__":
    main()