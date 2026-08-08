"""
I found a smaller inner radius than anything in the harvest on
two of the four graphs.  Are my points really minima, or is that a bug?
Answer: they are actually minima
"""

import json
import os
import numpy as np
import maqaoa_core as M

DATA = "AshayMAQAOAData"
OUT = "ian_cross.json"
ROWS = [10, 11, 12, 13]
FOLDER = {10: "AshayGlobalMinimums_0", 11: "AshayGlobalMinimums_1",
          12: "AshayGlobalMinimums_2", 13: "AshayGlobalMinimums_3"}

# Hessian for slope
def hessian(grad, x, h=1e-5):
    D = len(x)
    H = np.empty((D, D))
    for i in range(D):
        e = np.zeros(D)
        e[i] = h
        H[i] = (grad(x + e) - grad(x - e)) / (2 * h)
    return 0.5 * (H + H.T)

# Calculate geodesic distance
def radii(P):
    G = M.geodesic_vec(P, np.zeros(P.shape[1]))
    return np.sqrt((G ** 2).sum(1))

def main():
    rows = []
    print("%4s %4s %11s %11s %11s %11s %9s %6s %7s" % ("row", "pts", "E_err", "grad", "r_mine", "r_ian", "min_eig", "flat", "his<=me"))
    for row in ROWS:
        d = np.load("shell_row%d.npz" % row, allow_pickle=True)
        shell = d["shell"]
        floor = float(d["floor"])
        edges = [tuple(int(a) for a in e) for e in d["edges"]]
        n = 1 + max(max(e) for e in edges)
        energy, energy_batch, grad, D = M.make_energy(n, edges, p=1)

        E = energy_batch(shell)
        gn = np.array([np.linalg.norm(grad(x)) for x in shell])
        r_mine = radii(shell)

        # certify the single lowest-radius point properly
        ev = np.linalg.eigvalsh(hessian(grad, shell[int(r_mine.argmin())]))

        di = np.load(os.path.join(DATA, FOLDER[row], "er_graph_minima.npz"), allow_pickle=True)
        r_ian = radii(M.wrap_pi(di["minima"]))

        r = {"row": row, "D": D, "floor": floor, "shell_pts": int(len(shell)),
             "shell_max_energy_err": float(np.max(np.abs(E - floor))),
             "shell_max_grad_norm": float(gn.max()),
             "r_min_mine": round(float(r_mine.min()), 6),
             "r_min_ian": round(float(r_ian.min()), 6),
             "min_hessian_eig_at_r_min": float(ev.min()),
             "n_flat_dirs_at_r_min": int((ev < 1e-6).sum()),
             "ian_pts_at_or_below_my_r_min": int((r_ian <= r_mine.min() + 1e-6).sum()),
             "r_min_mine_in_quarter_pi_sq": round(float((r_mine.min() / (np.pi / 4)) ** 2), 4),
             "r_min_ian_in_quarter_pi_sq": round(float((r_ian.min() / (np.pi / 4)) ** 2), 4)}
        rows.append(r)
        print("%4d %4d %11.2e %11.2e %11.6f %11.6f %9.2e %6d %7d"
              % (row, r["shell_pts"], r["shell_max_energy_err"], r["shell_max_grad_norm"],
                 r["r_min_mine"], r["r_min_ian"], r["min_hessian_eig_at_r_min"],
                 r["n_flat_dirs_at_r_min"], r["ian_pts_at_or_below_my_r_min"]))
    json.dump(rows, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()