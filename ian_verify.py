"""
Do his global minima land on my floor, and does anything of his beat it?
"""

import glob
import json
import os
import numpy as np
import maqaoa_core as M

DATA = "AshayMAQAOAData"
OUT = "ian_verify.json"
ROW_OF = {"AshayGlobalMinimums_0": 10, "AshayGlobalMinimums_1": 11,
          "AshayGlobalMinimums_2": 12, "AshayGlobalMinimums_3": 13}

# Read one folder.  His gammas are same order as mine as found in ian_convention.py.
def load(f):
    d = np.load(f, allow_pickle=True)
    return (d["minima"], float(d["best_energy"][0]),
            [tuple(int(a) for a in e) for e in d["edges"]])

def main():
    rows = []
    print("%4s %3s %4s %9s %13s %11s %11s %8s %8s"
          % ("row", "n", "m", "n_minima", "max_E_err", "ian_floor", "my_floor",
             "match", "below"))
    for f in sorted(glob.glob(os.path.join(DATA, "AshayGlobalMinimums_*", "er_graph_minima.npz"))):
        folder = os.path.basename(os.path.dirname(f))
        X, e_ian, edges = load(f)
        n = 1 + max(max(e) for e in edges)
        energy, energy_batch, grad, D = M.make_energy(n, edges, p=1)

        E = energy_batch(X)
        floor = M.find_floor(energy, grad, D, restarts=300, seed=0) # my own, independently

        r = {"folder": folder, "row": ROW_OF[folder], "n": n, "m": len(edges),
             "n_minima": int(X.shape[0]),
             "ian_best_energy": e_ian,
             "max_abs_err_vs_his_floor": float(np.max(np.abs(E - e_ian))),
             "my_floor_300_restarts": floor,
             "floor_matches_ian": bool(abs(floor - e_ian) < 1e-6),
             "n_of_his_below_my_floor": int((E < floor - 1e-6).sum())}
        rows.append(r)
        print("%4d %3d %4d %9d %13.3e %11.4f %11.4f %8s %8d"
              % (r["row"], n, r["m"], r["n_minima"], r["max_abs_err_vs_his_floor"],
                 e_ian, floor, r["floor_matches_ian"], r["n_of_his_below_my_floor"]))
    json.dump(rows, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()