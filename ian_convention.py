"""
Orders are different between Ian's and mine as found out in ian_check.py. Are the gammas also different? 
Answer: No they are actually same.
"""

import glob
import json
import os
import numpy as np
import networkx as nx
import maqaoa_core as M

DATA = "AshayMAQAOAData"
OUT = "ian_convention.json"
ROW_OF = {"AshayGlobalMinimums_0": 10, "AshayGlobalMinimums_1": 11,
          "AshayGlobalMinimums_2": 12, "AshayGlobalMinimums_3": 13}

# Vector is either same, his is sorted and permute into mine, or reverse permutation of this
def orderings(X, edges_ian):
    m = len(edges_ian)
    canon = [frozenset(e) for e in nx.Graph(edges_ian).edges()]
    where = {frozenset(e): i for i, e in enumerate(edges_ian)}
    perm = np.array([where[e] for e in canon]) # canon slot c <- ian slot perm[c]
    inv = np.argsort(perm)
    gam, bet = X[:, :m], X[:, m:] # betas are node indexed, never reordered
    return {"identity": X,
            "ian_to_canon": np.concatenate([gam[:, perm], bet], 1),
            "canon_to_ian": np.concatenate([gam[:, inv], bet], 1)}

def main():
    rows = []
    print("%4s %14s %16s %16s" % ("row", "identity", "ian_to_canon", "canon_to_ian"))
    for f in sorted(glob.glob(os.path.join(DATA, "AshayGlobalMinimums_*", "er_graph_minima.npz"))):
        folder = os.path.basename(os.path.dirname(f))
        d = np.load(f, allow_pickle=True)
        X = d["minima"]
        e_ian = float(d["best_energy"][0])
        edges = [tuple(int(a) for a in e) for e in d["edges"]]
        n = 1 + max(max(e) for e in edges)
        _, energy_batch, _, _ = M.make_energy(n, edges, p=1)

        r = {"folder": folder, "row": ROW_OF[folder]}
        for name, Y in orderings(X, edges).items():
            r[name + "_max_err"] = float(np.max(np.abs(energy_batch(Y) - e_ian)))
        rows.append(r)
        print("%4d %14.3e %16.3e %16.3e"
              % (r["row"], r["identity_max_err"],
                 r["ian_to_canon_max_err"], r["canon_to_ian_max_err"]))
    json.dump(rows, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()