"""
Quick check to make sure we are on same page
"""

import ast
import glob
import json
import os
import numpy as np
import pandas as pd
import networkx as nx

CSV = "MaxCutMAQAOAData.csv"
DATA = "AshayMAQAOAData"
OUT = "ian_check.json"

def load_ian():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "AshayGlobalMinimums_*", "er_graph_minima.npz"))):
        d = np.load(f, allow_pickle=True)
        out.append({"folder": os.path.basename(os.path.dirname(f)),
                    "minima": d["minima"],
                    "best_energy": float(d["best_energy"][0]),
                    "edges": [tuple(int(a) for a in e) for e in d["edges"]]})
    return out

def match_row(df, edges):
    target = {frozenset(e) for e in edges}
    for i in range(len(df)):
        e = [tuple(t) for t in ast.literal_eval(df.loc[i, "Edges"])]
        if {frozenset(t) for t in e} == target: return i
    return None

def main():
    df = pd.read_csv(CSV)
    rows = []
    print("%22s %4s %3s %4s %8s %6s %6s %12s %9s"
          % ("folder", "row", "n", "m", "n_minima", "D_exp", "D_act", "best_energy", "order_ok"))
    for g in load_ian():
        edges = g["edges"]
        n = 1 + max(max(e) for e in edges)
        row = match_row(df, edges)
        canon = [tuple(e) for e in nx.Graph(edges).edges()]
        order_ok = canon == [tuple(e) for e in edges]
        rows.append({"folder": g["folder"], "row": row, "n": n, "m": len(edges),
                     "n_minima": int(g["minima"].shape[0]),
                     "D_expected": len(edges) + n,
                     "D_actual": int(g["minima"].shape[1]),
                     "ian_best_energy": g["best_energy"],
                     "edge_order_matches_canonical": bool(order_ok)})
        r = rows[-1]
        print("%22s %4d %3d %4d %8d %6d %6d %12.6f %9s"
              % (r["folder"], r["row"], r["n"], r["m"], r["n_minima"],
                 r["D_expected"], r["D_actual"], r["ian_best_energy"],
                 r["edge_order_matches_canonical"]))
    json.dump(rows, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()