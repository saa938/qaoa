"""
Final quantization test for the QAOA minima.
"""

import ast
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import maqaoa_core as M

CSV = "MaxCutMAQAOAData.csv"
ON_GRID = 1e-6      # distance below which an angle counts as on the grid
COSET_TOL = 1e-4    # componentwise tolerance for the coset test
ZERO_EV = 1e-2      # |eigenvalue| below this counts as a flat direction

# Extract the true max cut value by brute force.  Only works for small graphs (n <= 20).
def true_maxcut(edges, n):
    E = list(nx.Graph(edges).edges())
    return max(sum(1 for (i, j) in E if ((mask >> i) & 1) != ((mask >> j) & 1))
               for mask in range(1 << n))

# Count the number of near-zero Hessian eigenvalues at a given point.
def zero_modes(grad, x, D, h=1e-5):
    H = np.zeros((D, D))
    for i in range(D):
        e = np.zeros(D)
        e[i] = h
        H[:, i] = (grad(x + e) - grad(x - e)) / (2 * h)
    w = np.linalg.eigvalsh(0.5 * (H + H.T))
    return int((np.abs(w) <= ZERO_EV).sum())


def main():
    paths = sorted(glob.glob("shell_row*.npz"),
                   key=lambda s: int(s.split("row")[1].split(".")[0]))
    if not paths:
        print("ERROR: no shell_row*.npz files found in", os.getcwd())
        print()
        print("These hold the harvested minima and every test here reads them.")
        print("Either copy them into this folder, or regenerate them with:")
        print("    python lowest_norm_survey.py")
        print("(that run is slow - it re-optimizes all ten graphs)")
        sys.exit(1)

    if not os.path.exists(CSV):
        print(f"ERROR: {CSV} not found in {os.getcwd()}")
        sys.exit(1)

    df = pd.read_csv(CSV)
    rows = []

    for path in paths:
        row = int(path.split("row")[1].split(".")[0])
        d = np.load(path)
        shell = d["shell"]
        edges = [tuple(e) for e in d["edges"]]
        floor = float(d["floor"])
        n = int(df.loc[row, "Number of Nodes"])
        _, _, grad, D = M.make_energy(n, edges, p=1)

        mc = true_maxcut(edges, n)

        # absolute quantization test: is every point on the pi/4 grid?
        dists = np.array([float(np.linalg.norm(
            M.geodesic_vec(M.snap_to_grid(x, 4), x))) for x in shell])
        frac_on = float((dists < ON_GRID).mean())
        absolute = bool(dists.max() < ON_GRID)

        # Shifted lattice test: are all points in a single coset of the pi/4 grid?
        if len(shell) > 1:
            base = shell[0]
            dev = 0.0
            for x in shell[1:]:
                u = M.geodesic_vec(x, base) / (np.pi / 4)
                dev = max(dev, float(np.abs(u - np.round(u)).max()))
        else:
            dev = float("nan")
        coset = (not absolute) and (dev < COSET_TOL)

        # flat directions: how many near-zero Hessian eigenvalues?
        idx = np.linspace(0, len(shell) - 1, min(3, len(shell))).astype(int)
        zm = int(np.median([zero_modes(grad, shell[k], D) for k in idx]))

        verdict = ("QUANTIZED" if absolute
                   else ("shifted lattice" if coset else "no"))
        rows.append(dict(row=row, n_shell=int(len(shell)), floor=floor,
                         true_maxcut=mc, gap=mc + floor,
                         frac_on_grid=frac_on, max_dist_grid=float(dists.max()),
                         coset_dev=dev, zero_modes=zm, verdict=verdict))

    print("=" * 100)
    print("FINAL QUANTIZATION TEST")
    print("=" * 100)
    print(f"{'row':>3} {'shell':>6} {'floor':>8} {'trueMC':>7} {'gap':>5} "
          f"{'dist to grid':>13} {'coset dev':>11} {'flat':>5} {'verdict':>17}")
    for r in rows:
        print(f"{r['row']:>3} {r['n_shell']:>6} {r['floor']:>8.2f} "
              f"{r['true_maxcut']:>7d} {r['gap']:>5.1f} "
              f"{r['max_dist_grid']:>13.2e} {r['coset_dev']:>11.2e} "
              f"{r['zero_modes']:>5} {r['verdict']:>17}")

    q = [r["row"] for r in rows if r["verdict"] == "QUANTIZED"]
    c = [r["row"] for r in rows if r["verdict"] == "shifted lattice"]
    no = [r["row"] for r in rows if r["verdict"] == "no"]
    print()
    print(f"pi/4 quantized        : {q}")
    print(f"shifted lattice only  : {c}")
    print(f"no quantization       : {no}")

    iso = [r["row"] for r in rows if r["zero_modes"] == 0]
    print()
    print(f"minima are isolated points (0 flat directions): {iso}")
    print(f"quantized graphs                              : {q}")
    print("these two sets are identical"
          if iso == q else "WARNING: these two sets differ")

    json.dump(rows, open("quantization_final.json", "w"), indent=1)
    print("\nwrote quantization_final.json")


if __name__ == "__main__":
    main()