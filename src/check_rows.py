"""
Usage:  python src/check_rows.py [csv] [max_n]
Checks a csv written by make_dataset.py merge, or MaxCutMAQAOAData.csv, without using any run_filter code.
For every row it checks that the gamma edge order is nx.Graph(Edges).edges() and rebuilds the energy from the
gammas, betas and weights with its own statevector (phases for the ZZ terms, scipy expm for each X rotation).
The rebuilt energy has to equal the Energy column.  Rows with more than max_n nodes are skipped.
With no arguments it checks data/MaxCutMAQAOAData.csv, which has to match to about 1e-15.
"""

import ast
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "MaxCutMAQAOAData.csv"
MAX_N = 14
X = np.array([[0, 1], [1, 0]], dtype=complex)

# Energy of one layer of MA-QAOA: |+>^n, exp(i gamma_e w_e Z_u Z_v) on every edge, exp(i beta_j X) on every node.
def energy(n, gedges, w, gam, bet):
    k = np.arange(1 << n)
    z = 1 - 2 * ((k[:, None] >> np.arange(n)[None, :]) & 1)
    psi = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=complex)
    for (u, v), wk, g in zip(gedges, w, gam):
        psi = psi * np.exp(1j * g * wk * z[:, u] * z[:, v])
    for j in range(n):
        U = expm(1j * bet[j] * X)
        t = psi.reshape(1 << (n - 1 - j), 2, 1 << j)
        psi = np.einsum("ab,xby->xay", U, t).reshape(-1)
    prob = np.abs(psi) ** 2
    return float(sum(0.5 * wk * (prob @ (z[:, u] * z[:, v]) - 1) for (u, v), wk in zip(gedges, w)))

def main():
    path = Path(sys.argv[1]) if len(sys.argv) >= 2 else CSV
    max_n = int(sys.argv[2]) if len(sys.argv) >= 3 else MAX_N
    df = pd.read_csv(path)
    ian = "Gammas" not in df.columns
    worst, order_ok, done = 0.0, 0, 0
    for r in range(len(df)):
        n = int(df.loc[r, "Number of Nodes"])
        edges = [tuple(t) for t in ast.literal_eval(df.loc[r, "Edges"])]
        nxo = [tuple(sorted(t)) for t in nx.Graph(edges).edges()]
        if ian:
            gedges, w = nxo, [1.0] * len(edges)
            gam = ast.literal_eval(df.loc[r, "Predicted Gammas"])
            bet = ast.literal_eval(df.loc[r, "Predicted Betas"])
        else:
            gedges = [tuple(t) for t in ast.literal_eval(df.loc[r, "Gamma Edges"])]
            w = ast.literal_eval(df.loc[r, "Weights"])
            gam = ast.literal_eval(df.loc[r, "Gammas"])
            bet = ast.literal_eval(df.loc[r, "Betas"])
        order_ok += gedges == nxo
        if n > max_n:
            continue
        worst = max(worst, abs(energy(n, gedges, w, np.ravel(gam), np.ravel(bet)) - float(df.loc[r, "Energy"])))
        done += 1
    print("%s: %d rows, gamma order == nx.Graph(Edges).edges() on %d, energy rebuilt on %d, max |E - Energy| = %.2e"
          % (path.name, len(df), order_ok, done, worst))

if __name__ == "__main__":
    main()