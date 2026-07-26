"""
Check 1: Feed the predictions and verify energies
Check 2: Rebuild the circuit with explicit Kronecker products and dense matrix-vector products.
Check 3: Compare parameter-shift gradients against central finite-difference gradients.
Check 4: Shift each coordinate by pi and confirm the energy is unchanged.
"""

import ast
import numpy as np
import pandas as pd
import maqaoa_core as M

CSV = "MaxCutMAQAOAData.csv"

def check_csv_energies(): # Check 1
    print("=" * 72)
    print("CHECK 1: reproduce the CSV energies from the NN predictions")
    print("=" * 72)
    df = pd.read_csv(CSV)
    worst = 0.0
    for i, row in df.iterrows():
        # Read graph structure and predicted parameters from the CSV row
        edges = ast.literal_eval(row["Edges"])
        betas = np.array(ast.literal_eval(row["Predicted Betas"]), float)
        gammas = np.array(ast.literal_eval(row["Predicted Gammas"]), float)
        n = int(row["Number of Nodes"])
        energy, _, _, D = M.make_energy(n, edges, p=1)
        # packing order is gammas first, then betas
        x = np.concatenate([gammas, betas])
        # Compare the energy computed by the engine against the CSV value
        got = energy(x)
        err = abs(got - float(row["Energy"]))
        worst = max(worst, err)
        if i in (0, 10, 19):
            print(f"  row {i:2d} ({row['Type']:>11s}): "
                  f"csv={row['Energy']:.15f}  sim={got:.15f}  |diff|={err:.2e}")
    print(f"  worst absolute error over all {len(df)} rows: {worst:.3e}")
    return worst


def check_engine_and_grad(): # Checks 2, 3, 4
    print()
    print("=" * 72)
    print("CHECK 2/3/4: engine cross-check, gradient, pi-periodicity")
    print("=" * 72)
    df = pd.read_csv(CSV)
    edges = ast.literal_eval(df.loc[10, "Edges"])   # the ER graph already studied
    n = 8
    for p in (1, 2, 3):
        # Optimized simulator vs explicit Kronecker product simulator
        kron_err = M.verify_against_kron(n, edges, p=p, trials=2)
        # Build the energy and gradient functions 
        energy, _, grad, D = M.make_energy(n, edges, p=p)
        # Check the parameter-shift gradient against central finite-difference gradient
        g_err = M.check_gradient(energy, grad, D, trials=2)
        # Check that shifting each coordinate by pi leaves the energy unchanged
        per_err = M.check_pi_periodicity(energy, D, trials=3)
        print(f"  p={p}  D={D:3d}   kron-vs-fast={kron_err:.2e}   "
              f"grad-vs-FD={g_err:.2e}   pi-periodicity={per_err:.2e}")


if __name__ == "__main__":
    check_csv_energies()
    check_engine_and_grad()
    print("\nAll checks complete.")
