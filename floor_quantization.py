"""
I noticed the floor was always a half-integer.  Is that true?  
And if so, is it because the floor points are quantized to the pi/4 grid?  This script checks that.
"""
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from analytic_fast import P1

RESTARTS = 40


def frac_half(v):
    return np.abs(v * 2 - np.round(v * 2)) / 2


df = pd.read_csv("MaxCutMAQAOAData.csv")
rec = []
edgevals = []

for row in range(10, 20):
    edges = ast.literal_eval(df.loc[row, "Edges"])
    n = int(df.loc[row, "Number of Nodes"])
    P = P1(n, edges)
    D, sh = P.D, (np.pi / 4) * np.eye(P.D)

    def f(x):
        return P.energy(x)

    def g(x):
        x = np.asarray(x, float)
        return np.array([P.energy(x + sh[i]) - P.energy(x - sh[i]) for i in range(D)])

    floor = float(np.load("shell_row%d.npz" % row)["floor"])
    rng = np.random.default_rng(4000 + row)
    for _ in range(RESTARTS):
        xr, _, _ = P.rotosolve(rng.uniform(0, np.pi, D), sweeps=300)
        r = minimize(f, xr, jac=g, method="L-BFGS-B",
                     options={"ftol": 1e-18, "gtol": 1e-14, "maxiter": 20000})
        v = float(r.fun)
        rec.append({"row": row, "value": v, "is_floor": int(v <= floor + 1e-7),
                    "dist_half": float(frac_half(v))})

    # per-edge <ZZ> at the certified floor points
    for x in np.load("shell_row%d.npz" % row)["shell"][:40]:
        A, B, C = P.coeffs(x[:P.m])
        b = 2.0 * x[P.m:]
        c, s = np.cos(b), np.sin(b)
        ev = A * c[P.U] * s[P.V] + B * s[P.U] * c[P.V] + C * s[P.U] * s[P.V]
        edgevals.append(ev)
    print("row %d done" % row, flush=True)

d = pd.DataFrame(rec)
d.to_csv("floor_quantization.csv", index=False)
ev = np.concatenate(edgevals)

fl = d.loc[d.is_floor == 1, "dist_half"].values
nf = d.loc[d.is_floor == 0, "dist_half"].values
print("floor points      %4d   max dist to n/2 = %.2e" % (len(fl), fl.max()))
print("higher local min  %4d   max dist to n/2 = %.2e" % (len(nf), nf.max()))
print("higher local min, fraction that are half-integers: %.3f"
      % float((nf < 1e-9).mean()))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# Panel 1: distance to nearest half-integer, floor vs non-floor
bins = np.logspace(-16, 0, 33)
ax[0].hist(np.clip(fl, 1e-16, None), bins=bins, alpha=.75,
           label="global floor (n=%d)" % len(fl), color="#1f77b4")
ax[0].hist(np.clip(nf, 1e-16, None), bins=bins, alpha=.75,
           label="higher local min (n=%d)" % len(nf), color="#d62728")
ax[0].set_xscale("log")
ax[0].set_xlabel("distance to nearest multiple of 1/2")
ax[0].set_ylabel("count")
ax[0].set_title("Floors are half-integers; local minima need not be")
ax[0].legend(fontsize=8)

# Panel 2: per-edge <ZZ> at floor points
ax[1].hist(ev, bins=200, color="#2ca02c")
ax[1].set_xlabel(r"$\langle Z_u Z_v \rangle$ per edge at floor points")
ax[1].set_ylabel("count")
ax[1].set_title("Edge values pile up on 0 and $\\pm 1$")
for t in (-1, 0, 1):
    ax[1].axvline(t, ls=":", lw=1, color="k")

# Panel 3: gap vs MaxCut/m, showing no clean rule
gap_rows = {10: 1.0, 11: 0.0, 12: 1.5, 13: 1.5, 14: 0.0, 15: 0.0,
            16: 0.0, 17: 1.0, 18: 0.0, 19: 0.5}
xs, ys, lb = [], [], []
for row in range(10, 20):
    edges = list(pd.unique(pd.Series([tuple(sorted(e)) for e in
                 ast.literal_eval(df.loc[row, "Edges"])])))
    m = len(edges)
    fldr = float(np.load("shell_row%d.npz" % row)["floor"])
    mc = -(fldr - gap_rows[row])
    xs.append(mc / m)
    ys.append(gap_rows[row])
    lb.append(str(row))
ax[2].scatter(xs, ys, s=55, color="#9467bd")
for x, y, t in zip(xs, ys, lb):
    ax[2].annotate(t, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax[2].set_xlabel("MaxCut / m")
ax[2].set_ylabel("gap = floor + MaxCut")
ax[2].set_title("Gap is not explained by cut fraction")

plt.tight_layout()
plt.savefig("floor_quantization.png", dpi=150)
print("wrote floor_quantization.csv and floor_quantization.png")
