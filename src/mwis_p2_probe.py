"""
Usage:  python src/mwis_p2_probe.py

What the MWIS landscape looks like at p = 2, on one graph.  The full pipeline (run_problems.py with P = 2)
did not finish: after 143 restarts (about 45 minutes) it had 82 floor points, each reached once, and the
pick changed after every batch.  This script measures why, with a plain multistart:
  how many starts reach the floor (the exact MWIS, since p = 2 can do anything p = 1 can),
  how many flat directions the floor points have (Hessian eigenvalues below rf.EV_ZERO),
  the radius of the raw floor points, and of the first SETTLE of them after settle, slide and refine,
  whether the settled points are the same point (up to the symmetries),
  and how far refine got (largest part of x along the flat directions, 0 at the closest point).
The p = 1 pick with layer 2 set to 0 is also a floor point, and its radius is printed for comparison.
Results go to results/mwis_p2_probe.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

import run_filter as rf
import run_problems as rp

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "mwis_p2_probe.json"

ROW = 13
P = 2
SEED = 1000 * ROW + 10 * P + 1 # the seed run_problems.py uses for this graph at p = 2
N_STARTS = 100
SETTLE = 12 # floor points taken through settle, slide and refine (about 35 s each)

# Number of flat directions at x, and the largest part of x along them.
def flatness(grad, x):
    ev, V = eigh(rf.hessian(grad, x))
    N = V[:, ev < rf.EV_ZERO * max(1.0, ev[-1])]
    return N.shape[1], float(np.abs(N.T @ x).max()) if N.shape[1] else 0.0

def main():
    df = pd.read_csv(rf.CSV)
    prob = rp.instance("mwis", df, ROW)
    n, terms, c, const = prob["n"], prob["terms"], prob["c"], prob["const"]
    T, m = len(terms), len(prob["edges"])
    energy, energy_batch, grad, D, state = rp.make_energy(n, terms, c, const, P)
    per = rp.periods(terms, n, P, c)
    autos, images = rp.symmetry_group(n, terms, c, P)
    reduce = lambda x: rp.reduce_point(x, terms, n, c, P)
    chain = lambda x: rf.settle(energy, grad, rf.slide(energy, grad, rf.settle(energy, grad, x, reduce), reduce), reduce)
    best, opt = rp.exact(prob)

    p1 = [d for d in json.load(open(ROOT / "results" / "mwis_p1.json")) if d["row"] == ROW][0]
    x1 = np.array(p1["pick"])
    emb = np.concatenate([x1[:T], np.zeros(T), x1[T:], np.zeros(n)])
    out = {"row": ROW, "p": P, "D": D, "seed": SEED, "exact": best, "p1_embedded_energy": energy(emb),
           "p1_embedded_radius": rf.norm_origin(reduce(emb), per)}
    print("row %d  p %d  D %d  exact %.4f  p=1 pick with layer 2 = 0: energy %.6f radius %.5f"
          % (ROW, P, D, best, out["p1_embedded_energy"], out["p1_embedded_radius"]), flush=True)

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    ends = []
    for _ in range(N_STARTS):
        r = rf.polish(energy, grad, (rng.random(D) - 0.5) * per)
        ends.append((float(r.fun), rf.fold(r.x, per)))
    floor = [x for f, x in ends if f <= best + rf.TOL_E]
    flat = [flatness(grad, x)[0] for x in floor[:30]]
    rad = [rf.norm_origin(reduce(x), per) for x in floor]
    out.update({"starts": N_STARTS, "floor_hits": len(floor), "flat_raw": flat,
                "raw_radius": {"min": min(rad), "median": float(np.median(rad)), "max": max(rad)}})
    print("multistart: %d of %d starts reach the exact floor (%.0f s)" % (len(floor), N_STARTS, time.time() - t0))
    print("flat directions at floor points (first 30): min %d  median %.0f  max %d  of %d"
          % (min(flat), np.median(flat), max(flat), D))
    print("radius of raw floor points: min %.4f  median %.4f  max %.4f" % (min(rad), np.median(rad), max(rad)), flush=True)

    settled = []
    for k, x in enumerate(floor[:SETTLE]):
        t1 = time.time()
        y = rp.refine(energy, grad, chain(x), reduce, rp.REFINE_FINAL)
        k_flat, left = flatness(grad, y)
        settled.append({"radius": rf.norm_origin(y, per), "energy_minus_exact": energy(y) - best, "flat": k_flat,
                        "left_along_flat": left, "x": [float(a) for a in y]})
        print("  settled point %2d: energy - exact %.1e  radius %.5f  flat %2d  left along flat %.1e  (%.0f s)"
              % (k, energy(y) - best, settled[-1]["radius"], k_flat, left, time.time() - t1), flush=True)
    groups = []
    for s in settled:
        y = np.array(s["x"])
        if all(min(rf.geodesic_dist(z, g, per) for z in images(y)) > rf.TOL_D for g in groups):
            groups.append(y)
    out.update({"settled": settled, "distinct_settled": len(groups)})
    print("distinct settled points (up to the symmetries): %d of %d" % (len(groups), len(settled)))
    print("settled radii: %s" % ", ".join("%.5f" % r for r in sorted(s["radius"] for s in settled)))
    y = np.array(min(settled, key=lambda s: s["radius"])["x"])
    print("closest settled point: betas layer 1 %s" % np.round(y[P * T:P * T + n], 4))
    print("                       betas layer 2 %s" % np.round(y[P * T + n:], 4))
    print("                       largest |edge gamma| layer 1 %.4f, layer 2 %.4f" % (np.abs(y[:m]).max(), np.abs(y[T:T + m]).max()))
    json.dump(out, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    main()