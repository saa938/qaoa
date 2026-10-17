"""
Usage:  python src/multistart_compare.py [problem | summary]
With no arguments it runs every problem in PROBLEMS, then prints a summary per problem.
"summary" only prints the summary from results/multistart_compare.json.

Compares the filter pipeline (run_problems.run, which is run_filter.py's pipeline for Z terms) with a plain
multistart on the same graphs, the same instances and the same starting points.  Restart i of the filter
and start i of the multistart begin at the same point, because both draw (rng.random(D) - 0.5) * period
from default_rng(seed) in the same order.  The filter then minimizes energy + lambda r^2 (lambda cycling
through rf.LAMS) and polishes, settles flat valleys, folds with the symmetries and filters down to one
point.  The multistart only minimizes the energy from each start and keeps every end point.

For every graph and seed, for both searches:
  floor hits   starts that end at the lowest energy either search found (within rf.TOL_E)
  cost         gradient calls (descents only, as in rf.harvest) in total and per floor hit
For the multistart, its floor points are counted three ways:
  raw          more than rf.TOL_D apart on the torus, no symmetries used
  folded       distinct after folding with the symmetries, so copies of the same point count once
  settled      distinct after settle, slide and refine as well, so points in one flat valley count once
and whether any of them is the filter's pick.
Geometry of every settled floor point and of the pick: radius, number of flat directions (Hessian
eigenvalues below rf.EV_ZERO), whether every angle sits on the pi/8 grid in phase units (2 c gamma and
beta), and the probability of measuring an optimal bitstring.

Picks come from results/<problem>_p1.json (run_problems.py).  For Max-Cut they come from
results/ordered_filter_v2.json (run_filter.py, weight draw 0), which only has the first seed, so the
second Max-Cut seed is compared with that same pick.
Results go to results/multistart_compare.json after every case.
"""

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

import run_filter as rf
import run_problems as rp

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "multistart_compare.json"

P = 1
PROBLEMS = ["spinglass_nofield", "maxcut"]
ROWS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
SEED_SHIFTS = [0, 500] # the same two seeds as run_problems.py
N_STARTS = 500 # the filter stopped at 500 restarts on every one of these cases
GRID = np.pi / 8
GRID_TOL = 1e-4 # an angle counts as on the grid when its phase is this close to a multiple of pi/8

# The instance for one graph.  Max-Cut uses weight draw 0, the same weights as ordered_filter_v2.json.
def build(problem, df, row):
    if problem != "maxcut":
        return rp.instance(problem, df, row)
    edges = rf.canonical_edges([tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])])
    n = int(df.loc[row, "Number of Nodes"])
    prob = rp.maxcut(edges, n, rf.draw_weights(len(edges), 100 * row))
    prob["row"] = row
    return prob

# The filter's record for this graph and seed, or None if it has not been run yet.
def filter_record(problem, row, seed):
    if problem == "maxcut":
        res = json.load(open(ROOT / "results" / "ordered_filter_v2.json"))
        hit = [d for d in res if d["row"] == row and d["p"] == P and d["weighted"] and d.get("trial") == 0]
    else:
        path = ROOT / "results" / ("%s_p%d.json" % (problem, P))
        res = json.load(open(path)) if path.exists() else []
        hit = [d for d in res if d["row"] == row and seed in d.get("seeds", [d["seed"]])]
    return hit[0] if hit else None

# Add x to pts unless one of its images is already there.  Returns True when x is new.
def add_new(pts, x, images, per):
    imgs = images(x)
    for y in pts:
        if min(rf.geodesic_dist(z, y, per) for z in imgs) <= rf.TOL_D:
            return False
    pts.append(x)
    return True

# Radius, flat directions, pi/8 grid and probability of an optimal bitstring at one floor point.
def geometry(x, prob, per, grad, state, opt):
    c = np.asarray(prob["c"], float)
    T = len(prob["terms"])
    ev = eigh(rf.hessian(grad, x), eigvals_only=True)
    phase = np.concatenate([2 * c * x[:T], x[T:]])
    on = np.abs(phase - GRID * np.round(phase / GRID)).max() < GRID_TOL
    psi = state(x)
    return {"radius": rf.norm_origin(x, per), "flat": int((ev < rf.EV_ZERO * max(1.0, ev[-1])).sum()),
            "min_eig": float(ev[0]), "grid": bool(on), "p_opt": float(sum(abs(psi[i]) ** 2 for i in opt))}

def compare(problem, prob, seed, rec):
    t0 = time.time()
    n, terms, c, const = prob["n"], prob["terms"], prob["c"], prob["const"]
    energy, energy_batch, grad, D, state = rp.make_energy(n, terms, c, const, P)
    per = rp.periods(terms, n, P, c)
    autos, images = rp.symmetry_group(n, terms, c, P)
    reduce = lambda x: rp.reduce_point(x, terms, n, c, P)
    chain = lambda x: rf.settle(energy, grad, rf.slide(energy, grad, rf.settle(energy, grad, x, reduce), reduce), reduce)
    settle_fn = lambda x: rp.refine(energy, grad, chain(x), reduce, rp.REFINE_TOL)
    best, opt = rp.exact(prob)

    rng = np.random.default_rng(seed)
    ms, fs = [], []
    for i in range(N_STARTS):
        x0 = (rng.random(D) - 0.5) * per
        r = rf.polish(energy, grad, x0)
        ms.append((float(r.fun), rf.fold(r.x, per), int(r.nfev)))
        lam = rf.LAMS[i % len(rf.LAMS)]
        if lam == 0:
            fs.append(ms[-1])
        else:
            f, x, nf = rf.descend(energy, grad, per, x0, lam)
            fs.append((f, rf.fold(x, per), nf))

    floor = min(min(f for f, _, _ in ms), min(f for f, _, _ in fs), rec["floor"])
    out = {"problem": problem, "row": prob["row"], "seed": seed, "n": n, "T": len(terms), "D": D,
           "best": best, "n_opt": len(opt), "floor": floor, "filter_floor": rec["floor"], "n_auto": len(autos)}
    for name, runs in (("multistart", ms), ("filter_search", fs)):
        hits = sum(f <= floor + rf.TOL_E for f, _, _ in runs)
        nf = sum(k for _, _, k in runs)
        out[name] = {"hits": int(hits), "starts": len(runs), "nfev": int(nf),
                     "nfev_per_hit": float(nf / hits) if hits else None,
                     "first_hit": next((i + 1 for i, (f, _, _) in enumerate(runs) if f <= floor + rf.TOL_E), None)}
    levels = sorted({round(f, 6) for f, _, _ in ms})
    out["multistart"]["levels"] = len(levels)

    raw = []
    for f, x, _ in ms:
        if f <= floor + rf.TOL_E:
            rf.add_distinct(raw, x, per)
    folded = []
    for x in raw:
        add_new(folded, reduce(x), images, per)
    settled = []
    for x in folded:
        y = settle_fn(x)
        if energy(y) <= floor + rf.TOL_E:
            add_new(settled, y, images, per)
    settled = [rp.refine(energy, grad, y, reduce, rp.REFINE_FINAL) for y in settled]

    pick = np.array(rec["pick"])
    same = lambda x: min(rf.geodesic_dist(z, pick, per) for z in images(x)) <= rf.TOL_D
    out["distinct"] = {"raw": len(raw), "folded": len(folded), "settled": len(settled)}
    out["found_pick"] = {"raw": bool(any(same(reduce(x)) for x in raw)), "settled": bool(any(same(y) for y in settled))}
    out["pick"] = geometry(pick, prob, per, grad, state, opt)
    out["settled_points"] = [dict(geometry(y, prob, per, grad, state, opt), is_pick=bool(same(y))) for y in settled]
    rad = [rf.norm_origin(reduce(x), per) for x in raw]
    flat = [geometry(reduce(x), prob, per, grad, state, opt)["flat"] for x in raw[:40]]
    out["raw_radius"] = {"min": float(min(rad)), "median": float(np.median(rad)), "max": float(max(rad))} if rad else None
    out["raw_flat"] = {"checked": len(flat), "with_flat_dirs": int(sum(k > 0 for k in flat)), "max_flat": int(max(flat)) if flat else 0}
    out["filter"] = {k: rec.get(k) for k in ("pool", "shell", "filter", "used", "hits", "restarts", "nfev", "r_min")}
    out["secs"] = round(time.time() - t0, 1)
    return out

# One summary per problem over every graph and seed in res.
def summary(res):
    for problem in PROBLEMS:
        R = [d for d in res if d["problem"] == problem]
        if not R:
            continue
        med = lambda v: float(np.median(v))
        ms = [d["multistart"]["hits"] / d["multistart"]["starts"] for d in R]
        fs = [d["filter_search"]["hits"] / d["filter_search"]["starts"] for d in R]
        cm = [d["multistart"]["nfev_per_hit"] for d in R if d["multistart"]["nfev_per_hit"]]
        cf = [d["filter_search"]["nfev_per_hit"] for d in R if d["filter_search"]["nfev_per_hit"]]
        dist = {k: [d["distinct"][k] for d in R] for k in ("raw", "folded", "settled")}
        ratio = sorted({(d["row"], round(d["floor"] / d["best"], 4)) for d in R})
        flat_raw = sum(d["raw_flat"]["with_flat_dirs"] for d in R) / max(1, sum(d["raw_flat"]["checked"] for d in R))
        print("\n%s: %d graphs, %d cases (graph x seed)" % (problem, len({d["row"] for d in R}), len(R)))
        print("   floor / exact optimum by graph: %s" % ", ".join("%d: %.3f" % rv for rv in ratio))
        print("   floor reached by multistart in %d of %d cases, by the filter search in %d" % (
            sum(d["multistart"]["hits"] > 0 for d in R), len(R), sum(d["filter_search"]["hits"] > 0 for d in R)))
        print("   starts reaching the floor, median (range): multistart %.3f (%.3f-%.3f), filter search %.3f (%.3f-%.3f)" % (
            med(ms), min(ms), max(ms), med(fs), min(fs), max(fs)))
        print("   gradient calls per floor hit, median: multistart %.0f, filter search %.0f" % (med(cm), med(cf)))
        print("   distinct floor points from multistart, median (max): raw %.0f (%d), folded %.0f (%d), settled %.0f (%d)" % (
            med(dist["raw"]), max(dist["raw"]), med(dist["folded"]), max(dist["folded"]), med(dist["settled"]), max(dist["settled"])))
        print("   multistart contains the pick: as raw end points %d of %d, after settling %d of %d" % (
            sum(d["found_pick"]["raw"] for d in R), len(R), sum(d["found_pick"]["settled"] for d in R), len(R)))
        picks = {d["row"]: d["pick"] for d in R}
        print("   picks: %d of %d isolated (no flat direction), %d of %d on the pi/8 grid, P(optimum) %s" % (
            sum(p["flat"] == 0 for p in picks.values()), len(picks), sum(p["grid"] for p in picks.values()), len(picks),
            ", ".join("%.2f" % picks[r]["p_opt"] for r in sorted(picks))))
        print("   multistart floor end points with a flat direction: %.0f%%" % (100 * flat_raw))
        print("   pick radius vs median radius of the raw floor points: %s" % ", ".join(
            "%d: %.3f/%.3f" % (d["row"], d["pick"]["radius"], d["raw_radius"]["median"]) for d in R if d["seed"] % 1000 < 500))

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "summary":
        summary(json.load(open(OUT)))
        return
    todo = [sys.argv[1]] if len(sys.argv) >= 2 else PROBLEMS
    df = pd.read_csv(rf.CSV)
    res = json.load(open(OUT)) if OUT.exists() else []
    for problem in todo:
        for row in ROWS:
            prob = build(problem, df, row)
            for shift in SEED_SHIFTS:
                seed = 1000 * row + 10 * P + 1 + shift
                if any(d["problem"] == problem and d["row"] == row and d["seed"] == seed for d in res):
                    continue
                rec = filter_record(problem, row, seed if problem != "maxcut" else 1000 * row + 10 * P + 1)
                if rec is None:
                    print("%s row %d seed %d: no filter result yet, skipping" % (problem, row, seed), flush=True)
                    continue
                d = compare(problem, prob, seed, rec)
                res.append(d)
                json.dump(res, open(OUT, "w"), indent=1) # checkpoint after every case
                ms, fs = d["multistart"], d["filter_search"]
                print("%-17s row %2d seed %5d  floor %9.5f  hits multistart %3d/%d  filter search %3d/%d  "
                      "distinct raw %3d folded %3d settled %2d  pick found raw %-5s settled %-5s  "
                      "pick r %.4f flat %d grid %s  raw r median %.4f  %4.0f s"
                      % (problem, row, seed, d["floor"], ms["hits"], ms["starts"], fs["hits"], fs["starts"],
                         d["distinct"]["raw"], d["distinct"]["folded"], d["distinct"]["settled"],
                         d["found_pick"]["raw"], d["found_pick"]["settled"], d["pick"]["radius"], d["pick"]["flat"],
                         d["pick"]["grid"], d["raw_radius"]["median"] if d["raw_radius"] else float("nan"), d["secs"]),
                      flush=True)
    summary(res)

if __name__ == "__main__":
    main()