"""
Usage:  python src/pick_stability.py [problem]
With no arguments it runs every problem in PROBLEMS, then prints a summary.

Is the filter's pick a function of the graph alone, and does it change smoothly with the weights?
Both matter if the pick is going to be the target a GNN learns.
  base      the pick for the graph as given (run_problems.run, seed 1000 row + 11), taken from
            results/<problem>_p1.json when that run exists
  relabel   renumber the nodes at random (this also reorders the edges) and rerun with the same seed.
            Mapped back to the original numbering, the pick should be the same point.
  perturb   multiply every weight by (1 + eps u), u uniform on [-1, 1], for each eps in EPS, and rerun.
            The pick is compared through its phases (2 c gamma and beta), because a point whose phases
            stay put still moves in gamma when the weights change.  A small phase distance means the
            pick stayed or moved a little; a large one means the choice jumped to a different minimum.
Distances are period-aware and allow for the symmetries (the images of the base pick), in the base
graph's gamma units.
Results go to results/pick_stability_<problem>.json after every run.
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import run_filter as rf
import run_problems as rp

ROOT = Path(__file__).resolve().parent.parent

P = 1
PROBLEMS = ["spinglass_nofield", "maxcut"]
ROWS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
EPS = [0.02, 0.10]
SAME = 1e-3 # phase distance at or below this is the same point (rf.TOL_D)
JUMP = 0.2 # phase distance at or above this is a different minimum

# The instance with these edges (in any order) and one weight per edge.  Spin glass weights are the
# couplings J, Max-Cut weights are w.
def make(problem, edges, n, wts):
    pairs = sorted((tuple(sorted(e)), float(a)) for e, a in zip(edges, wts))
    e2 = [e for e, _ in pairs]
    a2 = np.array([a for _, a in pairs])
    return rp.maxcut(e2, n, a2) if problem == "maxcut" else rp.spinglass(e2, n, a2)

# The graph's edges, node count and weights (Max-Cut uses weight draw 0, as in ordered_filter_v2.json).
def original(problem, df, row):
    if problem == "maxcut":
        edges = rf.canonical_edges([tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])])
        return edges, int(df.loc[row, "Number of Nodes"]), rf.draw_weights(len(edges), 100 * row)
    prob = rp.instance(problem, df, row)
    return prob["edges"], prob["n"], prob["J"]

# Phase distance from x (a pick of the instance prob2) to the base pick, both written in the base
# instance's gamma units.  With map_back, x comes from the relabelled instance and is first renumbered.
def distance(base, prob2, x, perm=None):
    m, n = len(base["edges"]), base["n"]
    x = np.asarray(x, float)
    if perm is not None:
        idx = {e: i for i, e in enumerate(prob2["edges"])}
        y = np.empty(m + n)
        for e, (u, v) in enumerate(base["edges"]):
            y[e] = x[idx[tuple(sorted((perm[u], perm[v])))]]
        y[m:] = x[m + np.asarray(perm)]
        x = y
    else:
        x = x.copy()
        x[:m] = x[:m] * np.asarray(prob2["c"]) / np.asarray(base["c"])
    per = rp.periods(base["terms"], n, P, base["c"])
    images = rp.symmetry_group(n, base["terms"], base["c"], P)[1]
    pick = np.array(base["pick"])
    return float(min(rf.geodesic_dist(z, pick, per) for z in images(x)))

def run_one(problem, prob, row, variant, extra):
    seed = 1000 * row + 10 * P + 1
    prob["row"] = row
    rec = rp.run(prob, P, seed)
    rec.update({"variant": variant, "row": row})
    rec.update(extra)
    return rec

def main():
    todo = [sys.argv[1]] if len(sys.argv) >= 2 else PROBLEMS
    df = pd.read_csv(rf.CSV)
    for problem in todo:
        out = ROOT / "results" / ("pick_stability_%s.json" % problem)
        res = json.load(open(out)) if out.exists() else []
        have = {(d["row"], d["variant"]) for d in res}
        for row in ROWS:
            edges, n, wts = original(problem, df, row)
            base_prob = make(problem, edges, n, wts)
            if (row, "base") not in have:
                saved = ROOT / "results" / ("%s_p%d.json" % (problem, P))
                old = [d for d in json.load(open(saved)) if d["row"] == row and d["seed"] == 1000 * row + 10 * P + 1] if saved.exists() else []
                res.append(dict(old[0], variant="base") if old else run_one(problem, base_prob, row, "base", {}))
                json.dump(res, open(out, "w"), indent=1)
            base = dict(base_prob, pick=[d for d in res if d["row"] == row and d["variant"] == "base"][0]["pick"])

            perm = [int(a) for a in np.random.default_rng(1000 + row).permutation(n)]
            if (row, "relabel") not in have:
                p2 = make(problem, [(perm[u], perm[v]) for (u, v) in edges], n, wts)
                rec = run_one(problem, p2, row, "relabel", {"perm": perm})
                rec["phase_distance"] = distance(base, p2, rec["pick"], perm)
                res.append(rec)
                json.dump(res, open(out, "w"), indent=1)

            for eps in EPS:
                name = "perturb %.2f" % eps
                if (row, name) in have:
                    continue
                u = np.random.default_rng(2000 + row).uniform(-1, 1, len(edges))
                w2 = np.asarray(wts) * (1 + eps * u)
                p2 = make(problem, edges, n, w2)
                rec = run_one(problem, p2, row, name, {"eps": eps, "weights": [float(a) for a in w2]})
                rec["phase_distance"] = distance(base, p2, rec["pick"])
                rec["gamma_distance"] = float(np.abs(np.array(rec["pick"]) - np.array(base["pick"])).max())
                res.append(rec)
                json.dump(res, open(out, "w"), indent=1)
            for d in res:
                if d["row"] == row and d["variant"] != "base":
                    print("%-17s row %2d  %-13s phase distance to base pick %.2e   floor %.5f  filter %s"
                          % (problem, row, d["variant"], d["phase_distance"], d["floor"], d["filter"]), flush=True)
    summary()

# Classify the stability of a pick with respect to a base instance.
def classify(base, prob2, rec):
    n = base["n"]
    m = len(base["edges"])
    energy, _, grad, D, _ = rp.make_energy(n, prob2["terms"], prob2["c"], prob2["const"], P)
    x = np.array(base["pick"], float)
    if rec["variant"].startswith("perturb"):
        x[:m] = x[:m] * np.asarray(base["c"]) / np.asarray(prob2["c"])
    else:
        y = np.empty(m + n)
        perm = rec["perm"]
        idx = {e: i for i, e in enumerate(prob2["edges"])}
        for e, (u, v) in enumerate(base["edges"]):
            y[idx[tuple(sorted((perm[u], perm[v])))]] = x[e]
        y[m + np.asarray(perm)] = x[m:]
        x = y
    e_old = float(rf.polish(energy, grad, x).fun)
    if e_old < rec["floor"] - rf.TOL_E:
        return "missed", e_old
    if e_old > rec["floor"] + rf.TOL_E:
        return "level", e_old
    return "tie", e_old

# How flat each edge term <Z_u Z_v> is at x, in phase units: the largest gradient entry of any edge term.
# If every edge term is flat, x stays a critical point for any edge weights.
def edge_flatness(prob, x, h=1e-6):
    n, terms, c = prob["n"], prob["terms"], np.asarray(prob["c"], float)
    T = len(terms)
    phase = np.concatenate([2 * c * np.asarray(x[:T]), np.asarray(x[T:])])
    unit = np.ones(T)
    state = rp.make_energy(n, terms, unit / 2, 0.0, P)[4]
    zs = rp.z_table(n, terms)
    f = lambda ph: zs @ np.abs(state(np.concatenate([ph[:T], ph[T:]]))) ** 2
    worst = 0.0
    for k in range(len(phase)):
        d = np.zeros(len(phase))
        d[k] = h
        worst = max(worst, np.abs((f(phase + d) - f(phase - d)) / (2 * h)).max())
    return float(worst)

# One line per problem and variant.
def summary():
    for problem in PROBLEMS:
        path = ROOT / "results" / ("pick_stability_%s.json" % problem)
        if not path.exists():
            continue
        res = json.load(open(path))
        df = pd.read_csv(rf.CSV)
        print("\n%s" % problem)
        for variant in ["relabel"] + ["perturb %.2f" % e for e in EPS]:
            D = sorted((d["row"], d["phase_distance"]) for d in res if d["variant"] == variant)
            if not D:
                continue
            same = sum(v <= SAME for _, v in D)
            jump = sum(v >= JUMP for _, v in D)
            print("   %-13s same %d, moved a little %d, jumped %d  (of %d)   %s" % (
                variant, same, len(D) - same - jump, jump, len(D), ", ".join("%d: %.1e" % rv for rv in D)))
        for d in sorted(res, key=lambda d: (d["row"], d["variant"])):
            if d["variant"] == "base" or d["phase_distance"] <= SAME:
                continue
            edges, n, wts = original(problem, df, d["row"])
            base_prob = make(problem, edges, n, wts)
            base = dict(base_prob, pick=[b for b in res if b["row"] == d["row"] and b["variant"] == "base"][0]["pick"])
            if d["variant"] == "relabel":
                p2 = make(problem, [(d["perm"][u], d["perm"][v]) for (u, v) in edges], n, wts)
            else:
                p2 = make(problem, edges, n, d["weights"])
            kind, e_old = classify(base, p2, d)
            print("   row %d %-13s changed by %.2e: %-6s (run floor %.6f, old pick polished at the new weights %.6f)"
                  % (d["row"], d["variant"], d["phase_distance"], kind, d["floor"], e_old))
        flat = []
        for b in sorted((b for b in res if b["variant"] == "base"), key=lambda b: b["row"]):
            edges, n, wts = original(problem, df, b["row"])
            flat.append((b["row"], edge_flatness(make(problem, edges, n, wts), b["pick"])))
        print("   largest edge-term gradient at each base pick (0 = every edge term flat): %s"
              % ", ".join("%d: %.0e" % rv for rv in flat))

if __name__ == "__main__":
    main()