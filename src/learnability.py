"""
Usage:  python src/learnability.py [check | build | build_rev | learn] [problem]
With no arguments it runs build, then learn, for every problem in PROBLEMS.  With a mode but no problem, that
mode runs for every problem in PROBLEMS.  build_rev is build going through the draws from the last one down,
writing to its own file, so a second process can share a build.

Can a GNN learn the filter's pick, so that QAOA does not have to be tuned?  Before training one, this tests how
well each edge's and node's angle in the pick can be predicted from the graph around that edge or node.

check   The dataset is built with STALL restarts instead of run_filter's 400 (a seed stops once its pick has not
        changed for STALL restarts), backed up by the two-seed guard.  check reruns the saved weighted p = 1
        cases in CHECK_ROWS that way (same seeds) and compares them with their saved picks, which used 400.
build   For every problem in PROBLEMS, every graph shape in SHAPES and DRAWS[problem] random weight draws, the
        guarded pipeline (run_problems.run) at p = 1.  Max-Cut weights are uniform on [0.5, 2]; spin glass
        couplings have random signs and the same sizes.  The draws go round the shapes, so a run stopped early
        is still balanced.  Results go to results/learnability_<problem>.json (build_rev: _rev.json) after every
        instance.
learn   Labels come from confirmed picks only, in phase units on the pi/8 grid: round(2|c_e| gamma_e / (pi/8))
        for edge e and round(beta_j / (pi/8)) for node j.  The pick is folded (reduce_point), so edge labels
        are in {-1, 0, 1, 2} and node labels in {-2, ..., 2}.  Two targets:
          size    |label|, which no symmetry changes
          signed  the label itself
        Features of each edge or node (feature_sets), in growing rings around it:
          own     the edge's weight and sign; the node's degree and weighted degree
          1-hop   + the edges and nodes it touches
          2-hop   + the same summaries for those nodes (max and min over the two ends of an edge, mean and max
                  over the neighbours of a node)
          oracle  + whether the edge is cut (satisfied) in the exact classical answer, which a GNN does not
                  have; it shows how much of the label is the classical answer itself
        Model: scikit-learn HistGradientBoostingClassifier, one per target and kind (edge or node).  Checked
        with a random forest, and with 25% and 50% of the training data (does more data still help?).
        Baselines: the most common label, and the most common label at that position in the same shape.
        Splits: FOLDS-fold cross validation over instances (new weights on shapes seen in training), and
        leave one shape out (a shape never seen in training).
        Energy test, out of fold, for every instance:
          predicted   energy at the angles built from the predicted signed labels, no tuning
          polished    after at most POLISH_ITERS L-BFGS-B iterations from there
          size+signs  predicted sizes, signs chosen by evaluating every sign pattern (energies only, no
                      gradients); an edge of size 2 sits on the edge of its window and is always +2
          random      plain L-BFGS-B from RANDOM_STARTS random points, for comparison
        A summary goes to results/learnability_summary_<problem>.json.
"""

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

import run_filter as rf
import run_problems as rp

ROOT = Path(__file__).resolve().parent.parent

P = 1
PROBLEMS = ["maxcut", "spinglass_nofield"]
SHAPES = [0, 13, 16] # rows of MaxCutMAQAOAData.csv: a 3-regular graph and two Erdos-Renyi graphs
DRAWS = {"maxcut": 35, "spinglass_nofield": 23} # weight draws per shape in the saved results; raise and rerun build to add more
STALL = 100 # restarts a seed's pick has to stay the same for before the seed stops (run_filter uses 400)
SEED0 = 10 ** 7 # instance seeds are SEED0 + 100000 shape + 1000 draw, apart from every other experiment
CHECK_ROWS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
FEATURES = ["own", "1-hop", "2-hop", "oracle"]
FOLDS = 5
POLISH_ITERS = 50 # most L-BFGS-B iterations allowed from the predicted angles
RANDOM_STARTS = 20
SIGN_CAP = 16 # most coordinates whose signs are enumerated (2^16 energy evaluations)
GRID = np.pi / 8
GRID_TOL = 1e-3 # a pick is on the grid when every phase is this close to a multiple of pi/8

# The instance for one shape and weight draw, and its seed.
def instance(problem, df, row, draw):
    edges = rf.canonical_edges([tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])])
    n = int(df.loc[row, "Number of Nodes"])
    m = len(edges)
    seed = SEED0 + 100000 * row + 1000 * draw
    w = rf.draw_weights(m, seed)
    if problem == "maxcut":
        prob = rp.maxcut(edges, n, w)
    else:
        prob = rp.spinglass(edges, n, np.random.default_rng(seed + 1).choice([-1, 1], size=m) * w)
    prob.update({"row": row, "type": df.loc[row, "Type"]})
    return prob, seed

# The saved weighted p = 1 case for a CSV row: the instance, its seed and its pick.
def saved_case(problem, df, row):
    if problem == "maxcut":
        res = json.load(open(ROOT / "results" / "ordered_filter_v2.json"))
        d = [d for d in res if d["row"] == row and d["p"] == P and d["weighted"] and d.get("trial") == 0][0]
        edges = rf.canonical_edges([tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])])
        prob = rp.maxcut(edges, int(df.loc[row, "Number of Nodes"]), np.array(d["w"]))
    else:
        res = json.load(open(ROOT / "results" / ("%s_p%d.json" % (problem, P))))
        d = rp.label(res, row)[0]
        prob = rp.instance(problem, df, row)
    prob["row"] = row
    return prob, d["seed"], d

def check(problem):
    rf.STALL = {P: STALL}
    df = pd.read_csv(rf.CSV)
    for row in CHECK_ROWS:
        prob, seed, old = saved_case(problem, df, row)
        rec = rp.run(prob, P, seed)
        per = rp.periods(prob["terms"], prob["n"], P, prob["c"])
        images = rp.symmetry_group(prob["n"], prob["terms"], prob["c"], P)[1]
        print("CHECK %-17s row %2d  same floor %s  same pick %s  confirmed %s  seeds %d  restarts %d (saved run %d)"
              % (problem, row, abs(rec["floor"] - old["floor"]) <= 2 * rf.TOL_E,
                 rp.same_pick(old["pick"], rec["pick"], images, per), rec["confirmed"], len(rec["seeds"]),
                 rec["restarts"], old["restarts"]), flush=True)

# The files build writes for a problem: the main one, and the one a second process going through the draws
# from the last one down writes (build_rev).
def build_files(problem):
    return [ROOT / "results" / ("learnability_%s%s.json" % (problem, s)) for s in ("", "_rev")]

# Run every (shape, draw) that neither file has yet.  With reverse = True the draws go from the last one down
# and the results go to the second file, so two processes can share the work.
def build(problem, reverse=False):
    rf.STALL = {P: STALL}
    df = pd.read_csv(rf.CSV)
    files = build_files(problem)
    out = files[1] if reverse else files[0]
    res = rp.load(out)
    draws = range(DRAWS[problem] - 1, -1, -1) if reverse else range(DRAWS[problem])
    for draw in draws:
        for row in SHAPES:
            if (row, draw) in {(d["shape"], d["draw"]) for f in files for d in rp.load(f)}:
                continue
            prob, seed = instance(problem, df, row, draw)
            rec = rp.run(prob, P, seed)
            rec.update({"shape": row, "draw": draw, "coupling": [float(a) for a in prob["c"]]})
            res.append(rec)
            json.dump(res, open(out, "w"), indent=1) # checkpoint after every instance

# Edge and node labels of a pick, in units of pi/8, and the largest distance of a phase from the grid.
def labels(prob, pick):
    T = len(prob["terms"])
    x = np.asarray(pick, float)
    ph = np.abs(2 * np.asarray(prob["c"], float)) * x[:T] / GRID
    b = x[T:] / GRID
    off = max(np.abs(ph - np.round(ph)).max(), np.abs(b - np.round(b)).max())
    return np.round(ph).astype(int), np.round(b).astype(int), float(off)

# The point whose phases are the given labels (the inverse of labels).
def point(prob, el, nl):
    return np.concatenate([np.asarray(el) * GRID / np.abs(2 * np.asarray(prob["c"], float)), np.asarray(nl) * GRID])

# Features of every edge and node for one instance, as {set name: (edge matrix, node matrix)}.  Edge weights
# are W = |2 c| (w for Max-Cut) and signs S = sign(c); S = +1 wants the two spins opposite.
def feature_sets(prob, opt):
    n, edges = prob["n"], prob["edges"]
    m = len(edges)
    c = np.asarray(prob["c"], float)
    W, S = np.abs(2 * c), np.sign(c)
    nb = [[] for _ in range(n)]
    for e, (u, v) in enumerate(edges):
        nb[u].append((v, e))
        nb[v].append((u, e))
    adj = [{k for k, _ in nb[j]} for j in range(n)]
    eidx = {frozenset(ed): e for e, ed in enumerate(edges)}
    deg = np.array([len(nb[j]) for j in range(n)], float)
    wdeg = np.array([sum(W[e] for _, e in nb[j]) for j in range(n)])
    sdeg = np.array([sum(S[e] * W[e] for _, e in nb[j]) for j in range(n)])
    etri, eftri = np.zeros(m), np.zeros(m)
    for e, (u, v) in enumerate(edges):
        for k in adj[u] & adj[v]:
            etri[e] += 1
            eftri[e] += S[e] * S[eidx[frozenset((u, k))]] * S[eidx[frozenset((v, k))]] > 0
    tri = np.array([0.5 * sum(etri[e] for _, e in nb[j]) for j in range(n)])
    ftri = np.array([0.5 * sum(eftri[e] for _, e in nb[j]) for j in range(n)])

    node1 = []
    for j in range(n):
        ws = [W[e] for _, e in nb[j]]
        ks = [k for k, _ in nb[j]]
        node1.append([deg[j], wdeg[j], sdeg[j], sum(S[e] < 0 for _, e in nb[j]), max(ws), min(ws), tri[j], ftri[j],
                      np.mean(deg[ks]), max(deg[ks]), np.mean(wdeg[ks]), max(wdeg[ks]),
                      wdeg[j] - max(wdeg[ks]), deg[j] - max(deg[ks])])
    node1 = np.array(node1)
    near = [adj[j] | {k2 for k in adj[j] for k2 in adj[k]} for j in range(n)]
    node2 = np.hstack([node1, np.array([np.concatenate([node1[list(adj[j])].mean(0), node1[list(adj[j])].max(0)])
                                        for j in range(n)]),
                       np.array([[np.mean([wdeg[k] < wdeg[j] for k in near[j] - {j}])] for j in range(n)])])

    edge1, edge2 = [], []
    for e, (u, v) in enumerate(edges):
        ou = [W[f] for _, f in nb[u] if f != e]
        ov = [W[f] for _, f in nb[v] if f != e]
        ru = np.mean([w_ < W[e] for w_ in ou]) if ou else 0.5
        rv = np.mean([w_ < W[e] for w_ in ov]) if ov else 0.5
        oth = ou + ov
        row1 = [W[e], S[e], deg[u] + deg[v], abs(deg[u] - deg[v]), wdeg[u] + wdeg[v], abs(wdeg[u] - wdeg[v]),
                sdeg[u] + sdeg[v], W[e] / wdeg[u] + W[e] / wdeg[v], ru + rv, max(ru, rv), etri[e], eftri[e],
                max(oth) if oth else 0.0, np.mean(oth) if oth else 0.0]
        edge1.append(row1)
        edge2.append(row1 + list(np.maximum(node1[u], node1[v])) + list(np.minimum(node1[u], node1[v])))
    edge1, edge2 = np.array(edge1), np.array(edge2)

    Z = 1 - 2 * ((np.array(opt)[:, None] >> np.arange(n)[None, :]) & 1)
    sat = np.array([[c[e] * z[u] * z[v] < 0 for e, (u, v) in enumerate(edges)] for z in Z], float)
    hub = int(np.argmax(wdeg))
    z0 = Z[0] * Z[0][hub]
    edge_o = np.hstack([edge2, sat[0][:, None], sat.mean(0)[:, None]])
    node_o = np.hstack([node2, np.array([[np.mean([sat[0][e] for _, e in nb[j]]), z0[j]] for j in range(n)])])
    return {"own": (edge1[:, :2], node1[:, :2]), "1-hop": (edge1, node1), "2-hop": (edge2, node2),
            "oracle": (edge_o, node_o)}

# Everything learn needs from the saved instances: labels, features, and the energy functions.
def dataset(problem):
    df = pd.read_csv(rf.CSV)
    seen, res = set(), []
    for f in build_files(problem):
        for d in rp.load(f):
            if d["draw"] < DRAWS[problem] and (d["shape"], d["draw"]) not in seen:
                seen.add((d["shape"], d["draw"]))
                res.append(d)
    data = []
    for d in sorted(res, key=lambda d: (d["draw"], d["shape"])):
        if not d.get("confirmed") or d.get("pick") is None:
            continue
        prob, _ = instance(problem, df, d["shape"], d["draw"])
        el, nl, off = labels(prob, d["pick"])
        data.append({"rec": d, "prob": prob, "edge": el, "node": nl, "off": off,
                     "feat": feature_sets(prob, d["opt"])})
    return res, data

# The most common value in v.
def most_common(v):
    vals, cnt = np.unique(v, return_counts=True)
    return vals[np.argmax(cnt)]

# Predict the labels of every edge and node in every instance, with cross validation over instances.  
def predict(data, kind, target, fs, fold_of, model="boosted", frac=1.0):
    pred, base_major, base_pos = [None] * len(data), [None] * len(data), [None] * len(data)
    lab = lambda d: np.abs(d[kind]) if target == "size" else d[kind]
    col = 0 if kind == "edge" else 1
    for f in sorted(set(fold_of)):
        tr = [i for i in range(len(data)) if fold_of[i] != f]
        te = [i for i in range(len(data)) if fold_of[i] == f]
        if frac < 1.0:
            tr = sorted(np.random.default_rng(1000 + int(f)).permutation(tr)[:max(2, int(round(frac * len(tr))))])
        X = np.vstack([data[i]["feat"][fs][col] for i in tr])
        y = np.concatenate([lab(data[i]) for i in tr])
        if len(set(y)) == 1:
            fit = None
        elif model == "forest":
            fit = RandomForestClassifier(n_estimators=300, random_state=0).fit(X, y)
        else:
            fit = HistGradientBoostingClassifier(random_state=0).fit(X, y)
        major = most_common(y)
        for i in te:
            Xi = data[i]["feat"][fs][col]
            pred[i] = np.full(len(Xi), major) if fit is None else fit.predict(Xi)
            base_major[i] = np.full(len(Xi), major)
            same = [lab(data[k]) for k in tr if data[k]["rec"]["shape"] == data[i]["rec"]["shape"]]
            if same:
                A = np.array(same)
                base_pos[i] = np.array([most_common(A[:, a]) for a in range(A.shape[1])])
    return pred, base_major, base_pos

# Accuracy of a list of per-instance predictions against the labels, per label and per instance.
def score(data, kind, target, pred):
    lab = lambda d: np.abs(d[kind]) if target == "size" else d[kind]
    ok = [(p == lab(d)) for d, p in zip(data, pred) if p is not None]
    if not ok:
        return None, None
    return float(np.mean(np.concatenate(ok))), float(np.mean([o.all() for o in ok]))

# Polish from x with at most iters L-BFGS-B iterations.  Returns the energy and the number of gradient calls.
def short_polish(grad, x, iters):
    r = minimize(lambda z: grad(z, True), x, jac=True, method="L-BFGS-B",
                 options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": iters})
    return float(r.fun), int(r.nfev)

# Lowest energy over every sign pattern of the predicted sizes.  Edges of size 2 stay +2.
def best_signs(prob, energy_batch, es, ns):
    free = [("e", a) for a in range(len(es)) if es[a] == 1] + [("n", j) for j in range(len(ns)) if ns[j] > 0]
    if len(free) > SIGN_CAP:
        return None, None
    el, nl = np.array(es, int), np.array(ns, int)
    k = len(free)
    signs = 1 - 2 * ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1)
    X = np.tile(point(prob, el, nl), (1 << k, 1))
    T = len(prob["terms"])
    for i, (kind, a) in enumerate(free):
        X[:, a if kind == "e" else T + a] *= signs[:, i]
    vals = np.concatenate([energy_batch(X[lo:lo + 4096]) for lo in range(0, len(X), 4096)])
    return float(vals.min()), X[int(np.argmin(vals))]

def learn(problem):
    t0 = time.time()
    res, data = dataset(problem)
    N = len(data)
    shapes = sorted({d["rec"]["shape"] for d in data})
    print("\n%s: %d instances built, %d confirmed and used, %d not confirmed" % (problem, len(res), N, len(res) - N))
    seeds = [len(d["seeds"]) for d in res]
    print("   seeds used: %s" % {k: seeds.count(k) for k in sorted(set(seeds))})
    print("   picks on the pi/8 grid: %d of %d   floor = exact classical optimum: %d of %d   flat directions at pick: %d of %d"
          % (sum(d["off"] < GRID_TOL for d in data), N, sum(abs(d["rec"]["ratio"] - 1) < 1e-6 for d in data), N,
             sum(d["rec"]["flat"] > 0 for d in data), N))
    for kind in ("edge", "node"):
        allv = np.concatenate([d[kind] for d in data])
        print("   %s labels (units of pi/8): %s" % (kind, {int(k): int(v) for k, v in zip(*np.unique(allv, return_counts=True))}))
    star_edges, active, star_inst, hub_center, centered = 0, 0, 0, 0, 0
    for d in data:
        edges, n = d["prob"]["edges"], d["prob"]["n"]
        act = [e for e in range(len(edges)) if abs(d["edge"][e]) == 2]
        ok = [e for e in act if (d["node"][edges[e][0]] == 0) != (d["node"][edges[e][1]] == 0)]
        star_edges += len(ok)
        active += len(act)
        star_inst += len(ok) == len(act) and not np.any(np.abs(d["edge"]) == 1)
        cnt = np.zeros(n)
        for e in ok:
            cnt[edges[e][0] if d["node"][edges[e][0]] == 0 else edges[e][1]] += 1
        if cnt.max() > 0 and (cnt == cnt.max()).sum() == 1:
            W = np.abs(2 * np.asarray(d["prob"]["c"], float))
            wdeg = np.array([sum(W[e] for e, ed in enumerate(edges) if j in ed) for j in range(n)])
            centered += 1
            hub_center += int(np.argmax(cnt)) == int(np.argmax(wdeg))
    print("   stars: %d of %d edges at pi/4 join a node with beta = 0 (center) to one with beta != 0 (leaf); "
          "%d of %d picks are only such stars" % (star_edges, active, star_inst, N))
    print("   the center with the most pi/4 edges is the node with the largest weighted degree in %d of %d picks"
          % (hub_center, centered))
    for s in shapes:
        pats = {tuple(d["edge"]) + tuple(d["node"]) for d in data if d["rec"]["shape"] == s}
        sizes = {tuple(np.abs(d["edge"])) + tuple(np.abs(d["node"])) for d in data if d["rec"]["shape"] == s}
        k = sum(d["rec"]["shape"] == s for d in data)
        print("   shape %d: %d instances, %d different signed label patterns, %d different size patterns" % (s, k, len(pats), len(sizes)))

    rng = np.random.default_rng(0)
    cv = list(rng.permutation(N) % FOLDS)
    out_shape = [d["rec"]["shape"] for d in data]
    table, preds = {}, {}
    for split, fold_of in (("new weights", cv), ("new shape", out_shape)):
        for target in ("size", "signed"):
            for fs in FEATURES:
                row = {}
                for kind in ("edge", "node"):
                    pred, major, pos = predict(data, kind, target, fs, fold_of)
                    preds[(split, target, fs, kind)] = pred
                    preds[(split, target, "position", kind)] = pos
                    preds[("true", target, "labels", kind)] = [np.abs(d[kind]) if target == "size" else d[kind] for d in data]
                    row[kind] = score(data, kind, target, pred)
                    row[kind + " majority"] = score(data, kind, target, major)
                    row[kind + " position"] = score(data, kind, target, pos)
                both = [np.concatenate([(pe == (np.abs(d["edge"]) if target == "size" else d["edge"])),
                                        (pn == (np.abs(d["node"]) if target == "size" else d["node"]))]).all()
                        for d, pe, pn in zip(data, preds[(split, target, fs, "edge")], preds[(split, target, fs, "node")])]
                row["instance"] = float(np.mean(both))
                table[(split, target, fs)] = row

    print("\n   label accuracy (per label / whole instance right)        edges                 nodes          instance")
    for split in ("new weights", "new shape"):
        for target in ("size", "signed"):
            r0 = table[(split, target, "own")]
            fmt = lambda v: "  -  " if v is None or v[0] is None else "%.3f" % v[0]
            print("   %-11s %-6s most common label             %s                 %s" % (split, target, fmt(r0["edge majority"]), fmt(r0["node majority"])))
            if split == "new weights":
                print("   %-11s %-6s most common at that position  %s                 %s" % (split, target, fmt(r0["edge position"]), fmt(r0["node position"])))
            for fs in FEATURES:
                r = table[(split, target, fs)]
                print("   %-11s %-6s features %-7s               %.3f (%.3f)         %.3f (%.3f)     %.3f"
                      % (split, target, fs, r["edge"][0], r["edge"][1], r["node"][0], r["node"][1], r["instance"]))

    print("\n   cross-checks, new weights, 2-hop features: label accuracy edges / nodes, and whole instance right")
    checks = {}
    for name, kw in (("random forest", {"model": "forest"}), ("boosted, 25% of training", {"frac": 0.25}),
                     ("boosted, 50% of training", {"frac": 0.5}), ("boosted, all training", {})):
        for target in ("size", "signed"):
            pe = predict(data, "edge", target, "2-hop", cv, **kw)[0]
            pn = predict(data, "node", target, "2-hop", cv, **kw)[0]
            se, sn = score(data, "edge", target, pe), score(data, "node", target, pn)
            whole = float(np.mean([np.concatenate([a == (np.abs(d["edge"]) if target == "size" else d["edge"]),
                                                   b == (np.abs(d["node"]) if target == "size" else d["node"])]).all()
                                   for d, a, b in zip(data, pe, pn)]))
            checks[name + " | " + target] = {"edge": se[0], "node": sn[0], "instance": whole}
            print("   %-26s %-6s %.3f / %.3f   %.3f" % (name, target, se[0], sn[0], whole))

    print("\n   energy test (out of fold): share of instances whose angles reach the floor, and the median energy / floor")
    print("   (true labels: the pick's own labels, a check; position: the most common label at that position in the shape)")
    energy_rows = {}
    for split, fs in (("true", "labels"), ("new weights", "position"), ("new weights", "2-hop"), ("new shape", "2-hop"),
                      ("new weights", "oracle")):
        at, pol, enum_at, enum_pol, ratio, ratio_enum, calls = [], [], [], [], [], [], []
        for i, d in enumerate(data):
            prob, floor = d["prob"], d["rec"]["floor"]
            energy, energy_batch, grad, D, _ = rp.make_energy(prob["n"], prob["terms"], prob["c"], prob["const"], P)
            x = point(prob, preds[(split, "signed", fs, "edge")][i], preds[(split, "signed", fs, "node")][i])
            e0 = energy(x)
            e1, nf = short_polish(grad, x, POLISH_ITERS)
            at.append(e0 <= floor + rf.TOL_E)
            pol.append(e1 <= floor + rf.TOL_E)
            ratio.append(e0 / floor)
            if pol[-1]:
                calls.append(nf)
            eb, xb = best_signs(prob, energy_batch, preds[(split, "size", fs, "edge")][i], preds[(split, "size", fs, "node")][i])
            if eb is None:
                enum_at.append(False)
                enum_pol.append(False)
                ratio_enum.append(np.nan)
                continue
            enum_at.append(eb <= floor + rf.TOL_E)
            enum_pol.append(short_polish(grad, xb, POLISH_ITERS)[0] <= floor + rf.TOL_E)
            ratio_enum.append(eb / floor)
        r = {"predicted at floor": float(np.mean(at)), "polished at floor": float(np.mean(pol)),
             "size+signs at floor": float(np.mean(enum_at)), "size+signs polished at floor": float(np.mean(enum_pol)),
             "median predicted / floor": float(np.median(ratio)), "median size+signs / floor": float(np.nanmedian(ratio_enum)),
             "median gradient calls of a successful polish": float(np.median(calls)) if calls else None}
        energy_rows[split + " | " + fs] = r
        print("   %-11s %-8s predicted %.3f (median E/floor %.3f)   polished <= %d steps %.3f (median %s gradient calls)"
              "   size+signs %.3f (median %.3f), polished %.3f"
              % (split, fs, r["predicted at floor"], r["median predicted / floor"], POLISH_ITERS, r["polished at floor"],
                 "%.0f" % r["median gradient calls of a successful polish"] if calls else "-", r["size+signs at floor"],
                 r["median size+signs / floor"], r["size+signs polished at floor"]))

    hit, hit_short, calls, calls_short = [], [], [], []
    for i, d in enumerate(data):
        prob, rec = d["prob"], d["rec"]
        energy, _, grad, D, _ = rp.make_energy(prob["n"], prob["terms"], prob["c"], prob["const"], P)
        per = rp.periods(prob["terms"], prob["n"], P, prob["c"])
        r_rng = np.random.default_rng(1000 + i)
        for _ in range(RANDOM_STARTS):
            x0 = (r_rng.random(D) - 0.5) * per
            r = rf.polish(energy, grad, x0)
            hit.append(r.fun <= rec["floor"] + rf.TOL_E)
            calls.append(r.nfev)
            e1, nf = short_polish(grad, x0, POLISH_ITERS)
            hit_short.append(e1 <= rec["floor"] + rf.TOL_E)
            calls_short.append(nf)
    random_row = {"random start at floor": float(np.mean(hit)), "mean gradient calls per start": float(np.mean(calls)),
                  "gradient calls per floor hit": float(np.sum(calls) / max(1, np.sum(hit))),
                  "random start, <= POLISH_ITERS steps, at floor": float(np.mean(hit_short)),
                  "gradient calls per floor hit, <= POLISH_ITERS steps": float(np.sum(calls_short) / max(1, np.sum(hit_short)))}
    print("   random start + full polish: reaches the floor %.3f of the time, %.0f gradient calls per start, %.0f per floor hit"
          % (random_row["random start at floor"], random_row["mean gradient calls per start"], random_row["gradient calls per floor hit"]))
    print("   random start + <= %d steps: reaches the floor %.3f of the time, %.0f gradient calls per floor hit"
          % (POLISH_ITERS, random_row["random start, <= POLISH_ITERS steps, at floor"],
             random_row["gradient calls per floor hit, <= POLISH_ITERS steps"]))

    summary = {"problem": problem, "instances": len(res), "confirmed": N, "seeds": seeds,
               "on_grid": int(sum(d["off"] < GRID_TOL for d in data)),
               "stars": {"pi/4 edges": active, "center-leaf pi/4 edges": star_edges, "picks only stars": int(star_inst),
                         "picks with one main center": centered, "main center is largest weighted degree": hub_center},
               "table": {" | ".join(k): v for k, v in table.items()}, "checks": checks, "energy": energy_rows,
               "random": random_row,
               "secs": round(time.time() - t0, 1)}
    json.dump(summary, open(ROOT / "results" / ("learnability_summary_%s.json" % problem), "w"), indent=1)

def main():
    mode = sys.argv[1] if len(sys.argv) >= 2 else "all"
    todo = [sys.argv[2]] if len(sys.argv) >= 3 else PROBLEMS
    for problem in todo:
        if mode == "check":
            check(problem)
            continue
        if mode == "build_rev":
            build(problem, reverse=True)
            continue
        if mode in ("build", "all"):
            build(problem)
        if mode in ("learn", "all"):
            learn(problem)

if __name__ == "__main__":
    main()