"""
Usage:
  python src/graph_sweep.py --build                 write data/sweep_graphs.csv
  python src/graph_sweep.py --estimate              time one polish per case, print a cost table
  python src/graph_sweep.py --selftest              check the driver against results/ordered_filter.json
  python src/graph_sweep.py --run                   run every case
  python src/graph_sweep.py --run --nodes 6,7 --layers 1,2 --types cycle,complete
  python src/graph_sweep.py --run --max-secs 3600   stop launching new cases after an hour

Builds a library of graph types (Erdos-Renyi at three densities, regular graphs,
cycles, paths, stars, wheels, complete, complete bipartite, grids, trees, prisms,
barbells, Petersen) at several node counts, then runs the run_filter.py pipeline
on each one for p = 1, 2, 3, unweighted and with random edge weights.

Nothing about the method is reimplemented here.  This file imports run_filter and
calls run_filter.run(), which is the same function src/run_filter.py calls from
the command line.  The only things changed are the csv it reads, the json it
writes, and the addition of a p=3 entry to the restart table.

Results are appended to results/graph_sweep.json after every case, so the run can
be stopped and restarted at any time and will pick up where it left off.
"""

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from networkx.algorithms.isomorphism import GraphMatcher

import run_filter as rf

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "graphs_extra.csv"
OUT = ROOT / "results" / "graph_sweep.json"

NODES = [6, 7, 8, 9, 10]
LAYERS = [1, 2, 3]

# Restarts per layer count.  p=1 and p=2 are run_filter's own numbers, p=3 is new.
# More parameters means more places to get stuck, so p=3 is the weakest of the
# three and its shells should be read as the loosest lower bound.
RESTARTS = {1: 100, 2: 25, 3: 15}

# Weighted runs use this many independent weight draws per graph and layer count.
N_TRIALS = 2


# Each entry returns None when the type does not exist at that node count.
# Every graph is relabelled onto 0..n-1 and checked for connectivity before use.
def library():
    def er(n, m, seed0=0):
        for s in range(seed0, seed0 + 10000):
            G = nx.gnm_random_graph(n, m, seed=s)
            if nx.is_connected(G):
                return G
        return None

    def regular(d, n, seed0=0):
        if n <= d or (d * n) % 2:
            return None
        for s in range(seed0, seed0 + 10000):
            G = nx.random_regular_graph(d, n, seed=s)
            if nx.is_connected(G):
                return G
        return None

    def tree(n, seed0=0):
        maker = getattr(nx, "random_labeled_tree", None) or nx.random_tree
        return maker(n, seed=seed0)

    def grid(n):
        if n % 2:
            return None
        return nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, n // 2))

    def bipartite(n):
        return nx.convert_node_labels_to_integers(
            nx.complete_bipartite_graph(n // 2, n - n // 2))

    def prism(n):
        if n % 2 or n < 6:
            return None
        return nx.circular_ladder_graph(n // 2)

    def barbell(n):
        if n < 6 or n % 2:
            return None
        return nx.barbell_graph(n // 2, 0)

    def wheel(n):
        return nx.wheel_graph(n) if n >= 4 else None

    def petersen(n):
        return nx.petersen_graph() if n == 10 else None

    # Densities are fractions of the complete graph, with a floor of n edges so
    # the sparse case still has a cycle in it rather than being a near-tree.
    def er_frac(n, frac):
        return er(n, max(n, int(round(frac * n * (n - 1) / 2))))

    return [
        ("er-sparse", lambda n: er_frac(n, 0.30)),
        ("er-medium", lambda n: er_frac(n, 0.50)),
        ("er-dense", lambda n: er_frac(n, 0.70)),
        ("3-regular", lambda n: regular(3, n)),
        ("4-regular", lambda n: regular(4, n)),
        ("cycle", lambda n: nx.cycle_graph(n)),
        ("path", lambda n: nx.path_graph(n)),
        ("tree", lambda n: tree(n)),
        ("star", lambda n: nx.star_graph(n - 1)),
        ("wheel", wheel),
        ("grid", grid),
        ("prism", prism),
        ("barbell", barbell),
        ("bipartite", bipartite),
        ("complete", lambda n: nx.complete_graph(n)),
        ("petersen", petersen),
    ]


# Write the library to a csv with the two columns run_filter.main reads out of
# MaxCutMAQAOAData.csv, so the row index in that file is the row index here.
def build_csv():
    rows = []
    for n in NODES:
        for (name, make) in library():
            G = make(n)
            if G is None:
                continue
            G = nx.convert_node_labels_to_integers(G)
            if G.number_of_nodes() != n or not nx.is_connected(G):
                continue
            edges = rf.canonical_edges(G.edges())
            rows.append({"Number of Nodes": n, "Type": name,
                         "Edges": str([tuple(e) for e in edges])})
    df = pd.DataFrame(rows, columns=["Number of Nodes", "Type", "Edges"])
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)
    return df


def load_csv():
    if not CSV.exists():
        return build_csv()
    return pd.read_csv(CSV)


def edges_of(df, row):
    return [tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])]


# Same weight draw and seed scheme run_filter.main uses, so a case here is
# reproducible the same way a case there is.
def weights_for(m, row, trial, weighted):
    if not weighted:
        return np.ones(m)
    return rf.draw_weights(m, 100 * row + trial)


def seed_for(row, p, trial):
    return 1000 * row + 10 * p + (0 if trial is None else trial + 1)


# run_filter.symmetry_group materialises every automorphism, and canonicalize()
# builds 2*|Aut| images of every shell point.  A star on 10 nodes has 9! = 362880
# automorphisms, so those cases are not runnable with this pipeline at all.
# Count with an early exit and skip anything above the cap.
def aut_count_capped(edges, n, w, cap):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for k, (u, v) in enumerate(rf.canonical_edges(edges)):
        G.add_edge(u, v, weight=float(w[k]))
    em = lambda a, b: abs(a["weight"] - b["weight"]) < 1e-12
    it = GraphMatcher(G, G, edge_match=em).isomorphisms_iter()
    count = 0
    for _ in it:
        count += 1
        if count > cap:
            return None
    return count


def case_key(row, p, weighted, trial):
    return "%d_p%d_%s_t%s" % (row, p, "w" if weighted else "u", trial)


def build_cases(df, nodes, layers, types, weighted_modes, trials):
    cases = []
    for row in df.index:
        n = int(df.loc[row, "Number of Nodes"])
        typ = str(df.loc[row, "Type"])
        if n not in nodes or (types and typ not in types):
            continue
        m = len(rf.canonical_edges(edges_of(df, row)))
        for p in layers:
            for weighted in weighted_modes:
                for t in (list(range(trials)) if weighted else [None]):
                    cases.append({"row": row, "n": n, "m": m, "type": typ,
                                  "p": p, "weighted": weighted, "trial": t,
                                  "key": case_key(row, p, weighted, t)})
    # Cheapest first, so a run that is cut short still covers the small cases at
    # every layer count.  Cost of one energy call is 2^n, and L-BFGS-B needs
    # roughly D^2 of them, with D = p*(m+n).
    cost = lambda c: (2 ** c["n"]) * (c["p"] * (c["n"] + c["m"])) ** 2 * RESTARTS[c["p"]]
    return sorted(cases, key=cost)


# Time a couple of L-BFGS-B runs and scale up.  harvest() does `restarts` plain
# polishes and then, for each of the four penalty strengths, `restarts` penalised
# minimisations each followed by a polish, so about 9*restarts optimisations.
def estimate(df, cases, reps=2):
    print("%5s %-11s %2s %2s %2s %4s %4s %10s %12s"
          % ("row", "type", "n", "m", "p", "D", "R", "s/polish", "est. total"))
    total = 0.0
    for c in cases:
        edges = edges_of(df, c["row"])
        w = weights_for(c["m"], c["row"], 0, c["weighted"])
        energy, _, grad, D = rf.make_energy(c["n"], edges, w=w, p=c["p"])
        rng = np.random.default_rng(0)
        t0 = time.time()
        for _ in range(reps):
            rf.polish(energy, grad, rng.uniform(0, np.pi, D))
        dt = (time.time() - t0) / reps
        est = dt * 9 * RESTARTS[c["p"]]
        total += est
        print("%5d %-11s %2d %2d %2d %4d %4d %10.3f %10.0f s"
              % (c["row"], c["type"], c["n"], c["m"], c["p"], D,
                 RESTARTS[c["p"]], dt, est), flush=True)
    print("\n%d cases, estimated %.0f s total (%.1f h)" % (len(cases), total, total / 3600.0))


# Run one row of the original csv through the driver and compare with the stored
# result, to confirm the driver reproduces run_filter.py exactly.
def selftest(row=13, p=1):
    stored = json.load(open(ROOT / "results" / "ordered_filter.json"))
    ref = [d for d in stored if d["row"] == row and d["p"] == p and not d["weighted"]]
    if not ref:
        raise SystemExit("no stored unweighted result for row %d p %d" % (row, p))
    ref = ref[0]
    df = pd.read_csv(ROOT / "data" / "MaxCutMAQAOAData.csv")
    edges = [tuple(t) for t in ast.literal_eval(df.loc[row, "Edges"])]
    n = int(df.loc[row, "Number of Nodes"])
    m = len(rf.canonical_edges(edges))
    got = rf.run(row, edges, n, np.ones(m), p, seed_for(row, p, None), False, None)
    keys = ["floor", "shell", "exact", "n_auto", "filter", "groups", "r_min"]
    print("\n%-8s %14s %14s" % ("field", "driver", "stored"))
    ok = True
    for k in keys:
        print("%-8s %14s %14s" % (k, got[k], ref[k]))
        ok = ok and got[k] == ref[k]
    print("\nmatch: %s" % ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--nodes", default=",".join(str(x) for x in NODES))
    ap.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
    ap.add_argument("--types", default="")
    ap.add_argument("--weights", default="both", choices=["both", "on", "off"])
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--max-secs", type=float, default=float("inf"))
    ap.add_argument("--aut-cap", type=int, default=2000)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    df = build_csv() if args.build else load_csv()
    if args.build:
        print("%5s %3s %-11s %4s" % ("row", "n", "type", "m"))
        for r in df.index:
            e = ast.literal_eval(df.loc[r, "Edges"])
            print("%5d %3d %-11s %4d" % (r, df.loc[r, "Number of Nodes"],
                                          df.loc[r, "Type"], len(e)))
        print("\nwrote %d graphs to %s" % (len(df), CSV))
        if not (args.estimate or args.run):
            return

    nodes = [int(x) for x in args.nodes.split(",") if x]
    layers = [int(x) for x in args.layers.split(",") if x]
    types = [x for x in args.types.split(",") if x]
    modes = {"both": [False, True], "on": [True], "off": [False]}[args.weights]
    cases = build_cases(df, nodes, layers, types, modes, args.trials)

    if args.estimate:
        estimate(df, cases)
        return
    if not args.run:
        print(__doc__)
        return

    rf.RESTARTS = dict(RESTARTS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(OUT)) if OUT.exists() else []
    done = {d["key"] for d in res if "key" in d}

    t_start = time.time()
    for c in cases:
        if c["key"] in done:
            continue
        if time.time() - t_start > args.max_secs:
            print("time budget reached, stopping", flush=True)
            break
        edges = edges_of(df, c["row"])
        w = weights_for(c["m"], c["row"], c["trial"], c["weighted"])
        na = aut_count_capped(edges, c["n"], w, args.aut_cap)
        if na is None:
            print("row %2d %-11s p %d %s: more than %d automorphisms, skipped"
                  % (c["row"], c["type"], c["p"],
                     "weighted" if c["weighted"] else "unweighted", args.aut_cap),
                  flush=True)
            res.append({"key": c["key"], "row": c["row"], "type": c["type"],
                        "n": c["n"], "m": c["m"], "p": c["p"],
                        "weighted": c["weighted"], "trial": c["trial"],
                        "skipped": "aut>%d" % args.aut_cap})
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        print("\n== row %d  %s  n %d  m %d  p %d  %s  trial %s"
              % (c["row"], c["type"], c["n"], c["m"], c["p"],
                 "weighted" if c["weighted"] else "unweighted", c["trial"]),
              flush=True)
        rec = rf.run(c["row"], edges, c["n"], w, c["p"],
                     seed_for(c["row"], c["p"], c["trial"]),
                     c["weighted"], c["trial"])
        rec["key"] = c["key"]
        rec["type"] = c["type"]
        res.append(rec)
        json.dump(res, open(OUT, "w"), indent=1)

    summary(res, cases)


def summary(res, cases):
    keys = {c["key"] for c in cases}
    shown = [d for d in res if d.get("key") in keys and "skipped" not in d]
    if not shown:
        return
    print("\n%5s %-11s %2s %2s %2s %6s %6s %7s %6s %7s %8s"
          % ("row", "type", "n", "m", "p", "trial", "floor", "shell", "|Aut|",
             "filter", "1 group"))
    for d in sorted(shown, key=lambda d: (d["n"], d.get("type", ""), d["p"],
                                          d["weighted"], str(d["trial"]))):
        print("%5d %-11s %2d %2d %2d %6s %6.1f %7d %6d %7s %8s"
              % (d["row"], d.get("type", "?"), d["n"], d["m"], d["p"],
                 "u" if not d["weighted"] else str(d["trial"]), d["floor"],
                 d["shell"], d["n_auto"], d["filter"], d["one_group"]))


if __name__ == "__main__":
    main()