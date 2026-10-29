"""
Usage:  python src/make_dataset.py [mode] [shard] [n_shards]
        python src/make_dataset.py merge [mode]
        python src/make_dataset.py compare
mode is full (default), pilot, pilot400 or test.  With no arguments it runs full, shard 0 of 1 (what the VS Code debugger does).
  full      every graph in COUNTS, written to results/dataset/
  pilot     the first PILOT_PER_N graphs of every node count in COUNTS, every one audited, written to results/dataset_pilot/
  pilot400  the same graphs with run_filter's old stopping rule (400 restarts), written to results/dataset_pilot400/
  test      larger held-out graphs from TEST_COUNTS and TEST_SEED, written to results/dataset_test/
  compare   compares the pilot and pilot400 picks graph by graph
On a cluster start n_shards jobs with shard = 0 .. n_shards-1.  Every shard writes its own file, so no two jobs write the
same file.  When they are all done, run merge once for that mode.
Graph i is the same graph however many shards there are, so the dataset does not depend on how it was split.
Changing COUNTS changes which graph each id is, so never change it after a full run has started.
Run every shard with the same numpy and networkx versions, since the random graph generators change between versions.
Each graph goes through run_filter.run (search, two-seed guard, inner shell, filter) with nothing reimplemented.
In the shard files edges, w_sorted and pick are in sorted edge order (run_filter's order, gammas then betas).
In dataset.csv Gammas and Weights are in nx.Graph(edges).edges() order, the same order as MaxCutMAQAOAData.csv.
A row is Usable when two seeds reach the same floor and both have the pick, the filter leaves one point,
the pick is on the floor with gradient below GRAD_TOL, and, for the graphs searched again on unrelated seeds,
that search gives the same pick.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import run_filter as rf
import graph_sweep as gs

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "dataset"
MERGED = OUT_DIR / "dataset.csv"

P = 1
WEIGHTED = True
STALL = 100 # restarts without a change before a seed stops; run_filter uses 400, 100 gave the same pick on 21 of 21 test graphs
PLAN_COUNTS = {5: 4000, 6: 12000, 7: 18000, 8: 22000, 9: 22000, 10: 12000, 11: 6000, 12: 4000} # graphs per node count, 100,000
TEST_COUNTS = {13: 300, 14: 300}
PLAN_SEED = 20261026
TEST_SEED = 20261027
PILOT_PER_N = 40
MODES = ["full", "pilot", "pilot400", "test"]
TYPES = ["er-0.3", "er-0.4", "er-0.5", "er-0.6", "3-regular", "4-regular", "tree", "bipartite",
         "barabasi-albert", "watts-strogatz"]
COUNTS, BASE_SEED = dict(PLAN_COUNTS), PLAN_SEED # the plan and seed in use; setup() switches them, build() and plan() read them
AUDIT_FRAC = 0.05 # fraction of graphs searched again on unrelated seeds to check the pick does not change
AUDIT_BASE = 10 ** 9 # audit seeds are AUDIT_BASE + the normal seed, far from every normal seed
AUT_CAP = 2000 # skip graphs with more automorphisms than this (only happens unweighted)
TRIES = 10000 # draws allowed before a family gives up on a connected graph
GRAD_TOL = 1e-6

# A random connected graph of one family on nodes 0..n-1, or None if the family does not exist at n.
def random_graph(kind, n, rng):
    def connected(make):
        for _ in range(TRIES):
            G = make(int(rng.integers(2 ** 31)))
            if G.number_of_nodes() == n and nx.is_connected(G):
                return G
        return None

    if kind.startswith("er-"):
        q = float(kind[3:])
        return connected(lambda s: nx.gnp_random_graph(n, q, seed=s))
    if kind == "3-regular":
        return None if n < 4 or n % 2 else connected(lambda s: nx.random_regular_graph(3, n, seed=s))
    if kind == "4-regular":
        return None if n < 5 else connected(lambda s: nx.random_regular_graph(4, n, seed=s))
    if kind == "tree":
        maker = getattr(nx, "random_labeled_tree", None) or nx.random_tree
        return maker(n, seed=int(rng.integers(2 ** 31)))
    if kind == "bipartite":
        if n < 4:
            return None
        a = int(rng.integers(2, n - 1))
        return connected(lambda s: nx.bipartite.random_graph(a, n - a, 0.5, seed=s))
    if kind == "barabasi-albert":
        return None if n < 4 else nx.barabasi_albert_graph(n, 2, seed=int(rng.integers(2 ** 31)))
    if kind == "watts-strogatz":
        return None if n < 6 else connected(lambda s: nx.watts_strogatz_graph(n, 4, 0.3, seed=s))
    raise ValueError(kind)

# Point COUNTS, BASE_SEED, OUT_DIR and the stopping rule at one mode.
def setup(mode):
    global COUNTS, BASE_SEED, OUT_DIR, MERGED
    if mode not in MODES:
        raise SystemExit("mode must be one of %s" % MODES)
    COUNTS, BASE_SEED = (dict(TEST_COUNTS), TEST_SEED) if mode == "test" else (dict(PLAN_COUNTS), PLAN_SEED)
    OUT_DIR = ROOT / "results" / ("dataset" if mode == "full" else "dataset_" + mode)
    MERGED = OUT_DIR / "dataset.csv"
    rf.STALL[P] = 400 if mode == "pilot400" else STALL

# Every graph in the dataset, cheapest node count first.  Graph i only depends on BASE_SEED and i.
def plan():
    out, i = [], 0
    for n in sorted(COUNTS):
        for _ in range(COUNTS[n]):
            out.append((i, n))
            i += 1
    return out

# The graphs a mode runs: every graph, or the first PILOT_PER_N of every node count for the pilots.
def selected(mode):
    out, seen = [], {}
    for (i, n) in plan():
        seen[n] = seen.get(n, 0) + 1
        if mode not in ("pilot", "pilot400") or seen[n] <= PILOT_PER_N:
            out.append((i, n))
    return out

# Build graph i: pick a family, draw the graph, shuffle the node labels, draw weights in sorted edge order.
def build(i, n):
    rng = np.random.default_rng([BASE_SEED, i])
    kinds = list(TYPES)
    while kinds:
        kind = kinds.pop(int(rng.integers(len(kinds))))
        G = random_graph(kind, n, rng)
        if G is not None:
            break
    else:
        raise RuntimeError("no family gives a connected graph on %d nodes" % n)
    lab = rng.permutation(n)
    edges = rf.canonical_edges([(lab[u], lab[v]) for (u, v) in G.edges()])
    m = len(edges)
    w = rf.draw_weights(m, int(rng.integers(2 ** 31))) if WEIGHTED else np.ones(m)
    audit = bool(rng.random() < AUDIT_FRAC)
    return kind, edges, w, audit

# Edges in the order nx.Graph(edges).edges() lists them, and where each one sits in the sorted list.
def ian_order(edges):
    srt = rf.canonical_edges(edges)
    nxo = [tuple(sorted((int(u), int(v)))) for (u, v) in nx.Graph(srt).edges()]
    where = {e: k for k, e in enumerate(srt)}
    return nxo, np.array([where[e] for e in nxo])

# Exact (weighted) max cut by trying every split with node 0 on one side.
def max_cut(n, edges, w):
    s = 1 - 2 * ((np.arange(1 << (n - 1))[:, None] >> np.arange(n)[None, :]) & 1)
    u = np.array([a for (a, _) in edges])
    v = np.array([b for (_, b) in edges])
    return float((0.5 * (1 - s[:, u] * s[:, v]) @ w).max())

# Distance between two picks of the same graph, minimised over the graph's symmetry images of the first one.
def pick_dist(n, edges, w, a, b):
    per = rf.periods(len(edges), n, P, w)
    _, images = rf.symmetry_group(edges, n, w, P)
    return min(rf.geodesic_dist(y, np.array(b), per) for y in images(np.array(a)))

# Search graph i again on seeds no normal run uses, and measure how far the new pick is from the old one.
def audit(i, n, edges, w, rec):
    sub = rf.run(i, edges, n, w, P, AUDIT_BASE + rec["seed"], WEIGHTED, 0)
    if sub["pick"] is None:
        return {"audit_same": False, "audit_dist": None, "audit_raw_dist": None, "audit_floor": None}
    per = rf.periods(len(edges), n, P, w)
    d = pick_dist(n, edges, w, rec["pick"], sub["pick"])
    raw = rf.geodesic_dist(np.array(rec["pick"]), np.array(sub["pick"]), per)
    same = abs(sub["floor"] - rec["floor"]) <= rf.TOL_E and d <= rf.TOL_D
    return {"audit_same": bool(same), "audit_dist": float(d), "audit_raw_dist": float(raw), "audit_floor": sub["floor"]}

# Run graph i through run_filter and return its record.  check overrides the drawn audit flag when it is not None.
def solve(i, n, check=None):
    t0 = time.time()
    kind, edges, w, drawn = build(i, n)
    check = drawn if check is None else check
    m = len(edges)
    base = {"id": i, "n": n, "m": m, "type": kind, "p": P, "weighted": WEIGHTED, "edges": [list(e) for e in edges],
            "w_sorted": [float(a) for a in w], "wl_hash": nx.weisfeiler_lehman_graph_hash(nx.Graph(edges)),
            "max_cut": max_cut(n, edges, w), "stall": rf.STALL.get(P), "base_seed": BASE_SEED,
            "networkx": nx.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    if gs.aut_count_capped(edges, n, w, AUT_CAP) is None:
        base.update({"skipped": "aut>%d" % AUT_CAP, "usable": False})
        return base
    rec = rf.run(i, edges, n, w, P, 1000 * i + 10 * P + 1, WEIGHTED, 0)
    base.update({k: rec[k] for k in ["seed", "floor", "r_min", "shell", "n_auto", "filter", "groups", "used",
                                      "hits", "restarts", "nfev", "seeds", "seed_floors", "confirmed"]})
    base["secs_main"] = rec["secs"]
    if rec["pick"] is None:
        base.update({"pick": None, "usable": False})
        return base
    x = np.array(rec["pick"])
    energy, _, grad, _ = rf.make_energy(n, edges, w=w, p=P)
    base["pick"] = rec["pick"]
    base["energy"] = float(energy(x))
    base["grad_norm"] = float(np.linalg.norm(grad(x)))
    base["ratio"] = -base["energy"] / base["max_cut"]
    nxo, idx = ian_order(edges)
    base["gamma_edges"] = [list(e) for e in nxo]
    base["gammas"] = [float(a) for a in x[:P * m].reshape(P, m)[:, idx].ravel()]
    base["betas"] = [float(a) for a in x[P * m:]]
    base["weights"] = [float(a) for a in w[idx]]
    base["audit"] = check
    if check:
        base.update(audit(i, n, edges, w, rec))
    base["usable"] = bool(rec["confirmed"] and rec["filter"] == 1 and rec["groups"] == 1
                          and abs(base["energy"] - rec["floor"]) <= rf.TOL_E and base["grad_norm"] <= GRAD_TOL
                          and base.get("audit_same", True))
    base["secs"] = round(time.time() - t0, 1)
    return base

# Ids already in a shard file.  A job killed while writing leaves a cut-off last line, so end the file
# with a newline first, otherwise the next record would be glued onto it.
def load_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path, "rb+") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
    for line in open(path):
        try:
            done.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            pass
    return done

def load_records(folder):
    recs = {}
    for path in sorted(Path(folder).glob("shard_*.jsonl")):
        for line in open(path):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            recs[d["id"]] = d
    return recs

def run_shard(shard, n_shards, mode="full"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / ("shard_%04d_of_%04d.jsonl" % (shard, n_shards))
    done = load_done(path)
    todo = [(i, n) for (i, n) in selected(mode) if i % n_shards == shard and i not in done]
    check = {"pilot": True, "pilot400": False}.get(mode)
    print("%s shard %d of %d: %d graphs done, %d to go, writing %s" % (mode, shard, n_shards, len(done), len(todo), path),
          flush=True)
    for (i, n) in todo:
        print("\n== graph %d  n %d" % (i, n), flush=True)
        rec = solve(i, n, check)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

# Collect every shard file of the current mode into one csv with the columns of MaxCutMAQAOAData.csv first, then the checks.
def merge():
    recs = load_records(OUT_DIR)
    if not recs:
        raise SystemExit("no shard files in %s" % OUT_DIR)
    rows = []
    for i in sorted(recs):
        d = recs[i]
        if d.get("pick") is None:
            continue
        rows.append({"Id": i, "Number of Nodes": d["n"], "Type": d["type"],
                     "Edges": str([tuple(e) for e in d["edges"]]),
                     "Gamma Edges": str([tuple(e) for e in d["gamma_edges"]]),
                     "Weights": str(d["weights"]), "p": d["p"], "Betas": str(d["betas"]),
                     "Gammas": str(d["gammas"]), "Energy": d["energy"], "Max Cut": d["max_cut"],
                     "Ratio": d["ratio"], "Radius": d["r_min"], "Usable": d["usable"],
                     "Confirmed": d["confirmed"], "Seeds Tried": len(d["seeds"]), "Filter Left": d["filter"],
                     "Groups": d["groups"], "Automorphisms": d["n_auto"], "Hits": d["hits"],
                     "Restarts": d["restarts"], "Grad Norm": d["grad_norm"], "Audited": d["audit"],
                     "Audit Same": d.get("audit_same"), "Audit Dist": d.get("audit_dist"),
                     "Audit Raw Dist": d.get("audit_raw_dist"),
                     "WL Hash": d["wl_hash"], "Secs": d["secs"], "Secs Main": d.get("secs_main")})
    df = pd.DataFrame(rows)
    df.to_csv(MERGED, index=False)
    print("wrote %d rows to %s (%d records read, %d without a pick)\n"
          % (len(df), MERGED, len(recs), len(recs) - len(df)))
    print("%3s %7s %7s %10s %7s %8s %10s %10s" % ("n", "graphs", "usable", "confirmed", "audited",
                                                   "audit ok", "mean main", "core-hrs"))
    for n, g in df.groupby("Number of Nodes"):
        a = g[g["Audited"] == True]
        print("%3d %7d %7d %10d %7d %8d %10.1f %10.2f" % (n, len(g), g["Usable"].sum(), g["Confirmed"].sum(), len(a),
                                                        (a["Audit Same"] == True).sum(), g["Secs Main"].mean(),
                                                        g["Secs"].sum() / 3600))

# Graph by graph, does the pilot (STALL) pick match the pilot400 pick?
def compare():
    setup("pilot")
    a = load_records(ROOT / "results" / "dataset_pilot")
    b = load_records(ROOT / "results" / "dataset_pilot400")
    both = sorted(i for i in a if i in b and a[i].get("pick") is not None and b[i].get("pick") is not None)
    if not both:
        raise SystemExit("no graph has a pick in both results/dataset_pilot and results/dataset_pilot400")
    out = []
    for i in both:
        d = a[i]
        _, edges, w, _ = build(i, d["n"])
        dist = pick_dist(d["n"], edges, w, d["pick"], b[i]["pick"])
        same = abs(d["floor"] - b[i]["floor"]) <= rf.TOL_E and dist <= rf.TOL_D
        out.append({"id": i, "n": d["n"], "same": bool(same), "dist": float(dist), "floor": d["floor"],
                    "floor400": b[i]["floor"], "secs": d["secs_main"], "secs400": b[i]["secs_main"]})
    json.dump(out, open(ROOT / "results" / "dataset_pilot" / "compare.json", "w"), indent=1)
    print("%3s %7s %6s %10s %10s" % ("n", "graphs", "same", "mean main", "main 400"))
    for n in sorted({r["n"] for r in out}):
        g = [r for r in out if r["n"] == n]
        print("%3d %7d %6d %10.1f %10.1f" % (n, len(g), sum(r["same"] for r in g), np.mean([r["secs"] for r in g]),
                                            np.mean([r["secs400"] for r in g])))
    bad = [r for r in out if not r["same"]]
    print("\n%d of %d graphs give the same pick%s" % (len(out) - len(bad), len(out), ":" if bad else ""))
    for r in bad:
        print("  id %d  n %d  dist %.2e  floor %.6f vs %.6f" % (r["id"], r["n"], r["dist"], r["floor"], r["floor400"]))

def main():
    args = sys.argv[1:]
    if args and args[0] == "compare":
        compare()
        return
    if args and args[0] == "merge":
        setup(args[1] if len(args) >= 2 else "full")
        merge()
        return
    mode = args.pop(0) if args and args[0] in MODES else "full"
    setup(mode)
    shard = int(args[0]) if len(args) >= 1 else 0
    n_shards = int(args[1]) if len(args) >= 2 else 1
    if not 0 <= shard < n_shards:
        raise SystemExit("shard must be in 0 .. n_shards-1")
    run_shard(shard, n_shards, mode)

if __name__ == "__main__":
    main()