"""
Just noticed the floor was always a multiple of 0.5, just wanted to test it out
"""
import sys
import json
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from maqaoa_core import make_energy


# distance to the nearest multiple of 1/2
def frac_half(v):
    return abs(v * 2 - round(v * 2)) / 2


# brute force
def brute_maxcut(n, edges):
    best = 0
    for s in range(1 << n):
        b = [(s >> i) & 1 for i in range(n)]
        best = max(best, sum(1 for (u, v) in edges if b[u] != b[v]))
    return best


RESTARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NGRAPH = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OUTPUT = sys.argv[3] if len(sys.argv) > 3 else "results.json"

rng0 = np.random.default_rng(11)

bad_floor, bad_local, done = 0, 0, 0
results = []

while done < NGRAPH:
    n = int(rng0.choice([6, 7, 8, 9]))
    p = float(rng0.choice([0.3, 0.4, 0.5, 0.6, 0.7]))

    G = nx.gnp_random_graph(
        n, p, seed=int(rng0.integers(1 << 30))
    )

    if G.number_of_edges() < 3:
        continue

    edges = list(G.edges())
    energy, energy_batch, grad, D = make_energy(n, edges, p=1)
    mc = brute_maxcut(n, edges)

    rng = np.random.default_rng(done)
    vals = []

    for _ in range(RESTARTS):
        x0 = rng.uniform(0, np.pi, D)
        r = minimize(
            energy,
            x0,
            jac=grad,
            method="L-BFGS-B",
            options={
                "ftol": 1e-18,
                "gtol": 1e-13,
                "maxiter": 20000,
            },
        )
        vals.append(float(r.fun))

    vals = np.array(vals)
    floor = float(vals.min())
    uniq = np.unique(np.round(vals, 7))
    dh = np.array([frac_half(v) for v in uniq])

    nhalf = int((dh < 1e-7).sum())
    is_bad_floor = frac_half(floor) > 1e-7
    is_bad_local = nhalf < len(uniq)

    if is_bad_floor:
        bad_floor += 1
    if is_bad_local:
        bad_local += 1

    results.append({
        "index": done,
        "n": n,
        "m": len(edges),
        "edges": [list(edge) for edge in edges],
        "p": p,
        "maxcut": mc,
        "floor": floor,
        "distance_from_half_integer": float(frac_half(floor)),
        "nlocal": int(len(uniq)),
        "nhalf": nhalf,
        "worstlocal": float(dh.max()),
        "gap": float(floor + mc),
        "is_floor_half_integer": not is_bad_floor,
        "has_non_half_integer_local_min": is_bad_local,
        "local_minima": [float(v) for v in uniq],
    })

    print(
        "%3d %2d %3d %4d %11.7f %10.1e %8d %7d %10.1e %6.1f"
        % (
            done,
            n,
            len(edges),
            mc,
            floor,
            frac_half(floor),
            len(uniq),
            nhalf,
            dh.max(),
            floor + mc,
        ),
        flush=True,
    )

    done += 1


output = {
    "parameters": {
        "restarts": RESTARTS,
        "n_graphs": NGRAPH,
        "rng_seed": 11,
    },
    "summary": {
        "graphs_with_floor_not_half_integer": bad_floor,
        "graphs_with_non_half_integer_local_min": bad_local,
        "total_graphs": NGRAPH,
    },
    "results": results,
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print()
print(
    "graphs whose FLOOR is not a half-integer:            %d / %d"
    % (bad_floor, NGRAPH)
)
print(
    "graphs with at least one non-half-integer local min: %d / %d"
    % (bad_local, NGRAPH)
)
print(f"Results exported to {OUTPUT}")