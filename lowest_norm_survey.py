"""
Repeat the lowest-norm-minima procedure on every Erdos-Renyi graph in the
data set, not just the first one.

This is also used in sphere_uniformity.py to provide the cached lowest-norm shell.
"""

import ast
import json
import time
import numpy as np
import pandas as pd
import maqaoa_core as M

CSV = "MaxCutMAQAOAData.csv"
import os
RESTARTS_FLOOR = int(os.environ.get("RF", 100)) # restarts to establish floor
RESTARTS_HARVEST = int(os.environ.get("RH", 500)) # plain restarts to harvest minima
LAM = 0.5 # penalty strength on ||x||^2
TOL_E = 1e-6 # how close to the floor counts as "at the floor"
TOL_D = 1e-2 # torus distance below which two minima are the same
RAD_TOL = 1e-4 # radius band that defines "the lowest shell"

# Run the whole lowest-norm-minima pipeline on one graph and return a summary.
def analyze_graph(idx, edges, n, verbose=True):
    t0 = time.time()
    energy, _, grad, D = M.make_energy(n, edges, p=1)
    m = len(list(__import__("networkx").Graph(edges).edges()))

    # energy floor
    floor = M.find_floor(energy, grad, D, restarts=RESTARTS_FLOOR, seed=idx)

    # harvest low-norm minima
    from scipy.optimize import minimize
    rng = np.random.default_rng(1000 + idx)

    # Start with a penalized optimization to find low-norm minima.
    def penalized_start(lam):
        def pen(x):
            v = M.geodesic_vec(x, np.zeros(D))
            return energy(x) + lam * float(v @ v)
        def pen_grad(x):
            v = M.geodesic_vec(x, np.zeros(D))
            return grad(x) + 2 * lam * v
        r = minimize(pen, rng.uniform(0, np.pi, D), jac=pen_grad,
                     method="L-BFGS-B", options={"ftol": 1e-13, "gtol": 1e-10})
        rp = M.polish(energy, grad, r.x)
        return float(rp.fun), M.wrap_pi(rp.x)

    # Start with a plain optimization to get a broad coverage of the landscape.
    def plain_start(_lam=None):
        rp = M.polish(energy, grad, rng.uniform(0, np.pi, D))
        return float(rp.fun), M.wrap_pi(rp.x)

    plan = ([("plain", None)] * (RESTARTS_HARVEST // 4)
            + [("pen", 0.1)] * (RESTARTS_HARVEST // 4)
            + [("pen", 0.2)] * (RESTARTS_HARVEST // 4)
            + [("pen", 0.35)] * (RESTARTS_HARVEST // 4))

    # Collect all the minima found by the different starting strategies, and keep
    # only those that are at the floor.  Also keep only distinct minima (up to torus distance TOL_D).
    all_E = []
    all_X = []
    for kind, lam in plan:
        e, xw = plain_start() if kind == "plain" else penalized_start(lam)
        all_E.append(e)
        all_X.append(xw)
    all_E = np.array(all_E)

    floor = round(float(min(floor, all_E.min())), 6)
    # Filter the collected minima to only include those at the floor.
    hits = []
    distinct = []
    saturation = []
    for t, (e, xw) in enumerate(zip(all_E, all_X)):
        if e <= floor + TOL_E:
            hits.append(xw)
            if all(M.geodesic_dist(xw, q) > TOL_D for q in distinct):
                distinct.append(xw)
        if (t + 1) % 100 == 0:
            saturation.append((t + 1, len(distinct)))

    distinct = np.array(distinct)
    if len(distinct) == 0:
        return None

    # isolate the minimum-radius shell
    radii = np.array([M.norm_origin(x) for x in distinct])
    r_min = radii.min()
    shell = distinct[radii <= r_min + RAD_TOL]

    # quantization test
    snapped_ok = 0
    for x in shell:
        if energy(M.snap_to_grid(x, 4)) <= floor + 1e-9:
            snapped_ok += 1
    # also record the largest distance any shell point had to move to snap
    snap_shift = max(float(np.linalg.norm(M.geodesic_vec(M.snap_to_grid(x, 4), x)))
                     for x in shell)

    # structure
    # Express in units of pi/4 as signed integers: 0, +1, -1, 2.
    mult = np.array([M.grid_multiples(x, 4) for x in shell])
    signed = np.where(mult == 3, -1, mult)
    nonzero_counts = sorted(set((signed != 0).sum(1).tolist()))
    magnitudes = sorted(set(np.abs(signed).ravel().tolist()))
    always_zero = np.where((signed == 0).all(0))[0].tolist()
    gam_zero = [i for i in always_zero if i < m]
    bet_zero = [i - m for i in always_zero if i >= m]

    # symmetry group and orbits
    autos, images = M.symmetry_group(edges, p=1)
    labels = M.orbit_partition(list(shell), images, tol=TOL_D)
    n_orbits = len(set(labels.tolist()))

    # geometry of the shell
    Dm = M.geodesic_dist_matrix(shell)
    np.fill_diagonal(Dm, np.inf)
    nn = Dm.min(1) if len(shell) > 1 else np.array([np.nan])
    offdiag = Dm[np.isfinite(Dm)]

    # Cache the point sets so the uniformity analysis does not have to redo
    # the expensive harvest.
    np.savez(f"shell_row{idx}.npz", shell=shell, distinct=distinct,
             radii=radii, floor=floor, edges=np.array(edges, dtype=int))

    out = dict(
        row=idx, n=n, m=m, D=D, floor=floor, maxcut=-floor,
        r_min_over_pi=r_min / np.pi,
        r_min_units_pi4_squared=round((r_min / (np.pi / 4)) ** 2, 4),
        n_hits=len(hits), n_distinct_all_radii=len(distinct), n_shell=len(shell),
        saturation=saturation,
        snapped_ok=snapped_ok, snap_shift=snap_shift,
        nonzero_counts=nonzero_counts, magnitudes=magnitudes,
        n_always_zero=len(always_zero), gamma_always_zero=gam_zero,
        beta_always_zero=bet_zero,
        aut_order=len(autos), group_order=2 * len(autos), n_orbits=n_orbits,
        nn_min=float(nn.min()), nn_max=float(nn.max()),
        nn_unique=np.unique(np.round(nn, 6)).tolist(),
        pair_unique=np.unique(np.round(offdiag, 4)).tolist() if len(shell) > 1 else [],
        secs=round(time.time() - t0, 1),
    )

    if verbose:
        print(f"\n--- ER graph row {idx} (n={n}, m={m}, D={D}) ---")
        print(f"  floor = {floor}  (Max-Cut = {-floor:g})")
        print(f"  minimum radius = {r_min/np.pi:.9f} pi   "
              f"= sqrt({out['r_min_units_pi4_squared']:g}) * pi/4")
        print(f"  hits {len(hits)}/{RESTARTS_HARVEST}, distinct(all radii) "
              f"{len(distinct)}, ON LOWEST SHELL: {len(shell)}")
        print(f"  saturation (restarts, distinct): {saturation[::2]}")
        print(f"  all shell angles land on pi/4 grid and stay at floor: "
              f"{snapped_ok}/{len(shell)}   (max snap shift {snap_shift:.2e})")
        print(f"  |coeff| values present (units pi/4): {magnitudes}   "
              f"nonzero coords per point: {nonzero_counts}")
        print(f"  coords zero in EVERY shell point: {len(always_zero)} "
              f"(gammas {gam_zero}, betas {bet_zero})")
        print(f"  |Aut(G)| = {len(autos)}, group order (with sign flip) = "
              f"{2*len(autos)}  ->  shell splits into {n_orbits} orbit(s)")
        print(f"  nearest-neighbour distance: min {nn.min():.6f} max {nn.max():.6f}")
        print(f"  distinct NN distances: {np.unique(np.round(nn,6))}")
        print(f"  ({out['secs']}s)")
    return out


def main():
    import sys, os
    df = pd.read_csv(CSV)
    er = df[df["Type"] == "Erdos-Renyi"]
    wanted = [int(a) for a in sys.argv[1:]] or list(er.index)

    # results accumulate across chunked invocations
    results = []
    if os.path.exists("survey_results.json"):
        results = json.load(open("survey_results.json"))
    done = {r["row"] for r in results}

    for idx, row in er.iterrows():
        if idx not in wanted or idx in done:
            continue
        edges = ast.literal_eval(row["Edges"])
        n = int(row["Number of Nodes"])
        res = analyze_graph(idx, edges, n)
        if res:
            results.append(res)
        results.sort(key=lambda r: r["row"])
        with open("survey_results.json", "w") as f:
            json.dump(results, f, indent=1)

    print("\n" + "=" * 96)
    print("SUMMARY ACROSS ERDOS-RENYI GRAPHS")
    print("=" * 96)
    print(f"{'row':>3} {'m':>3} {'D':>3} {'cut':>4} {'Rmin/pi':>10} {'Sm^2':>5} "
          f"{'shell':>6} {'pi/4?':>7} {'nz':>6} {'zeros':>6} {'|Aut|':>6} {'orbits':>7} {'NN':>9}")
    for r in results:
        print(f"{r['row']:>3} {r['m']:>3} {r['D']:>3} {r['maxcut']:>4.0f} "
              f"{r['r_min_over_pi']:>10.6f} {r['r_min_units_pi4_squared']:>5.0f} "
              f"{r['n_shell']:>6} {r['snapped_ok']}/{r['n_shell']:<5} "
              f"{str(r['nonzero_counts']):>6} {r['n_always_zero']:>6} "
              f"{r['aut_order']:>6} {r['n_orbits']:>7} {r['nn_min']:>9.4f}")


if __name__ == "__main__":
    main()
