"""
Answer the question:
    "are all of the minima equally distributed about the higher dimensional
     sphere such that the nearest neighbor to each of these minima are always
     the same distance away?"

For each Erdos-Renyi graph we take the cached lowest-radius shell (produced by
lowest_norm_survey.py) and measure:

  A. For each shell point, compute its torus distance to the closest other point; 
     a perfectly symmetric shell would give identical distances for all points.

  B. Compute the pairwise torus distances between all shell points, and check that
     they are all integer multiples of (pi/4)^2. This is a necessary condition for
     the shell to be a discrete orbit of the symmetry group, and is a strong
     indicator of a highly symmetric shell.     

  C. Discrete or continuous?  Count the number of near-zero Hessian eigenvalues at
     a few shell points.  If any of them have a zero mode, the shell is
     continuous (a flat valley) rather than a discrete set of isolated points.

  D. Deterministic single target.  Canonicalize each shell point and take the
     lexicographically smallest.  This is a deterministic way to select a single
     representative point from the shell, which can be used as a target for
     downstream analysis.  Check that the target is at the same energy as the
     shell floor, and that the target is stable under disjoint-halves selection.
"""

import json
import glob
import numpy as np
import maqaoa_core as M

ZERO_EV = 1e-2 # eigenvalue magnitude below this counts as a flat direction
QUARTER = np.pi / 4

def hessian_from_grad(grad, x, h=1e-5):
    D = len(x)
    H = np.zeros((D, D))
    # Compute the Hessian by finite-difference of the gradient.  This is
    # expensive, but we only do it a few times to check for zero modes.
    for i in range(D):
        e = np.zeros(D)
        e[i] = h
        H[:, i] = (grad(x + e) - grad(x - e)) / (2 * h)
    return 0.5 * (H + H.T)      # symmetrize away tiny numerical asymmetry


def analyze(path):
    d = np.load(path)
    shell = d["shell"]
    edges = [tuple(e) for e in d["edges"]]
    floor = float(d["floor"])
    row = int(path.split("row")[1].split(".")[0])
    n = 1 + max(max(e) for e in edges)
    energy, _, grad, D = M.make_energy(n, edges, p=1)
    N = len(shell)

    # A. nearest-neighbor distances
    if N > 1:
        Dm = M.geodesic_dist_matrix(shell)
        np.fill_diagonal(Dm, np.inf)
        nn = Dm.min(1)
        nn_vals = np.unique(np.round(nn, 4))
        equidistant = len(nn_vals) == 1
        off = Dm[np.isfinite(Dm)]
    else:
        nn = np.array([np.nan]); nn_vals = np.array([]); equidistant = None; off = np.array([])

    # B. pairwise distances integer in (pi/4)^2 units?
    if len(off):
        sq_units = (off / QUARTER) ** 2
        int_resid = np.abs(sq_units - np.round(sq_units))
        max_resid = float(int_resid.max())
        distinct_sq = np.unique(np.round(sq_units, 2))
    else:
        max_resid = float("nan"); distinct_sq = np.array([])

    # C. discrete or continuous?
    # Sample a few shell points and count near-zero Hessian eigenvalues.
    idx = np.linspace(0, N - 1, min(4, N)).astype(int)
    zero_modes = []
    for k in idx:
        w = np.linalg.eigvalsh(hessian_from_grad(grad, shell[k]))
        zero_modes.append(int((np.abs(w) <= ZERO_EV).sum()))
    zm_med = int(np.median(zero_modes))
    kind = "continuum (flat valley)" if zm_med > 0 else "isolated points"

    # D. deterministic single target
    autos, images = M.symmetry_group(edges, p=1)

    def select(points):
        # Canonicalize each point and return the lexicographically smallest.
        cans = [M.canonicalize(p, images) for p in points]
        return sorted(cans, key=lambda y: tuple(np.round(y, 6)))[0]

    target = select(shell)
    # Stability check: does the rule return the same vector from disjoint halves?
    half_a = select(shell[0::2]) if N > 1 else target
    half_b = select(shell[1::2]) if N > 2 else target
    stable = (M.geodesic_dist(half_a, half_b) < 1e-6)
    tgt_ok = energy(target) <= floor + 1e-6
    tgt_r = M.norm_origin(target) / np.pi

    print(f"\n--- row {row}  (N={N} shell points, floor {floor}) ---")
    print(f"  structure: {kind}  (median zero Hessian modes = {zm_med})")
    if N > 1:
        print(f"  nearest-neighbour distances: {len(nn_vals)} distinct value(s) "
              f"-> {'EQUIDISTANT' if equidistant else 'NOT equidistant'}")
        print(f"    values: {nn_vals}")
        print(f"    in (pi/4)^2 units: {np.round((nn_vals/QUARTER)**2, 3)}")
        print(f"  pairwise squared distances integer in (pi/4)^2 units? "
              f"max residual = {max_resid:.2e}")
        print(f"    distinct squared values: {distinct_sq[:12]}"
              f"{' ...' if len(distinct_sq) > 12 else ''}")
    print(f"  deterministic target: at floor={tgt_ok}, radius={tgt_r:.6f} pi, "
          f"stable across disjoint halves={stable}")
    return dict(row=row, N=N, kind=kind, zero_modes=zm_med,
                equidistant=bool(equidistant) if equidistant is not None else None,
                nn_values=nn_vals.tolist(),
                nn_sq_units=np.round((nn_vals / QUARTER) ** 2, 3).tolist() if N > 1 else [],
                pair_int_residual=max_resid,
                target_at_floor=bool(tgt_ok), target_radius_over_pi=tgt_r,
                target_stable=bool(stable))


def main():
    out = []
    for path in sorted(glob.glob("shell_row*.npz"),
                       key=lambda s: int(s.split("row")[1].split(".")[0])):
        out.append(analyze(path))
    json.dump(out, open("uniformity_results.json", "w"), indent=1)

    print("\n" + "=" * 88)
    print("EQUIDISTRIBUTION SUMMARY")
    print("=" * 88)
    print(f"{'row':>3} {'N':>4} {'structure':>24} {'zero':>5} {'equidistant?':>13} {'#NN vals':>9}")
    for r in out:
        print(f"{r['row']:>3} {r['N']:>4} {r['kind']:>24} {r['zero_modes']:>5} "
              f"{str(r['equidistant']):>13} {len(r['nn_values']):>9}")

    disc = [r for r in out if r["zero_modes"] == 0]
    print(f"\nAmong the {len(disc)} graphs whose lowest shell is genuinely discrete:")
    for r in disc:
        print(f"  row {r['row']}: NN values in (pi/4)^2 units = {r['nn_sq_units']}")


if __name__ == "__main__":
    main()
