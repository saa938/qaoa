# Exact grid enumeration for weighted MA-QAOA at p=1 without visiting every grid point.
#
# At p=1 the expectation of edge (u,v) depends only on beta_u, beta_v, theta_uv and the
# theta of the edges sharing u or v.  check_locality() verifies both halves.  The energy
# is therefore a sum of small local terms, so the minimum over a grid is a dynamic
# program, not a sweep.  On a path the state carries three grid indices and the cost is
# K**5 per edge instead of K**(m+n) overall.
#
# Everything is in theta = w * gamma (see maqaoa_weighted.make_energy_theta), so the grid
# is the same pi/K grid whatever the weights are.  Pulled back to gamma the spacing on
# edge e is pi / (K w_e), which scales with the period.

import time
import numpy as np
import networkx as nx
import maqaoa_weighted as MW

GRID_K = 4                 # grid is {j * pi / K}; pi/4 is sufficient on every case tested
SEED = 20260918


# Per-edge <Z_u Z_v> for the whole graph at p=1, in theta coordinates.
def edge_expectations(n, edges, x):
    edges = list(nx.Graph(edges).edges())
    m = len(edges)
    dim = 1 << n
    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    spin = 1 - 2 * bits
    zz = np.stack([spin[:, i] * spin[:, j] for (i, j) in edges], 0).astype(float)
    psi = (1.0 / np.sqrt(dim)) * np.exp(1j * (x[:m][:, None] * zz).sum(0))
    for j in range(n):
        z = np.where(((np.arange(dim) >> j) & 1) == 0)[0]
        o = z ^ (1 << j)
        c, sf = np.cos(x[m + j]), 1j * np.sin(x[m + j])
        a0, a1 = psi[z], psi[o]
        psi[z] = c * a0 + sf * a1
        psi[o] = sf * a0 + c * a1
    return zz @ (np.abs(psi) ** 2)


# Confirm that an edge expectation only moves when a neighbouring angle moves, and that
# summing the expectations reproduces maqaoa_weighted's energy.
def check_locality(n, edges, w, trials=2, h=1e-5, seed=SEED):
    rng = np.random.default_rng(seed)
    el = list(nx.Graph(edges).edges())
    m = len(el)
    energy, _, _ = MW.make_energy_theta(n, edges, w, p=1)
    far = near = agree = 0.0
    for _ in range(trials):
        x = rng.uniform(0, np.pi, m + n)
        ev = edge_expectations(n, edges, x)
        agree = max(agree, abs(0.5 * float(ev @ np.asarray(w, float))
                               - np.sum(w) / 2.0 - energy(x)))
        for a, (u, v) in enumerate(el):
            for b in range(m + n):
                xp, xm = x.copy(), x.copy()
                xp[b] += h
                xm[b] -= h
                d = abs((edge_expectations(n, edges, xp)[a]
                         - edge_expectations(n, edges, xm)[a]) / (2 * h))
                touched = ({el[b][0], el[b][1]} & {u, v}) if b < m else ({b - m} & {u, v})
                if touched:
                    near = max(near, d)
                else:
                    far = max(far, d)
    return dict(non_local=far, local=near, vs_energy=agree)


def path_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


# Table of the edge-k term over (theta_{k-1}, theta_k, theta_{k+1}, beta_k, beta_{k+1}),
# built on the light cone of edge k rather than on the whole graph.
def edge_term_table(w, k, m, K):
    vals = np.arange(K) * (np.pi / K)
    has_l, has_r = k - 1 >= 0, k + 1 <= m - 1
    nl, nr = (K if has_l else 1), (K if has_r else 1)
    verts = ([k - 1] if has_l else []) + [k, k + 1] + ([k + 2] if has_r else [])
    idx = {v: i for i, v in enumerate(verts)}
    nq = len(verts)
    dim = 1 << nq
    bits = ((np.arange(dim)[:, None] >> np.arange(nq)[None, :]) & 1)
    spin = 1 - 2 * bits
    pairs = ([(k - 1, k)] if has_l else []) + [(k, k + 1)] + \
            ([(k + 1, k + 2)] if has_r else [])
    zz = np.stack([spin[:, idx[a]] * spin[:, idx[b]] for (a, b) in pairs], 0).astype(float)
    target = (spin[:, idx[k]] * spin[:, idx[k + 1]]).astype(float)
    g = np.meshgrid(np.arange(nl), np.arange(K), np.arange(nr),
                    np.arange(K), np.arange(K), indexing="ij")
    shape = g[0].shape
    fl, fc, fr, fb0, fb1 = [a.ravel() for a in g]
    cols = ([vals[fl]] if has_l else []) + [vals[fc]] + ([vals[fr]] if has_r else [])
    psi = (1.0 / np.sqrt(dim)) * np.exp(1j * (np.stack(cols, axis=1) @ zz))
    for q, fb in ((idx[k], fb0), (idx[k + 1], fb1)):
        z = np.where(((np.arange(dim) >> q) & 1) == 0)[0]
        o = z ^ (1 << q)
        c = np.cos(vals[fb])[:, None]
        sf = 1j * np.sin(vals[fb])[:, None]
        a0, a1 = psi[:, z], psi[:, o]
        psi[:, z] = c * a0 + sf * a1
        psi[:, o] = sf * a0 + c * a1
    return (0.5 * w[k] * ((np.abs(psi) ** 2) @ target)).reshape(shape)


# Minimum of the weighted p=1 energy over the pi/K grid on a path, by dynamic programming.
def path_grid_min(n, w, K=GRID_K):
    m = n - 1
    vals = np.arange(K) * (np.pi / K)
    tables = [edge_term_table(w, k, m, K) for k in range(m)]
    dp = tables[0][0].min(axis=2)
    backs = [tables[0][0].argmin(axis=2)]
    for k in range(1, m):
        total = np.transpose(dp[:, :, None, :, None] + tables[k], (1, 2, 4, 0, 3))
        s = total.shape
        flat = total.reshape(s[0], s[1], s[2], s[3] * s[4])
        dp = flat.min(axis=3)
        backs.append(flat.argmin(axis=3).reshape(s[0], s[1], s[2]))
    best = float(dp.min())
    tk, tk1, bk1 = np.unravel_index(int(dp.argmin()), dp.shape)
    theta, beta = [0] * m, [0] * n
    theta[m - 1], beta[n - 1] = tk, bk1
    for k in range(m - 1, 0, -1):
        tprev, bk = divmod(int(backs[k][tk, tk1, bk1]), tables[k].shape[3])
        theta[k - 1], beta[k] = tprev, bk
        if k - 1 >= 1:
            tk, tk1, bk1 = tprev, tk, bk
        else:
            beta[0] = int(backs[0][tprev, tk, bk])
    if m == 1:
        beta[0] = int(backs[0][tk, tk1, bk1])
    x = np.concatenate([vals[np.array(theta)], vals[np.array(beta)]])
    return best - float(np.sum(w)) / 2.0, x, 2 * (K ** 5) * m


# Same minimum by visiting all K**(m+n) grid points.  Only for validation.
def brute_grid_min(n, edges, w, K=GRID_K, chunk=250000):
    m = len(list(nx.Graph(edges).edges()))
    D = m + n
    _, energy_batch, _ = MW.make_energy_theta(n, edges, w, p=1)
    vals = np.arange(K) * (np.pi / K)
    total = K ** D
    best, bestx = np.inf, None
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total))
        buf = vals[np.stack(np.unravel_index(idx, (K,) * D), axis=1)]
        e = energy_batch(buf)
        j = int(np.argmin(e))
        if e[j] < best:
            best, bestx = float(e[j]), buf[j].copy()
    return best, bestx, total


def draw_weights(model, m, seed):
    rng = np.random.default_rng(seed)
    if model == "unit":
        return np.ones(m)
    if model == "u_half":
        return rng.uniform(0.5, 1.5, m)
    if model == "u01":
        return rng.uniform(0.1, 1.0, m)
    if model == "int":
        return rng.integers(1, 5, m).astype(float)
    raise ValueError(model)


def main():
    n, edges = 6, [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
    w = draw_weights("u_half", 8, 11)
    loc = check_locality(n, edges, w)
    print("locality  non-local derivative {non_local:.2e}  local {local:.2e}  "
          "edge sum vs energy {vs_energy:.2e}".format(**loc))

    print("\n{:<8}{:>3}{:>4}{:>15}{:>15}{:>14}{:>12}{:>9}".format(
        "weights", "n", "K", "dp_value", "brute_value", "brute_points", "dp_points", "dp_s"))
    for n in (4, 5, 6):
        for model in ("unit", "u_half", "u01", "int"):
            w = draw_weights(model, n - 1, 1000 + n)
            for K in (4, 8):
                t0 = time.time()
                dpv, dpx, dppts = path_grid_min(n, w, K)
                t_dp = time.time() - t0
                energy, _, _ = MW.make_energy_theta(n, path_edges(n), w, p=1)
                assert abs(dpv - energy(dpx)) < 1e-9, "reconstructed point disagrees"
                if K ** (2 * n - 1) > 3.0e7:
                    print("{:<8}{:>3}{:>4}{:>15.8f}{:>15}{:>14.3e}{:>12}{:>9.2f}".format(
                        model, n, K, dpv, "-", float(K) ** (2 * n - 1), dppts, t_dp))
                    continue
                bv, bx, btot = brute_grid_min(n, path_edges(n), w, K)
                assert abs(dpv - bv) < 1e-9, (dpv, bv)
                print("{:<8}{:>3}{:>4}{:>15.8f}{:>15.8f}{:>14.3e}{:>12}{:>9.2f}".format(
                    model, n, K, dpv, bv, float(btot), dppts, t_dp))

    print("\n{:<4}{:>4}{:>16}{:>16}{:>14}{:>12}{:>9}".format(
        "n", "D", "dp_grid_min", "check_energy", "brute_points", "dp_points", "dp_s"))
    rng = np.random.default_rng(SEED)
    for n in (8, 12, 16, 20):
        w = rng.uniform(0.1, 1.0, n - 1)
        t0 = time.time()
        v, x, pts = path_grid_min(n, w, 8)
        energy, _, _ = MW.make_energy_theta(n, path_edges(n), w, p=1)
        print("{:<4}{:>4}{:>16.8f}{:>16.8f}{:>14.3e}{:>12}{:>9.2f}".format(
            n, 2 * n - 1, v, energy(x), 8.0 ** (2 * n - 1), pts, time.time() - t0))


if __name__ == "__main__":
    main()