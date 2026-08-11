# Weighted-edge version of maqaoa_core.make_energy.
# Convention: C = sum_e w_e (1 - Z_u Z_v)/2, cost unitary exp(-i gamma_e w_e Z_u Z_v),
# so the weight multiplies the angle in the phase as well as the term in the expectation.

import numpy as np
import networkx as nx

# Build a statevector energy for a weighted graph.
def make_energy_weighted(n, edges, weights, p=1):
    edges = list(nx.Graph(edges).edges()) # canonical, de-duplicated edge order
    m = len(edges)
    w = np.asarray(weights, float)
    if len(w) != m:
        raise ValueError("need one weight per edge, got %d for %d edges" % (len(w), m))
    D = p * (m + n)
    dim = 1 << n # 2^n basis states

    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    spin = 1 - 2 * bits # +1 for bit 0, -1 for bit 1
    zz = np.stack([spin[:, i] * spin[:, j] for (i, j) in edges], 0)

    inv = 1.0 / np.sqrt(dim) # amplitude of the uniform |+>^n state
    zero_idx = [np.where(((np.arange(dim) >> j) & 1) == 0)[0] for j in range(n)]

    def _evolve(X):
        B = X.shape[0]
        gammas = X[:, :p * m].reshape(B, p, m)
        betas = X[:, p * m:].reshape(B, p, n)
        psi = np.full((B, dim), inv, dtype=complex)

        for layer in range(p):
            # cost layer, weight rides along with the angle
            phase = np.einsum('be,ek->bk', gammas[:, layer, :] * w, zz)
            psi *= np.exp(1j * phase)

            # mixer layer
            for j in range(n):
                b = betas[:, layer, j][:, None]
                c = np.cos(b)
                s = 1j * np.sin(b)
                i0 = zero_idx[j]
                i1 = i0 ^ (1 << j)
                a0 = psi[:, i0]
                a1 = psi[:, i1]
                psi[:, i0] = c * a0 + s * a1
                psi[:, i1] = s * a0 + c * a1

        return psi

    def energy_batch(X):
        X = np.atleast_2d(X)
        prob = np.abs(_evolve(X)) ** 2
        zsum = np.einsum('bk,ek->b', prob, zz * w[:, None])
        return 0.5 * zsum - w.sum() / 2.0

    def energy(x):
        return float(energy_batch(np.asarray(x, float)[None, :])[0])

    # Gradient by finite-difference of the energy.  This is expensive, but we only do it a few times to check for zero modes.
    def grad(x, h=1e-6):
        x = np.asarray(x, float)
        E = h * np.eye(D)
        vals = energy_batch(np.concatenate([x[None, :] + E, x[None, :] - E], 0))
        return (vals[:D] - vals[D:]) / (2.0 * h)

    return energy, energy_batch, grad, D

# Exact weighted MaxCut by enumerating all 2^n bitstrings.
def brute_weighted_maxcut(n, edges, weights):
    edges = list(nx.Graph(edges).edges())
    w = np.asarray(weights, float)
    best = 0.0
    for mask in range(1 << n):
        c = 0.0
        for k, (i, j) in enumerate(edges):
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                c += w[k]
        best = max(best, c)
    return -best
