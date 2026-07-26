# Reminder for me that peridicity is pi, not 2pi.

import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from scipy.optimize import minimize

# Build a fast statevector energy function for MA-QAOA on the given graph.
def make_energy(n, edges, p=1):
    edges = list(nx.Graph(edges).edges()) # canonical, de-duplicated edge order
    m = len(edges)
    D = p * (m + n)
    dim = 1 << n # 2^n basis states

    # bits[k, j] = value of qubit j in basis state k.
    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    # spin = +1 for bit 0, -1 for bit 1.  This is the Z eigenvalue.
    spin = 1 - 2 * bits
    # zz[e, k] = Z_i Z_j for edge e and basis state k.  This is the diagonal of the cost operator.
    zz = np.stack([spin[:, i] * spin[:, j] for (i, j) in edges], 0)

    inv = 1.0 / np.sqrt(dim) # amplitude of the uniform |+>^n state

    # For each qubit j, precompute the indices of basis states where the j-th bit is 0.
    zero_idx = [np.where(((np.arange(dim) >> j) & 1) == 0)[0] for j in range(n)]

    def _evolve(X):
        B = X.shape[0]
        gammas = X[:, :p * m].reshape(B, p, m)
        betas = X[:, p * m:].reshape(B, p, n)

        # Start every batch member in the uniform superposition |+>^n.
        psi = np.full((B, dim), inv, dtype=complex)

        for layer in range(p):
            # cost layer       
            phase = np.einsum('be,ek->bk', gammas[:, layer, :], zz)
            psi *= np.exp(1j * phase)

            # mixer layer
            for j in range(n):
                b = betas[:, layer, j][:, None]     # shape (B,1) -> broadcasts
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
        psi = _evolve(X)
        prob = np.abs(psi) ** 2  # measurement distribution
        # Expectation of the cost operator is sum_e <ZZ> = sum_e sum_k prob_k * zz[e, k]
        zsum = np.einsum('bk,ek->b', prob, zz)
        return 0.5 * zsum - m / 2.0

    def energy(x):
        return float(energy_batch(np.asarray(x, float)[None, :])[0])

    # Precompute the +-pi/4 shift pattern used by the parameter-shift gradient.
    shift = (np.pi / 4) * np.eye(D)

    # Compute the parameter-shift gradient by evaluating the energy at x +- pi/4 along each coordinate.
    def grad(x):
        x = np.asarray(x, float)
        plus = x[None, :] + shift
        minus = x[None, :] - shift
        vals = energy_batch(np.concatenate([plus, minus], 0))
        return vals[:D] - vals[D:]

    return energy, energy_batch, grad, D

# Verify the correctness of the fast QAOA implementation against the slow, exact one.
def verify_against_kron(n, edges, p=1, trials=3, seed=0):
    edges = list(nx.Graph(edges).edges())
    m = len(edges)
    dim = 1 << n
    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    # Create a tensor product of the given matrices on the specified qubits.
    def op_on(qubits, mats):
        out = np.array([[1.0 + 0j]])
        for q in reversed(range(n)):
            out = np.kron(out, mats[qubits.index(q)] if q in qubits else I2)
        return out

    ZZ = [op_on([i, j], [Z, Z]) for (i, j) in edges]
    XX = [op_on([j], [X]) for j in range(n)]
    C = sum(ZZ)  # cost operator sum_e Z_iZ_j

    energy, _, _, D = make_energy(n, edges, p)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        x = rng.uniform(0, np.pi, D)
        gammas = x[:p * m].reshape(p, m)
        betas = x[p * m:].reshape(p, n)

        psi = np.full(dim, 1 / np.sqrt(dim), dtype=complex)
        for layer in range(p):
            for e in range(m):
                # expm of a matrix with eigenvalues +-1 in closed form:
                # exp(i g ZZ) = cos(g) I + i sin(g) ZZ
                psi = (np.cos(gammas[layer, e]) * psi
                       + 1j * np.sin(gammas[layer, e]) * (ZZ[e] @ psi))
            for j in range(n):
                psi = (np.cos(betas[layer, j]) * psi
                       + 1j * np.sin(betas[layer, j]) * (XX[j] @ psi))
        slow = 0.5 * np.real(np.conj(psi) @ (C @ psi)) - m / 2.0
        worst = max(worst, abs(slow - energy(x)))
    return worst

# Check the gradient of the energy function.
def check_gradient(energy, grad, D, trials=3, seed=0, h=1e-5):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        x = rng.uniform(0, np.pi, D)
        g_exact = grad(x)
        for i in range(D):
            e = np.zeros(D)
            e[i] = h
            g_fd = (energy(x + e) - energy(x - e)) / (2 * h)
            worst = max(worst, abs(g_exact[i] - g_fd))
    return worst

# Check that shifting each coordinate by pi leaves the energy unchanged.
def check_pi_periodicity(energy, D, trials=5, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        x = rng.uniform(0, np.pi, D)
        base = energy(x)
        for i in range(D):
            e = np.zeros(D)
            e[i] = np.pi
            worst = max(worst, abs(energy(x + e) - base))
    return worst


# Wrap angles to the fundamental domain [0, pi).
def wrap_pi(x):
    return np.mod(x, np.pi)

# Compute the geodesic vector between two points on the torus.
def geodesic_vec(a, b):
    return np.mod(a - b + np.pi / 2, np.pi) - np.pi / 2

# Compute the geodesic distance between two points on the torus.
def geodesic_dist(a, b):
    v = geodesic_vec(a, b)
    return float(np.sqrt((v * v).sum()))

# Compute the matrix of all pairwise geodesic distances for a set of points.
def geodesic_dist_matrix(P):
    diff = np.mod(P[:, None, :] - P[None, :, :] + np.pi / 2, np.pi) - np.pi / 2
    return np.sqrt((diff ** 2).sum(-1))

# Compute the distance from a point to the origin on the torus.
def norm_origin(x):
    return float(np.linalg.norm(geodesic_vec(x, np.zeros(len(x)))))


# Build the symmetry group for a given graph via its automorphisms and the global sign flip.
def symmetry_group(edges, p=1):
    edge_order = list(nx.Graph(edges).edges())
    n = 1 + max(max(e) for e in edges)
    m = len(edge_order)
    eidx = {frozenset(e): i for i, e in enumerate(edge_order)}
    G = nx.Graph(edge_order)
    autos = list(GraphMatcher(G, G).isomorphisms_iter())

    # Build the induced action of a permutation on the parameter vector x.
    def induced(sig, x):
        gam = x[:p * m].reshape(p, m)
        bet = x[p * m:].reshape(p, n)
        gb = np.empty_like(gam)
        bb = np.empty_like(bet)
        for layer in range(p):
            for k in range(n):
                bb[layer, sig[k]] = bet[layer, k]
            for (u, v) in edge_order:
                gb[layer, eidx[frozenset((sig[u], sig[v]))]] = gam[layer, eidx[frozenset((u, v))]]
        return np.concatenate([gb.ravel(), bb.ravel()])

    # Compute all symmetry images of a point.
    def images(x): 
        out = []
        for sig in autos:
            y = induced(sig, x)
            out.append(wrap_pi(y))
            out.append(wrap_pi(-y))
        return out

    return autos, images

# Canonicalize a point by returning the lexicographically smallest of its symmetry images.
def canonicalize(x, images):
    cands = images(x)
    return sorted(cands, key=lambda y: tuple(np.round(y, 6)))[0]

# Partition a set of points into symmetry orbits.
def orbit_partition(reps, images, tol=1e-2):
    N = len(reps)
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(N):
        imgs = images(reps[i])
        for j in range(i + 1, N):
            if find(i) == find(j):
                continue
            if any(geodesic_dist(im, reps[j]) < tol for im in imgs):
                parent[find(i)] = find(j)
    roots = {}
    labels = np.empty(N, int)
    for i in range(N):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels

# Optimize a point using L-BFGS-B.
def polish(energy, grad, x0):
    return minimize(energy, x0, jac=grad, method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})

# Find the global minimum energy by brute-force random restarts.
def find_floor(energy, grad, D, restarts=200, seed=0):
    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(restarts):
        r = polish(energy, grad, rng.uniform(0, np.pi, D))
        best = min(best, r.fun)
    return round(float(best), 6)

# Find low-norm minima by a two-stage optimization process.         
def harvest_low_norm(energy, grad, D, floor, restarts=400, lam=0.5,
                     tol_e=1e-6, tol_d=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    
    # Stage 1: penalize the geodesic distance to the origin, then polish with the true energy.
    def pen(x):
        v = geodesic_vec(x, np.zeros(D))
        return energy(x) + lam * float(v @ v)

    # Stage 2: compute the gradient of the penalized energy.
    def pen_grad(x):
        v = geodesic_vec(x, np.zeros(D))
        return grad(x) + 2 * lam * v

    reps = []
    for _ in range(restarts):
        r = minimize(pen, rng.uniform(0, np.pi, D), jac=pen_grad,
                     method="L-BFGS-B", options={"ftol": 1e-13, "gtol": 1e-10})
        rp = polish(energy, grad, r.x)              # stage 2: back to true energy
        if rp.fun <= floor + tol_e:
            xw = wrap_pi(rp.x)
            if all(geodesic_dist(xw, q) > tol_d for q in reps):
                reps.append(xw)
    return np.array(reps) if reps else np.zeros((0, D))

# Snap a point to a grid.
def snap_to_grid(x, k=4):
    return wrap_pi(np.round(x / (np.pi / k)) * (np.pi / k))

# Express a point in terms of grid multiples.
def grid_multiples(x, k=4):
    return np.round(wrap_pi(x) / (np.pi / k)).astype(int) % k
