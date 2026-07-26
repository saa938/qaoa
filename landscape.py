# This one has different formatting because I did it on a different computer

import numpy as np
import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import expm
from networkx.algorithms.isomorphism import GraphMatcher
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

# Compute the cost Hamiltonian for a given graph.
def cost_hamiltonian(n, G):
  CostSTRVec = []
  coeffsvec = []
  for u, v in G.edges():
    str_now = ''
    for j in range(n):
      if j == u or j == v:
        str_now = 'Z' + str_now
      else:
        str_now = 'I' + str_now
    CostSTRVec.append(str_now)
    coeffsvec.append(1.0)
  return SparsePauliOp(CostSTRVec, coeffs=coeffsvec)

# Create a QAOA circuit for a given graph and parameters.
def qaoa_circuit_ma(n, qaoa_layers, gammas_ma, betas_ma, G, measure=True):
  qc = QuantumCircuit(n)
  qc.h(range(n))
  for i in range(qaoa_layers):
    for idx, (j, k) in enumerate(G.edges()):
      qc.rzz(-2 * gammas_ma[i, idx], j, k)
    for j in range(n):
      qc.rx(-2 * betas_ma[i, j], j)
  if measure:
    qc.measure_all()
  return qc

# Compute the expectation value of the cost Hamiltonian for a given QAOA circuit.
def expectation_ma(params, n, qaoa_layers, G, HC):
  m = len(list(G.edges()))
  gammas_ma = params[:qaoa_layers * m].reshape(qaoa_layers, m)
  betas_ma = params[qaoa_layers * m:].reshape(qaoa_layers, n)
  qc = qaoa_circuit_ma(n, qaoa_layers, gammas_ma, betas_ma, G, measure=False)
  state_vector = Statevector(qc)
  HC_val = state_vector.expectation_value(HC).real
  W = len(list(G.edges()))
  return -((W - HC_val) / 2.0)

# Create an energy function for a given graph.
def make_energy(n, edges):
  edges = list(nx.Graph(edges).edges())
  m = len(edges)
  dim = 1 << n
  bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
  spin = 1 - 2 * bits
  zz = np.stack([spin[:, i] * spin[:, j] for (i, j) in edges], 0)
  inv = 1.0 / np.sqrt(dim)
  zero_idx = [np.where(((np.arange(dim) >> j) & 1) == 0)[0] for j in range(n)]
  def energy(x):
    gammas = x[:m]
    betas = x[m:]
    psi = (inv * np.exp(1j * (gammas[:, None] * zz).sum(0))).astype(complex)
    for j in range(n):
      c = np.cos(betas[j])
      sf = 1j * np.sin(betas[j])
      a0 = psi[zero_idx[j]]
      a1 = psi[zero_idx[j] ^ (1 << j)]
      psi[zero_idx[j]] = c * a0 + sf * a1
      psi[zero_idx[j] ^ (1 << j)] = sf * a0 + c * a1
    prob = np.abs(psi) ** 2
    return 0.5 * (zz * prob[None, :]).sum() - m / 2.0
  return energy

# Verify the correctness of the energy function.
def verify_engine(n, edges, trials=4):
  G = nx.Graph(edges)
  m = len(edges)
  HC = cost_hamiltonian(n, G)
  energy = make_energy(n, edges)
  worst = 0.0
  for _ in range(trials):
    x = np.random.uniform(0, np.pi, m + n)
    fast = energy(x)
    qk = expectation_ma(x, n, 1, G, HC)
    worst = max(worst, abs(fast - qk))
  return worst

# Wrap a value to the interval [0, pi).
def wrap_pi(x):
  return np.mod(x, np.pi)

# Compute the geodesic distance between two points on the circle.
def geodesic_dist(a, b):
  d = np.abs(wrap_pi(a) - wrap_pi(b))
  d = np.minimum(d, np.pi - d)
  return np.sqrt((d * d).sum())

# Compute the geodesic vector from point b to point a on the circle.
def geodesic_vec(a, b):
  return np.mod(a - b + np.pi / 2, np.pi) - np.pi / 2

# Find the floor of the energy function.
def find_floor(energy, D, restarts=100):
  best = min(minimize(energy, np.random.uniform(0, np.pi, D), method="L-BFGS-B").fun
             for _ in range(restarts))
  return round(best, 6)

# Harvest the minima of the energy function.
def harvest_minima(energy, D, floor, restarts=1500, tol_e=1e-6, tol_d=1e-2):
  reps = []
  curve = []
  for t in range(restarts):
    r = minimize(energy, np.random.uniform(0, np.pi, D), method="L-BFGS-B",
                 options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000})
    if r.fun <= floor + tol_e:
      xw = wrap_pi(r.x)
      if all(geodesic_dist(xw, q) > tol_d for q in reps):
        reps.append(xw)
    if (t + 1) % 500 == 0:
      curve.append((t + 1, len(reps)))
  return np.array(reps), curve

# Compute the Hessian matrix of the energy function at a given point.
def hessian(energy, x, h=1e-4):
  D = len(x)
  H = np.zeros((D, D))
  e = np.eye(D)
  for i in range(D):
    for j in range(i, D):
      fpp = energy(x + h * e[i] + h * e[j])
      fpm = energy(x + h * e[i] - h * e[j])
      fmp = energy(x - h * e[i] + h * e[j])
      fmm = energy(x - h * e[i] - h * e[j])
      H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h * h)
  return H

# Analyze the eigenvalues of the Hessian matrix at a set of points.
def eigen_census(energy, reps, sample=200, ev_zero=1e-2):
  idx = np.random.choice(len(reps), min(sample, len(reps)), replace=False)
  spectra = np.array([np.linalg.eigvalsh(hessian(energy, reps[k])) for k in idx])
  smallest = spectra[:, 0]
  troughs = int((np.abs(smallest) <= ev_zero).sum())
  bowls = len(smallest) - troughs
  return spectra, bowls, troughs

# Test the nature of the minima by examining the behavior of the energy function along the directions of its smallest eigenvalues.
def trough_bowl_test(energy, reps, spectra, sample_idx, floor):
  smallest = spectra[:, 0]
  trough = reps[sample_idx[np.argmin(smallest)]]
  bowl = reps[sample_idx[np.argmax(smallest)]]
  wt, Vt = np.linalg.eigh(hessian(energy, trough))
  wb, Vb = np.linalg.eigh(hessian(energy, bowl))
  ts = np.linspace(0, 1.2, 25)
  walk_t = np.array([energy(trough + t * Vt[:, 0]) - floor for t in ts])
  walk_b = np.array([energy(bowl + t * Vb[:, 0]) - floor for t in ts])
  return ts, walk_t, walk_b

# Compute the distance from a point to the nearest point on a grid of size k.
def circular_mean(reps):
  two = 2 * reps
  C = np.cos(two).mean(0)
  S = np.sin(two).mean(0)
  ref = wrap_pi(0.5 * np.arctan2(S, C))
  R = np.sqrt(C ** 2 + S ** 2)
  return ref, R

# Compute the distance from a point to the nearest point on a grid of size k.
def make_penalty(energy, ref, d_min, combo, power, shifted):
  def dist(x):
    d = geodesic_dist(x, ref)
    if shifted:
      d = max(0.0, d - d_min)
    return d ** power
  if combo == 'add':
    return lambda x, lam: energy(x) + lam * dist(x)
  if combo == 'mult_exp':
    return lambda x, lam: energy(x) * np.exp(-lam * dist(x))
  if combo == 'mult_inv':
    return lambda x, lam: energy(x) / (1.0 + lam * dist(x))

# Run the penalty method.
def run_penalty(energy, floor, D, obj, lam, restarts=200, tol_e=1e-4, tol_d=1e-2):
  reps = []
  hits = 0
  for _ in range(restarts):
    r = minimize(lambda x: obj(x, lam), np.random.uniform(0, np.pi, D),
                 method="L-BFGS-B", options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2000})
    rp = minimize(energy, r.x, method="L-BFGS-B",
                  options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000})
    if rp.fun <= floor + tol_e:
      hits += 1
      xw = wrap_pi(rp.x)
      if all(geodesic_dist(xw, q) > tol_d for q in reps):
        reps.append(xw)
  return hits / restarts, len(reps)

# Find the symmetry group of a graph.
def symmetry_group(edges):
  edge_order = list(nx.Graph(edges).edges())
  n = 1 + max(max(e) for e in edges)
  m = len(edge_order)
  eidx = {frozenset(e): i for i, e in enumerate(edge_order)}
  G = nx.Graph(edge_order)
  autos = list(GraphMatcher(G, G).isomorphisms_iter())
  def induced(sig, x):
    gammas = x[:m]
    betas = x[m:]
    gb = np.empty(m)
    bb = np.empty(n)
    for k in range(n):
      bb[sig[k]] = betas[k]
    for (u, v) in edge_order:
      gb[eidx[frozenset((sig[u], sig[v]))]] = gammas[eidx[frozenset((u, v))]]
    return np.concatenate([gb, bb])
  def images(x):
    out = []
    for sig in autos:
      y = induced(sig, x)
      out.append(wrap_pi(y))
      out.append(wrap_pi(-y))
    return out
  return autos, images

# Compute the canonical form of a point under the symmetry group of a graph.
def canonicalize(x, images):
  cands = images(x)
  keyed = sorted(cands, key=lambda y: tuple(np.round(y, 6)))
  return keyed[0]

# Count the number of distinct orbits of a set of points under the symmetry group.
def orbit_count(reps, images, tol_d=1e-2):
  parent = list(range(len(reps)))
  def find(a):
    while parent[a] != a:
      parent[a] = parent[parent[a]]
      a = parent[a]
    return a
  for i in range(len(reps)):
    imgs = images(reps[i])
    for j in range(i + 1, len(reps)):
      if any(geodesic_dist(im, reps[j]) < tol_d for im in imgs):
        parent[find(i)] = find(j)
  return len({find(i) for i in range(len(reps))})

# Project the error of a GNN onto the soft directions of the energy landscape.
def gnn_error_projection(csv_path, n=1, qaoa_layers=1, ev_soft=1.0):
  df = pd.read_csv(csv_path)
  results = []
  for _, row in df.iterrows():
    edges = ast.literal_eval(row['Edges'])
    gammas = np.array(ast.literal_eval(row['Predicted Gammas']))
    betas = np.array(ast.literal_eval(row['Predicted Betas']))
    nn = 1 + max(max(e) for e in edges)
    m = len(edges)
    energy = make_energy(nn, edges)
    floor = find_floor(energy, m + nn, restarts=60)
    pred = np.concatenate([gammas.reshape(m), betas.reshape(nn)])
    endpoint = minimize(energy, pred, method="L-BFGS-B",
                        options={"ftol": 1e-14, "gtol": 1e-10}).x
    at_floor = endpoint if minimize(energy, pred).fun <= floor + 1e-3 else None
    err = geodesic_vec(pred, endpoint)
    w, V = np.linalg.eigh(hessian(energy, endpoint))
    coeff = V.T @ err
    cost_gnn = 0.5 * np.sum(w * coeff ** 2)
    cost_rand = 0.5 * np.mean(w) * np.sum(err ** 2)
    frac_soft = np.sum(coeff[w < ev_soft] ** 2) / np.sum(coeff ** 2)
    results.append(dict(err_norm=np.linalg.norm(err), cost_gnn=cost_gnn,
                        cost_rand=cost_rand, ratio=cost_gnn / max(cost_rand, 1e-12),
                        frac_soft=frac_soft, reached_floor=(minimize(energy, pred).fun <= floor + 1e-3)))
  out = pd.DataFrame(results)
  print(out.describe())
  print("\nmean ratio gnn/random:", out['ratio'].mean())
  print("mean fraction of error in soft directions:", out['frac_soft'].mean())
  return out

# Create plots to visualize the results.
def make_plots(spectra, curve, ts, walk_t, walk_b, origin_d, cent_d, reps):
  plt.rcParams.update({'font.size': 11})
  plt.figure(figsize=(6, 4))
  plt.hist(spectra[:, 0], bins=40, color='steelblue', edgecolor='k', lw=.3)
  plt.xlabel("smallest Hessian eigenvalue")
  plt.ylabel("count")
  plt.title("Bimodal curvature: troughs (~0) vs bowls (~2)")
  plt.tight_layout()
  plt.savefig("p_bimodal.png")

  plt.figure(figsize=(6, 4))
  sr = np.array(curve)
  plt.plot(sr[:, 0], sr[:, 1], 'o-', color='darkgreen')
  plt.xlabel("random restarts")
  plt.ylabel("distinct points found")
  plt.title("Count never saturates -> continuum")
  plt.tight_layout()
  plt.savefig("p_saturation.png")

  plt.figure(figsize=(6, 4))
  plt.plot(ts, walk_t, 'o-', color='seagreen', label='trough')
  plt.plot(ts, walk_b, 's-', color='firebrick', label='bowl')
  plt.xlabel("distance along softest eigenvector")
  plt.ylabel("energy above floor")
  plt.legend()
  plt.title("Trough stays flat; bowl rises")
  plt.tight_layout()
  plt.savefig("p_walk.png")

  plt.figure(figsize=(6, 4))
  plt.hist(reps[:, 7], bins=30, range=(0, np.pi), color='goldenrod', edgecolor='k', lw=.3)
  plt.xlabel("value of one gamma coordinate")
  plt.ylabel("count")
  plt.title("One coordinate spread across the full [0,pi) period")
  plt.tight_layout()
  plt.savefig("p_uniform.png")

# Main function to run the analysis.
def main():
  edges = [(0, 1), (0, 2), (0, 4), (1, 5), (1, 7), (2, 5), (2, 7), (3, 4),
           (3, 6), (3, 7), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
  n = 8
  m = len(edges)
  D = m + n
  energy = make_energy(n, edges)

  print("engine vs qiskit:", verify_engine(n, edges))
  floor = find_floor(energy, D)
  print("floor:", floor)

  reps, curve = harvest_minima(energy, D, floor, restarts=1500)
  print("distinct minima:", len(reps), "saturation:", curve)

  idx = np.random.choice(len(reps), min(200, len(reps)), replace=False)
  spectra = np.array([np.linalg.eigvalsh(hessian(energy, reps[k])) for k in idx])
  troughs = int((np.abs(spectra[:, 0]) <= 1e-2).sum())
  print("bowls:", len(idx) - troughs, "troughs:", troughs)

  ts, walk_t, walk_b = trough_bowl_test(energy, reps, spectra, idx, floor)

  ref, R = circular_mean(reps)
  print("centroid concentration R median:", np.median(R))
  cent_d = np.array([geodesic_dist(r, ref) for r in reps])
  origin_d = np.array([np.linalg.norm(geodesic_vec(r, np.zeros(D))) for r in reps])
  d_min = cent_d.min()
  print("d_min:", d_min)

  autos, images = symmetry_group(edges)
  n_orbits = orbit_count(reps, images)
  print("distinct", len(reps), "-> orbits", n_orbits, "group order", 2 * len(autos))

  for combo in ['add', 'mult_exp']:
    obj = make_penalty(energy, ref, d_min, combo, 2, False)
    lam = 0.5 if combo == 'add' else 0.05
    hit, ndist = run_penalty(energy, floor, D, obj, lam)
    print(f"{combo} d^2 lam={lam}: hit {hit:.2f} distinct {ndist}")

  make_plots(spectra, curve, ts, walk_t, walk_b, origin_d, cent_d, reps)

  # gnn_error_projection("MaxCutMAQAOAData.csv")


if __name__ == "__main__":
  main()