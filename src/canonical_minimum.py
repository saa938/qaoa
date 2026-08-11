import numpy as np
from scipy.optimize import minimize
import landscape as L
import time

# Define the graph and the energy function.
edges = [(0,1),(0,2),(0,4),(1,5),(1,7),(2,5),(2,7),(3,4),
         (3,6),(3,7),(4,6),(4,7),(5,6),(5,7),(6,7)]
n, m = 8, 15
D = m + n
energy = L.make_energy(n, edges)
floor  = -11.0
autos, images = L.symmetry_group(edges)

# Compute the squared geodesic distance to the reference point.
def d2_origin(x):
    g = L.geodesic_vec(x, np.zeros(D))
    return float(g @ g)

# Compute the norm of the geodesic vector to the reference point.
def norm_origin(x):
    return np.linalg.norm(L.geodesic_vec(x, np.zeros(D)))

# Snap every angle to the nearest multiple of pi/4 (mod pi).
def snap_quarter_pi(x):
    return L.wrap_pi(np.round(x / (np.pi / 4)) * (np.pi / 4))

restarts_low_norm = 300 # number of random starts to search for low-norm minima
lam = 0.5 # penalty weight for the squared distance to the origin in the low-norm search
rng = np.random.default_rng(123) # random number generator

print("\nSearching for low-norm minima")
t0 = time.time()
low_norm = []
for i in range(restarts_low_norm):
    r = minimize(lambda x: energy(x) + lam * d2_origin(x), rng.uniform(0, np.pi, D),
                 method="L-BFGS-B", options={"ftol": 1e-12, "gtol": 1e-9})
    rp = minimize(energy, r.x, method="L-BFGS-B",
                  options={"ftol": 1e-14, "gtol": 1e-10}).x
    if energy(rp) <= floor + 1e-4:
        low_norm.append(L.wrap_pi(rp))
t1 = time.time()
print("search time", t1 - t0, "seconds")
print("found", len(low_norm), "low-norm minima")

snapped = [snap_quarter_pi(x) for x in low_norm if energy(snap_quarter_pi(x)) <= floor + 1e-6]
print("snapped to", len(snapped), "quarter-pi minima\n")

# merge by true symmetry orbit (automorphisms x sign), via images()
reps = []
for c in snapped:
    is_new = not any(any(L.geodesic_dist(im, r) < 1e-2 for im in images(c)) for r in reps)
    if is_new:
        reps.append(c)

# deterministic tie-break: lexicographically smallest canonical form
canon = sorted((L.canonicalize(r, images) for r in reps),
               key=lambda y: tuple(np.round(y, 6)))

print("distinct canonical min-norm targets:", len(canon))
for i, t in enumerate(canon):
    gw = L.geodesic_vec(t, np.zeros(D)) / np.pi
    nz = int((np.abs(gw) > 1e-6).sum())
    all_quarter = bool(np.all(np.isin(np.round(gw * 4).astype(int), [-1, 0, 1])))
    print("target", i, "norm/pi=", norm_origin(t) / np.pi, "#nonzero=", nz,
          "all coords in {0,+-1/4}?", all_quarter)
    print("            coords/pi =", np.round(gw, 2))

print("\nsingle deterministic training target = canon[0] (lexicographic tie-break).")