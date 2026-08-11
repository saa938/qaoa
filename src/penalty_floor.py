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

restarts_harvest = 1600 # number of random starts to harvest reference minima
tol_e_harvest = 1e-6 # energy tolerance for harvesting reference minima
tol_d_harvest = 1e-2 # torus distance below which two minima are considered the same for harvesting

# Harvest the reference minima.
print("\nHarvesting reference minima")
t0 = time.time()
reps = []
for i in range(restarts_harvest):
    r = minimize(energy, np.random.uniform(0, np.pi, D), method="L-BFGS-B",
                 options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000})
    if r.fun <= floor + tol_e_harvest:
        xw = L.wrap_pi(r.x)
        if all(L.geodesic_dist(xw, q) > tol_d_harvest for q in reps):
            reps.append(xw)
t1 = time.time()
reps = np.array(reps)
print("harvest time", t1 - t0, "seconds")
print("harvested", len(reps), "minima")

# Compute the reference point and the minimum distance to it.
ref, _ = L.circular_mean(reps)
d_min = min(L.geodesic_dist(r, ref) for r in reps)
print("d_min =", d_min, "\n")

def dist2(x):
    d = L.geodesic_dist(x, ref)
    return d * d

def dist2_shifted(x):
    d = L.geodesic_dist(x, ref)
    d = max(0.0, d - d_min)
    return d * d

def run(name, obj, lam, restarts=100, tol_e=1e-4, tol_d=1e-2, seed=7):
    rng = np.random.default_rng(seed)
    reps_local = []
    hits = 0
    radii = []
    t0 = time.time()
    for i in range(restarts):
        r = minimize(lambda x: obj(x, lam), rng.uniform(0, np.pi, D),
                     method="L-BFGS-B", options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2000})
        rp = minimize(energy, r.x, method="L-BFGS-B",
                      options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000})
        if rp.fun <= floor + tol_e:
            hits += 1
            xw = L.wrap_pi(rp.x)
            radii.append(L.geodesic_dist(xw, ref))
            if all(L.geodesic_dist(xw, q) > tol_d for q in reps_local):
                reps_local.append(xw)
    t1 = time.time()
    mean_r = float(np.mean(radii)) if radii else float("nan")
    print(name, "lam=", lam, "hit=", hits / restarts, "distinct=", len(reps_local),
          "meanR=", mean_r, t1 - t0, "seconds")

run("baseline (no penalty)",         lambda x, l: energy(x),                          0.0)
run("E + l d^2",                     lambda x, l: energy(x)         + l * dist2(x),   0.5)
run("(E-fl) + l d^2",                lambda x, l: (energy(x) - floor) + l * dist2(x), 0.5)
run("E * exp(-l d^2)  [raw -11]",    lambda x, l: energy(x)         * np.exp(-l * dist2(x)), 0.05)
run("(E-fl)*exp(-l d^2) [floor 0]",  lambda x, l: (energy(x) - floor) * np.exp(-l * dist2(x)), 0.05)
run("(E-fl)/(1+l d^2)   [floor 0]",  lambda x, l: (energy(x) - floor) / (1 + l * dist2(x)), 0.5)
run("(E-fl)*(1+l d^2)   [floor 0]",  lambda x, l: (energy(x) - floor) * (1 + l * dist2(x)), 0.5)
run("(E-fl)*(l d^2)     [floor 0]",  lambda x, l: (energy(x) - floor) * (l * dist2(x)), 0.5)
run("E + l (d-d_min)^2",             lambda x, l: energy(x)         + l * dist2_shifted(x), 0.5)

print("\nDone.")