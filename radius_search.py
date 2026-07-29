"""
Radius-constrained search for the MA-QAOA global minimum with the smallest
radius and the largest positive component magnitude.
"""

import time
import numpy as np
from scipy.optimize import minimize
import maqaoa_core as M

BIG_RADIUS = 1.0e6 # A large constant to represent a large radius
BIG_FRAC = 2.0e6 # A large constant to represent a large fraction
SLACK = 1.0e-3 # slack added to the radius cap so that the optimizer can reach the boundary without being rejected
TOL_E = 1.0e-6 # Tolerance for the energy
TOL_D = 1.0e-2 # Tolerance for the gradient

# Return the largest geodesic radius any point can have
def full_radius(D):
    return float(np.sqrt(D) * np.pi / 2.0)

# Signed displacement of x from the centre, each coordinate folded into [-pi/2, pi/2).
def displacement(x, center=None):
    x = np.asarray(x, float)
    c = np.zeros(len(x)) if center is None else np.asarray(center, float)
    return M.geodesic_vec(x, c)

# Geodesic radius of x measured from the centre.
def radius(x, center=None):
    return float(np.linalg.norm(displacement(x, center)))

# Radius built from the positive components of the displacement only.
def positive_radius(x, center=None):
    g = displacement(x, center)
    return float(np.linalg.norm(g[g > 0.0]))

# Fraction of the radius that comes from positive components.
def positive_fraction(x, center=None):
    r = radius(x, center)
    return positive_radius(x, center) / r if r > 1e-12 else 0.0

# Wrap an energy and gradient pair so that every call is counted.  
class Counter:
    def __init__(self, energy, grad):
        self._energy = energy
        self._grad = grad
        self.n_energy = 0
        self.n_grad = 0

    def energy(self, x):
        self.n_energy += 1
        return self._energy(x)

    def grad(self, x):
        self.n_grad += 1
        return self._grad(x)

    def reset(self):
        self.n_energy = 0
        self.n_grad = 0

    def total(self):
        return self.n_energy + self.n_grad

# The radius-capped cost function.  Inside the ball  are the true 
# minima.  Outside it is a large constant with zero gradient, which
# L-BFGS-B reads as a flat plateau and stops
def make_capped_energy(energy, grad, R, slack=SLACK, big=BIG_RADIUS, center=None):
    lim = R + slack

    def f(x):
        if radius(x, center) > lim:
            return big
        return energy(x)

    def fg(x):
        if radius(x, center) > lim:
            return np.zeros(len(x))
        return np.asarray(grad(x), float)

    return f, fg

# Multiplicative radius shaping so minima don't move
def make_shaped_energy(energy, grad, floor, R, shape, dshape, slack=SLACK, big=BIG_RADIUS, center=None, frac_min=None, frac_shape=None, big_frac=BIG_FRAC):
    lim = R + slack

    def f(x):
        r = radius(x, center)
        if r > lim:
            return big
        if frac_min is not None and positive_fraction(x, center) < frac_min:
            return big_frac
        val = (energy(x) - floor) * shape(r)
        if frac_shape is not None:
            val *= frac_shape(1.0 - positive_fraction(x, center))
        return val

    # This function was done by Claude
    # Gradient of (E - floor) * g(r) is g(r) * dE + (E - floor) * g'(r) * dr/dx,
    # with dr/dx the unit displacement vector.  The positive-fraction factor is
    # left out of the analytic gradient and handled by finite differences when it
    # is switched on, because the fraction is not differentiable where a component
    # changes sign.
    def fg(x):
        r = radius(x, center)
        if r > lim:
            return np.zeros(len(x))
        if frac_min is not None and positive_fraction(x, center) < frac_min:
            return np.zeros(len(x))
        g = displacement(x, center)
        dE = np.asarray(grad(x), float)
        base = shape(r) * dE
        if r > 1e-12:
            base = base + (energy(x) - floor) * dshape(r) * (g / r)
        if frac_shape is None:
            return base
        h = 1e-6
        out = np.empty(len(x))
        for i in range(len(x)):
            e = np.zeros(len(x))
            e[i] = h
            out[i] = (f(x + e) - f(x - e)) / (2.0 * h)
        return out

    return f, fg

# Shaping families.  Each returns (g, g') with g > 0 on r >= 0.  a is the strength.
def shape_family(name, a):
    if name == "const":
        return (lambda r: 1.0), (lambda r: 0.0)
    if name == "linear":
        return (lambda r: 1.0 + a * r), (lambda r: a)
    if name == "power":
        return (lambda r: (1.0 + r) ** a), (lambda r: a * (1.0 + r) ** (a - 1.0))
    if name == "exp":
        return (lambda r: np.exp(a * r)), (lambda r: a * np.exp(a * r))
    raise ValueError(name)

# Draw a starting point uniformly in the fundamental domain
def sample_start(rng, D, R, center=None, tries=200):
    for _ in range(tries):
        x = rng.uniform(0.0, np.pi, D)
        if radius(x, center) <= R:
            return x
    g = rng.normal(size=D)
    g = g / np.linalg.norm(g) * (R * rng.uniform() ** (1.0 / D))
    return M.wrap_pi((np.zeros(D) if center is None else np.asarray(center)) + g)

# One optimization run on the capped objective, followed by a run on the raw energy
def one_run(cnt, D, R, floor, x0, shaped=None, center=None, tol_e=TOL_E):
    if shaped is None:
        f, fg = make_capped_energy(cnt.energy, cnt.grad, R, center=center)
    else:
        f, fg = shaped
    r = minimize(f, x0, jac=fg, method="L-BFGS-B",
                 options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})
    rp = minimize(cnt.energy, r.x, jac=cnt.grad, method="L-BFGS-B",
                  options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 4000})
    xw = M.wrap_pi(rp.x)
    e = float(rp.fun)
    if e > floor + tol_e:
        return None
    if radius(xw, center) > R + 1e-6:
        return None
    return xw, e

# After each iteration the radius is the smallest among the accepted minima 
# Stops when an iteration cannot beat the current radius.
def shrink_search(energy, grad, D, floor, restarts=60, max_iters=12, seed=0,
                  center=None, tol_e=TOL_E, verbose=True, patience=2):
    cnt = Counter(energy, grad)
    rng = np.random.default_rng(seed)
    R = full_radius(D)
    best_x = None
    best_r = np.inf
    stalls = 0
    history = []
    for it in range(max_iters):
        t0 = time.time()
        e0 = cnt.total()
        hits = []
        for _ in range(restarts):
            x0 = sample_start(rng, D, R, center)
            got = one_run(cnt, D, R, floor, x0, center=center, tol_e=tol_e)
            if got is not None:
                hits.append(got[0])
        if not hits:
            history.append({"iter": it, "R_in": R, "n_hits": 0, "R_out": R,
                            "evals": cnt.total() - e0, "secs": round(time.time() - t0, 1)})
            break
        radii = np.array([radius(x, center) for x in hits])
        j = int(radii.argmin())
        R_out = float(radii[j])
        if R_out < best_r - 1e-9:
            best_r = R_out
            best_x = hits[j]
        history.append({"iter": it, "R_in": float(R), "n_hits": len(hits),
                        "R_out": R_out, "mean_r": float(radii.mean()),
                        "evals": cnt.total() - e0, "secs": round(time.time() - t0, 1)})
        if verbose:
            print("  iter %2d  R_in %8.4f  hits %3d/%d  R_out %8.4f  mean_r %7.4f  evals %7d"
                  % (it, R, len(hits), restarts, R_out, radii.mean(), cnt.total() - e0))
        if R_out >= R - 1e-6:
            stalls += 1
            if stalls > patience:
                break
        else:
            stalls = 0
        R = min(R, best_r)
    return {"x": best_x, "radius": best_r, "history": history, "evals": cnt.total()}

# Shrink the radius cap and raise the positive-fraction gate at the same time.
# Radius first, then fraction
def joint_search(energy, grad, D, floor, restarts=60, max_iters=10, seed=0,
                 center=None, tol_e=TOL_E, shape_name="exp", shape_a=1.5,
                 use_frac_gate=True, frac_a=None, patience=2, verbose=True):
    cnt = Counter(energy, grad)
    rng = np.random.default_rng(seed)
    R = full_radius(D)
    f_min = 0.0
    best_x = None
    best_r = np.inf
    best_f = -1.0
    stalls = 0
    history = []
    shape, dshape = shape_family(shape_name, shape_a)
    fs = None if frac_a is None else (lambda t: np.exp(frac_a * t))
    for it in range(max_iters):
        e0 = cnt.total()
        t0 = time.time()
        hits = []
        for _ in range(restarts):
            x0 = sample_start(rng, D, R, center)
            shaped = make_shaped_energy(cnt.energy, cnt.grad, floor, R, shape, dshape,
                                       center=center,
                                       frac_min=(f_min if use_frac_gate and f_min > 0 else None),
                                       frac_shape=fs)
            got = one_run(cnt, D, R, floor, x0, shaped=shaped, center=center, tol_e=tol_e)
            if got is not None:
                hits.append(got[0])
        improved = False
        if hits:
            radii = np.array([radius(x, center) for x in hits])
            fracs = np.array([positive_fraction(x, center) for x in hits])
            r_new = float(radii.min())
            at_r = radii < r_new + 1e-6
            f_new = float(fracs[at_r].max())
            if r_new < best_r - 1e-6:
                best_r, best_f = r_new, f_new
                best_x = hits[int(np.where(at_r)[0][int(fracs[at_r].argmax())])]
                f_min = 0.0
                improved = True
            elif abs(r_new - best_r) <= 1e-6 and f_new > best_f + 1e-6:
                best_f = f_new
                best_x = hits[int(np.where(at_r)[0][int(fracs[at_r].argmax())])]
                improved = True
        history.append({"iter": it, "R_in": float(R), "f_min": f_min,
                        "n_hits": len(hits), "best_r": best_r, "best_f": best_f,
                        "evals": cnt.total() - e0, "secs": round(time.time() - t0, 1)})
        if verbose:
            print("  iter %2d  R %8.4f  gate %.4f  hits %3d/%d  best_r %8.4f  best_f %.5f  evals %7d"
                  % (it, R, f_min, len(hits), restarts, best_r, best_f, cnt.total() - e0))
        stalls = 0 if improved else stalls + 1
        if stalls > patience:
            break
        R = min(R, best_r)
        if use_frac_gate and best_f > 0:
            f_min = max(0.0, best_f - 1e-6)
    return {"x": best_x, "radius": best_r, "frac": best_f,
            "history": history, "evals": cnt.total()}

# Deduplicate a list of points up to geodesic distance.
def dedupe(points, tol=TOL_D):
    out = []
    for x in points:
        if all(M.geodesic_dist(x, q) > tol for q in out):
            out.append(x)
    return out

# Close a set of minima under the graph symmetry group and the global sign flip.
def close_under_symmetry(points, images, tol=TOL_D):
    out = []
    for x in points:
        for y in images(x):
            if all(M.geodesic_dist(y, q) > tol for q in out):
                out.append(y)
    return np.array(out) if out else np.zeros((0, len(points[0])))