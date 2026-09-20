# Verify that the weighted MA-QAOA grid scales with the period.
#
# The cost unitary on edge e is exp(i * gamma_e * w_e * Z_u Z_v), so shifting gamma_e by
# pi / w_e is a full period of the energy, and shifting by any other amount (in particular
# the unweighted period pi) is generally not. In theta = w * gamma coordinates the period
# is pi for every edge regardless of weight, so a uniform pi/K grid in theta pulls back to
# a pi/(K w_e) grid in gamma -- the same K points per edge, just spaced to match that
# edge's own period. This checks both halves directly: the period claim on the raw energy,
# and the grid-spacing claim on the pulled-back points. No enumeration or search here.

import numpy as np
import maqaoa_weighted as MW

SEED = 20260918


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


# Shift gamma_e by its own period (pi / w_e) and separately by the unweighted period
# (pi). The energy should not move at all under the first and should generally move
# under the second, unless w_e happens to be 1.
def check_period(n, edges, w, p=1, trials=3, seed=SEED):
    rng = np.random.default_rng(seed)
    m = len(edges)
    energy, _, _, D = MW.make_energy_weighted(n, edges, w, p=p)
    worst_own_period, worst_unit_period = 0.0, 0.0
    for _ in range(trials):
        x = rng.uniform(0, np.pi, D)
        base = energy(x)
        for layer in range(p):
            for e in range(m):
                idx = layer * m + e
                xp = x.copy()
                xp[idx] += np.pi / w[e]
                worst_own_period = max(worst_own_period, abs(energy(xp) - base))
                xu = x.copy()
                xu[idx] += np.pi
                worst_unit_period = max(worst_unit_period, abs(energy(xu) - base))
    return worst_own_period, worst_unit_period


# Build the theta grid {0, pi/K, ..., (K-1) pi/K} and pull it back through gamma_e =
# theta_e / w_e. Check the spacing between consecutive pulled-back points on edge e is
# exactly pi / (K w_e), and that evaluating a point in theta matches evaluating the same
# point pulled back to gamma -- they are supposed to be the same physical point.
def check_grid_spacing(n, edges, w, K=4, p=1, seed=SEED):
    rng = np.random.default_rng(seed)
    m = len(edges)
    energy_theta, _, D = MW.make_energy_theta(n, edges, w, p=p)
    energy_gamma, _, _, _ = MW.make_energy_weighted(n, edges, w, p=p)
    theta_vals = np.arange(K) * (np.pi / K)

    worst_spacing_err = 0.0
    for e in range(m):
        gamma_vals = theta_vals / w[e]
        expected = np.pi / (K * w[e])
        got = gamma_vals[1] - gamma_vals[0]
        worst_spacing_err = max(worst_spacing_err, abs(got - expected))

    # cross-check: a random point on the theta grid, evaluated via energy_theta, must
    # match the same point pulled back to gamma and evaluated via the raw energy.
    x_theta = theta_vals[rng.integers(0, K, D)]
    x_gamma = x_theta.copy()
    for layer in range(p):
        for e in range(m):
            x_gamma[layer * m + e] /= w[e]
    worst_consistency_err = abs(energy_theta(x_theta) - energy_gamma(x_gamma))

    return worst_spacing_err, worst_consistency_err


def main():
    n, edges = 6, [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]

    print("Does shifting gamma_e by its own period leave the energy unchanged?\n")
    print("{:<8}{:>3}{:>20}{:>20}".format(
        "weights", "p", "shift by pi/w_e", "shift by pi"))
    for model in ("unit", "u_half", "u01", "int"):
        w = draw_weights(model, len(edges), 11)
        for p in (1, 2, 3):
            own, unit = check_period(n, edges, w, p=p)
            print("{:<8}{:>3}{:>20.3e}{:>20.3e}".format(model, p, own, unit))

    print("\nDoes the theta grid pull back to spacing pi/(K w_e) in gamma?\n")
    print("{:<8}{:>3}{:>4}{:>22}{:>22}".format(
        "weights", "p", "K", "spacing error", "theta/gamma mismatch"))
    for model in ("unit", "u_half", "u01", "int"):
        w = draw_weights(model, len(edges), 11)
        for p in (1, 2):
            for K in (4, 8):
                sp_err, cons_err = check_grid_spacing(n, edges, w, K=K, p=p)
                print("{:<8}{:>3}{:>4}{:>22.3e}{:>22.3e}".format(
                    model, p, K, sp_err, cons_err))


if __name__ == "__main__":
    main()