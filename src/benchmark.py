"""
benchmark.py
============
Global vs local optimisation for MA-QAOA Max-Cut, weighted or unweighted.

Compares scipy basinhopping against a plain L-BFGS-B restart loop across graph
size, edge density and layer count. Cost is measured as energy-function
evaluations spent per true global minimum obtained, so each method is charged
for its failures. Also sweeps the radius cap and the positive-fraction gate to
see how both methods degrade as the admissible region shrinks.

Weighted graphs are handled by one change of variable. The phase on edge e is
exp(i gamma_e w_e Z_u Z_v), so the period in gamma_e is pi / w_e. Optimising in
theta_e = w_e gamma_e puts every coordinate back on a period of pi, which is what
radius(), positive_fraction(), wrap_pi() and the basinhopping stepsize all assume.
Set WEIGHT_MODEL to None to reproduce the unweighted sweep exactly.

Place in src/ next to landscape.py. Stores and tables are written to
results/ at the repository root, so the working directory does not matter.
Press run; it sweeps, then gates, then prints the tables. Re-running resumes
from the stores, so an interrupted run loses nothing.
"""

import os
import json
import time
import tempfile
import csv
import hashlib
import ast
import numpy as np
import networkx as nx
from scipy.optimize import minimize, basinhopping
import landscape as L
import maqaoa_weighted as MW

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS      = os.path.join(ROOT, "results")
DATA_CSV     = os.path.join(ROOT, "data", "graphs_extra.csv")
STORE        = os.path.join(RESULTS, "bench_store_all.json")
GATE_STORE   = os.path.join(RESULTS, "bench_gate_all.json")
MAIN_CSV     = os.path.join(RESULTS, "benchmark_main.csv")
GATE_CSV     = os.path.join(RESULTS, "benchmark_gates.csv")

RUN_SWEEP    = True        # set False to re-print tables without computing
RUN_GATES    = True        # radius and positive-fraction sweep
VERIFY       = True        # check the multi-layer engine against Qiskit first
TIME_BUDGET  = 3600.0      # seconds per invocation; stores are resumable
SEED         = 20260911
WEIGHT_MODEL = "u_half"    # None for unweighted, else "u_half", "u01" or "int"
WEIGHT_DRAWS = 2           # weight vectors per graph when WEIGHT_MODEL is set

OPTS         = {"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000}
TOL          = 1e-6        # a point is at the floor if E <= floor + this
BIG          = 1e6         # constant returned outside the gate
DEDUP_TOL    = 1e-2        # pi-geodesic radius for "same point"
GATE_NLOC    = 100         # local runs per gate configuration
GATE_NBH     = 8           # basinhopping runs per gate configuration
GATE_NITER   = 15
FLOOR_NITER  = 30          # basinhopping polish on the best restart, off the clock

GRAPHS = {}
if os.path.exists(DATA_CSV):
    with open(DATA_CSV, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            if not row: continue
            n = int(row[0])
            gtype = row[1]
            edges = ast.literal_eval(row[2])
            GRAPHS[f"{n}_{gtype}"] = (n, edges)

DRAWS = [0] if WEIGHT_MODEL is None else list(range(WEIGHT_DRAWS))
CELLS = [(name, p, d) for name in GRAPHS for p in (1, 2, 3) for d in DRAWS
         if not (GRAPHS[name][0] == 10 and p == 3)]


def draw_weights(name, m, draw):
  if WEIGHT_MODEL is None:
    return np.ones(m)
  seed = int(hashlib.md5(f"{name}|{WEIGHT_MODEL}|{draw}".encode()).hexdigest()[:8], 16)
  rng = np.random.default_rng(seed)
  if WEIGHT_MODEL == "u_half":
    return rng.uniform(0.5, 1.5, m)
  if WEIGHT_MODEL == "u01":
    return rng.uniform(0.1, 1.0, m)
  if WEIGHT_MODEL == "int":
    return rng.integers(1, 5, m).astype(float)
  raise ValueError(WEIGHT_MODEL)


def cell_key(name, layers, draw):
  tag = "unit" if WEIGHT_MODEL is None else WEIGHT_MODEL + str(draw)
  return "{}_{}_p{}".format(name, tag, layers)


def cell_weights(c):
  if c.get("weights") == WEIGHT_MODEL and "w" in c:
    return np.array(c["w"], float)
  if WEIGHT_MODEL is None and c.get("weights") is None and "w" not in c:
    return np.ones(c["m"])
  return None


def make_energy_p(n, edges, layers, w):
  energy_theta, energy_theta_batch, D = MW.make_energy_theta(n, edges, w, p=layers)
  return energy_theta


def verify_engine_p(n, edges, layers, w, trials=3):
  m = len(list(nx.Graph(edges).edges()))
  energy = make_energy_p(n, edges, layers, w)
  worst = 0.0
  if WEIGHT_MODEL is None:
    G = nx.Graph(edges)
    HC = L.cost_hamiltonian(n, G)
    for _ in range(trials):
      x = np.random.uniform(-np.pi, np.pi, layers * (m + n))
      worst = max(worst, abs(energy(x) - L.expectation_ma(x, n, layers, G, HC)))
    return worst
  for _ in range(trials):
    x = np.random.uniform(-np.pi, np.pi, layers * (m + n))
    base = energy(x)
    for k in range(layers * m):
      y = x.copy()
      y[k] += np.pi
      worst = max(worst, abs(energy(y) - base))
  return worst


def max_cut_exact(n, edges, w):
  return -MW.brute_weighted_maxcut(n, edges, w)


def counted(energy):
  box = {"n": 0}
  def f(x):
    box["n"] += 1
    return energy(x)
  return f, box


def radius(x):
  w = L.geodesic_vec(np.asarray(x), 0.0)
  return float(np.sqrt((w * w).sum()))


def positive_fraction(x):
  w = L.geodesic_vec(np.asarray(x), 0.0)
  r2 = float((w * w).sum())
  if r2 <= 0.0:
    return 1.0
  return float(np.sqrt(float((np.clip(w, 0.0, None) ** 2).sum()) / r2))


def box_start(rng, D, rmax, fmin, tries=20000):
  for _ in range(tries):
    x = rng.uniform(0, np.pi, D)
    if rmax is not None and radius(x) > rmax:
      continue
    if fmin > 0.0 and positive_fraction(x) < fmin:
      continue
    return x
  return None


def ball_start(rng, D, rmax, fmin, tries=20000):
  for _ in range(tries):
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)
    x = rmax * rng.random() ** (1.0 / D) * u
    if np.abs(x).max() > np.pi / 2:
      continue
    if fmin > 0.0 and positive_fraction(x) < fmin:
      continue
    return x
  return None


def safe_mean(v):
  return float(np.mean(v)) if len(v) else 0.0


def local_run(f, x0):
  return minimize(f, x0, method="L-BFGS-B", options=OPTS)


def bh_run(f, box, x0, floor, niter, stepsize, temp, seed):
  trace = []
  start = box["n"]
  def cb(x, fv, acc):
    trace.append((float(fv), box["n"] - start))
    return fv <= floor + TOL
  res = basinhopping(f, x0, niter=niter, T=temp, stepsize=stepsize,
                     minimizer_kwargs={"method": "L-BFGS-B", "options": OPTS},
                     callback=cb, seed=seed)
  used = box["n"] - start
  hit, mins_used = False, len(trace)
  for k, (fv, ne) in enumerate(trace):
    if fv <= floor + TOL:
      hit, mins_used, used = True, k + 1, ne
      break
  return dict(hit=bool(hit), nfev=int(used), mins=int(mins_used), fun=float(res.fun))


def cost_per_success(hits, runs, nf_hit, nf_miss):
  if hits == 0 or runs == 0:
    return None
  q = hits / runs
  return safe_mean(nf_hit) + (1.0 - q) / q * safe_mean(nf_miss)


def tier(t_run):
  if t_run <= 0.15:
    return dict(nfloor=80, nloc=200, nbh=20, nbhw=10, niter=25)
  if t_run <= 0.60:
    return dict(nfloor=60, nloc=150, nbh=14, nbhw=7, niter=20)
  if t_run <= 1.50:
    return dict(nfloor=40, nloc=100, nbh=10, nbhw=5, niter=12)
  if t_run <= 4.00:
    return dict(nfloor=25, nloc=60, nbh=6, nbhw=4, niter=8)
  return dict(nfloor=15, nloc=40, nbh=4, nbhw=4, niter=6)


def load(path):
  if not os.path.exists(path):
    return {}
  with open(path) as fh:
    return json.load(fh)


def save(st, path):
  folder = os.path.dirname(path) or "."
  name = os.path.basename(path)
  fd, tmp = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=folder)
  try:
    with os.fdopen(fd, "w") as fh:
      json.dump(st, fh)
      fh.flush()
      os.fsync(fh.fileno())
    for attempt in range(5):
      try:
        os.replace(tmp, path)
        return
      except PermissionError:
        if attempt == 4:
          raise
        time.sleep(0.2)
  finally:
    if os.path.exists(tmp):
      os.remove(tmp)


def run_cell(st, name, layers, draw, rng, deadline):
  key = cell_key(name, layers, draw)
  n, edges = GRAPHS[name]
  m = len(list(nx.Graph(edges).edges()))
  D = layers * (m + n)
  w = draw_weights(name, m, draw)
  energy = make_energy_p(n, edges, layers, w)
  c = st.get(key)
  if c is None:
    t0 = time.time()
    for _ in range(3):
      local_run(energy, rng.uniform(0, np.pi, D))
    t_run = (time.time() - t0) / 3.0
    c = dict(graph=name, key=key, weights=WEIGHT_MODEL, draw=draw,
             w=[float(v) for v in w], n=n, m=m, layers=layers, D=D,
             density=2.0 * m / (n * (n - 1)), max_cut=max_cut_exact(n, edges, w),
             t_run=t_run, cfg=tier(t_run), floor_samples=[], floor_x=None, floor=None,
             certified=False, local=[], bh_default=[], bh_wide=[])
    st[key] = c
    save(st, STORE)
  cfg, t_run = c["cfg"], c["t_run"]

  while len(c["floor_samples"]) < cfg["nfloor"]:
    if time.time() + t_run > deadline:
      save(st, STORE)
      return False
    r = local_run(energy, rng.uniform(0, np.pi, D))
    if c["floor_x"] is None or r.fun <= min(c["floor_samples"] + [np.inf]):
      c["floor_x"] = [float(v) for v in r.x]
    c["floor_samples"].append(float(r.fun))
  if c["floor"] is None:
    r = basinhopping(energy, np.array(c["floor_x"]), niter=FLOOR_NITER, T=0.5,
                     stepsize=float(np.pi / 2),
                     minimizer_kwargs={"method": "L-BFGS-B", "options": OPTS},
                     seed=int(rng.integers(1, 10 ** 8)))
    fl = min(float(min(c["floor_samples"])), float(r.fun))
    c["certified"] = bool(abs(fl + c["max_cut"]) < TOL)
    c["floor"] = -float(c["max_cut"]) if c["certified"] else fl
    save(st, STORE)

  floor = c["floor"]
  f, box = counted(energy)
  while len(c["local"]) < cfg["nloc"]:
    if time.time() + t_run > deadline:
      save(st, STORE)
      return False
    before = box["n"]
    r = local_run(f, rng.uniform(0, np.pi, D))
    rec = dict(nfev=int(box["n"] - before), fun=float(r.fun))
    if r.fun <= floor + TOL:
      rec["r"] = radius(r.x)
      rec["pf"] = positive_fraction(r.x)
      rec["x"] = [round(float(v), 6) for v in L.wrap_pi(r.x)]
    c["local"].append(rec)
    if len(c["local"]) % 10 == 0:
      save(st, STORE)
  save(st, STORE)

  for tag, step, temp, cnt in (("bh_default", 0.5, 1.0, cfg["nbh"]),
                               ("bh_wide", float(np.pi / 2), 0.5, cfg["nbhw"])):
    while len(c[tag]) < cnt:
      if time.time() + (cfg["niter"] + 1) * t_run > deadline:
        save(st, STORE)
        return False
      c[tag].append(bh_run(f, box, rng.uniform(0, np.pi, D), floor,
                           cfg["niter"], step, temp, int(rng.integers(1, 10 ** 8))))
      save(st, STORE)
  return True


def shell_of(c):
  pts = [r for r in c["local"] if "r" in r]
  if not pts:
    return None, None
  rmin = float(min(r["r"] for r in pts))
  return rmin, float(max(r["pf"] for r in pts if r["r"] <= rmin + 1e-3))


def gate_configs(shell_r, shell_pf):
  out = []
  for k in (None, 1.4, 1.2, 1.1, 1.0):
    out.append(dict(tag="R" + str(k), rmax=(None if k is None else k * shell_r), fmin=0.0))
  for a in (0.5, 0.8, 0.98):
    out.append(dict(tag="F" + str(a), rmax=1.1 * shell_r, fmin=a * shell_pf))
  return out


def run_gate(st, gt, key, rng, deadline):
  c = st[key]
  w = cell_weights(c)
  if w is None:
    return True
  shell_r, shell_pf = shell_of(c)
  if shell_r is None:
    return True
  n, edges = GRAPHS[c["graph"]]
  D, floor, t_run = c["D"], c["floor"], c["t_run"]
  energy = make_energy_p(n, edges, c["layers"], w)
  for cf in gate_configs(shell_r, shell_pf):
    for regime in ("box", "ball"):
      if regime == "ball" and cf["rmax"] is None:
        continue
      k = key + "|" + cf["tag"] + "|" + regime
      rec = gt.setdefault(k, dict(cell=key, tag=cf["tag"], regime=regime, rmax=cf["rmax"],
                                  fmin=cf["fmin"], shell_r=shell_r, shell_pf=shell_pf,
                                  loc=[], bh=[]))
      f, box = counted(energy)
      while len(rec["loc"]) < GATE_NLOC:
        if time.time() + t_run > deadline:
          save(gt, GATE_STORE)
          return False
        x0 = (box_start(rng, D, cf["rmax"], cf["fmin"]) if regime == "box"
              else ball_start(rng, D, cf["rmax"], cf["fmin"]))
        if x0 is None:
          rec["loc"].append(dict(nfev=0, fun=BIG, skipped=True))
          continue
        b = box["n"]
        r = local_run(f, x0)
        rec["loc"].append(dict(nfev=int(box["n"] - b), fun=float(r.fun)))
      while len(rec["bh"]) < GATE_NBH:
        if time.time() + (GATE_NITER + 1) * t_run > deadline:
          save(gt, GATE_STORE)
          return False
        x0 = (box_start(rng, D, cf["rmax"], cf["fmin"]) if regime == "box"
              else ball_start(rng, D, cf["rmax"], cf["fmin"]))
        if x0 is None:
          rec["bh"].append(dict(hit=False, nfev=0, mins=0, fun=BIG, skipped=True))
          continue
        rec["bh"].append(bh_run(f, box, x0, floor, GATE_NITER, 0.5, 1.0,
                                int(rng.integers(1, 10 ** 8))))
      save(gt, GATE_STORE)
  return True


def summarize(c):
  floor = c["floor"]
  if floor is None:
    return None
  hit_nf = [r["nfev"] for r in c["local"] if r["fun"] <= floor + TOL]
  miss_nf = [r["nfev"] for r in c["local"] if r["fun"] > floor + TOL]
  runs, hits = len(c["local"]), len(hit_nf)
  out = dict(graph=c.get("key", c["graph"]), weights=c.get("weights"), n=c["n"],
             m=c["m"], p=c["layers"], D=c["D"],
             density=round(c["density"], 3), floor=floor, max_cut=c["max_cut"],
             certified=c.get("certified", False), loc_runs=runs, loc_hits=hits,
             loc_q=(hits / runs if runs else 0.0),
             loc_runs_to_hit=(runs / hits if hits else None),
             loc_nfev_mean=safe_mean([r["nfev"] for r in c["local"]]),
             loc_cost=cost_per_success(hits, runs, hit_nf, miss_nf))
  for tag in ("bh_default", "bh_wide"):
    rs = c[tag]
    h = [r["nfev"] for r in rs if r["hit"]]
    mi = [r["nfev"] for r in rs if not r["hit"]]
    out[tag + "_runs"] = len(rs)
    out[tag + "_hits"] = len(h)
    out[tag + "_q"] = (len(h) / len(rs) if rs else 0.0)
    out[tag + "_basins"] = (safe_mean([r["mins"] for r in rs if r["hit"]]) if h else None)
    out[tag + "_nfev_mean"] = (safe_mean([r["nfev"] for r in rs]) if rs else None)
    out[tag + "_cost"] = cost_per_success(len(h), len(rs), h, mi)
  shell_r, shell_pf = shell_of(c)
  out["shell_r"] = shell_r
  out["shell_pf"] = shell_pf
  if shell_r is not None:
    pts = [r for r in c["local"] if "r" in r]
    out["r_median"] = float(np.median([r["r"] for r in pts]))
    out["shell_r_over_rand"] = shell_r / (0.9069 * np.sqrt(c["D"]))
    reps = []
    for r in pts:
      xw = np.array(r["x"])
      if all(L.geodesic_dist(xw, q) > DEDUP_TOL for q in reps):
        reps.append(xw)
    out["distinct"] = len(reps)
  return out


def write_csv(rows, cols, path):
  with open(path, "w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in rows:
      vals = []
      for c in cols:
        v = r.get(c)
        vals.append("" if v is None else
                    (f"{v:.6g}" if isinstance(v, float) else str(v)))
      fh.write(",".join(vals) + "\n")


def report():
  st = load(STORE)
  rows = [s for s in (summarize(c) for c in st.values()) if s]
  if not rows:
    print("no completed cells in", STORE)
    print("set RUN_SWEEP = True and run again")
    return []
  rows.sort(key=lambda r: (r["graph"], r["p"]))
  print("{:<20}{:>2}{:>5}{:>6}{:>8}{:>5}{:>7}{:>9}{:>9}{:>7}{:>8}{:>9}{:>8}".format(
    "graph", "p", "D", "dens", "floor", "cert", "locQ", "runs/hit", "locCost",
    "bhQ", "basins", "bhCost", "bh/loc"))
  for r in rows:
    ratio = (r["bh_default_cost"] / r["loc_cost"]
             if (r["bh_default_cost"] and r["loc_cost"]) else float("nan"))
    print("{:<20}{:>2}{:>5}{:>6.2f}{:>8.2f}{:>5}{:>7.3f}{:>9.1f}{:>9.0f}{:>7.3f}"
          "{:>8.2f}{:>9.0f}{:>8.2f}".format(
            r["graph"], r["p"], r["D"], r["density"], r["floor"],
            "Y" if r["certified"] else "n", r["loc_q"],
            r["loc_runs_to_hit"] or float("nan"), r["loc_cost"] or float("nan"),
            r["bh_default_q"], r["bh_default_basins"] or float("nan"),
            r["bh_default_cost"] or float("nan"), ratio))
  d = np.array([r["bh_default_cost"] / r["loc_cost"] for r in rows
                if r["bh_default_cost"] and r["loc_cost"]])
  w = np.array([r["bh_wide_cost"] / r["loc_cost"] for r in rows
                if r["bh_wide_cost"] and r["loc_cost"]])
  if len(d):
    print("\nbh/local cost  default: median {:.2f} geomean {:.2f}  local cheaper {}/{}".format(
      np.median(d), np.exp(np.mean(np.log(d))), int((d > 1).sum()), len(d)))
  if len(w):
    print("bh/local cost  wide   : median {:.2f} geomean {:.2f}  local cheaper {}/{}".format(
      np.median(w), np.exp(np.mean(np.log(w))), int((w > 1).sum()), len(w)))
  write_csv(rows, ["graph", "weights", "n", "m", "p", "D", "density", "floor", "max_cut", "certified",
                   "loc_runs", "loc_hits", "loc_q", "loc_runs_to_hit", "loc_nfev_mean",
                   "loc_cost", "bh_default_runs", "bh_default_hits", "bh_default_q",
                   "bh_default_basins", "bh_default_nfev_mean", "bh_default_cost",
                   "bh_wide_q", "bh_wide_cost", "shell_r", "shell_pf", "r_median",
                   "shell_r_over_rand", "distinct"], MAIN_CSV)
  print("wrote", MAIN_CSV)
  return rows


def gate_report():
  st = load(STORE)
  gt = load(GATE_STORE)
  if not gt:
    print("\nno gate data in", GATE_STORE)
    return
  print("\n{:<13}{:<8}{:<6}{:>7}{:>6}{:>7}{:>10}{:>7}{:>11}".format(
    "cell", "gate", "regime", "rmax", "fmin", "locQ", "locCost", "bhQ", "bhCost"))
  rows = []
  for k in sorted(gt):
    r = gt[k]
    floor = st[r["cell"]]["floor"]
    loc = [a for a in r["loc"] if not a.get("skipped")]
    bhr = [a for a in r["bh"] if not a.get("skipped")]
    lh = [a["nfev"] for a in loc if a["fun"] <= floor + TOL]
    lm = [a["nfev"] for a in loc if a["fun"] > floor + TOL]
    bh = [a["nfev"] for a in bhr if a["hit"]]
    bm = [a["nfev"] for a in bhr if not a["hit"]]
    lc = cost_per_success(len(lh), len(r["loc"]), lh, lm)
    bc = cost_per_success(len(bh), len(r["bh"]), bh, bm)
    rows.append(dict(cell=r["cell"], gate=r["tag"], regime=r["regime"], rmax=r["rmax"],
                     fmin=r["fmin"], shell_r=r["shell_r"], shell_pf=r["shell_pf"],
                     rmax_over_shell=(r["rmax"] / r["shell_r"] if r["rmax"] else None),
                     loc_runs=len(loc), loc_hits=len(lh),
                     loc_skipped=len(r["loc"]) - len(loc),
                     loc_q=len(lh) / len(loc) if loc else 0.0,
                     loc_nfev_mean=safe_mean([a["nfev"] for a in loc]),
                     loc_cost=lc, bh_runs=len(bhr), bh_hits=len(bh),
                     bh_q=len(bh) / len(bhr) if bhr else 0.0, bh_cost=bc))
    print("{:<13}{:<8}{:<6}{:>7.2f}{:>6.2f}{:>7.2f}{:>10.0f}{:>7.2f}{:>11.0f}".format(
      r["cell"], r["tag"], r["regime"], r["rmax"] or 0.0, r["fmin"],
      rows[-1]["loc_q"], lc if lc else float("nan"),
      rows[-1]["bh_q"], bc if bc else float("nan")))
  write_csv(rows, ["cell", "gate", "regime", "rmax", "fmin", "shell_r", "shell_pf",
                   "rmax_over_shell", "loc_runs", "loc_hits", "loc_skipped", "loc_q", "loc_nfev_mean",
                   "loc_cost", "bh_runs", "bh_hits", "bh_q", "bh_cost"], GATE_CSV)
  print("wrote", GATE_CSV)


def main():
  os.makedirs(RESULTS, exist_ok=True)
  rng = np.random.default_rng(SEED)
  if VERIFY and GRAPHS:
    verify_name = list(GRAPHS.keys())[0]
    n, edges = GRAPHS[verify_name]
    m = len(list(nx.Graph(edges).edges()))
    w = draw_weights(verify_name, m, 0)
    what = "vs qiskit" if WEIGHT_MODEL is None else "under a pi shift in theta"
    for p in (1, 2, 3):
      print("engine check p={} max abs diff {} {:.2e}".format(
        p, what, verify_engine_p(n, edges, p, w)))
  if RUN_SWEEP:
    st = load(STORE)
    deadline = time.time() + TIME_BUDGET
    for name, p, d in CELLS:
      if not run_cell(st, name, p, d, rng, deadline):
        print("budget reached, partial at", cell_key(name, p, d))
        break
      print("cell done", cell_key(name, p, d), flush=True)
    save(st, STORE)
  if RUN_GATES:
    st = load(STORE)
    gt = load(GATE_STORE)
    deadline = time.time() + TIME_BUDGET
    for key in [k for k in st if st[k]["floor"] is not None and cell_weights(st[k]) is not None]:
      if not run_gate(st, gt, key, rng, deadline):
        print("budget reached, partial gate at", key)
        break
      print("gate done", key, flush=True)
    save(gt, GATE_STORE)
  report()
  gate_report()


if __name__ == "__main__":
  main()