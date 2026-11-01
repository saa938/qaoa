"""
Usage: python src/research_verification.py
Independently certify small weighted Max-Cut instances and export paper-comparison evidence.
This stage uses local CPU only and never submits Modal work.
"""

import csv
import hashlib
import json
import os
import platform
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "results"
EXPANDED = ROOT / "full_100k" / "results"
OUT = EXPANDED / "paper_comparison"
WEIGHT_SCALE = 10000
BOUND_TOLERANCE = 1e-10
FLOOR_TOLERANCE = 1e-6
MILP_CHECKS_PER_N = 3
MODE = "verify"
STATE_CACHE = {}
PARITY_CACHE = {}


# Bound p=1 cut expectation using zero correlations between disjoint radius-one lightcones.
def locality_bound(n, edges, states, values, maximum):
    neighborhoods = [1 << vertex for vertex in range(n)]
    for u, v in edges:
        neighborhoods[u] |= 1 << v
        neighborhoods[v] |= 1 << u
    best, witness, eligible = 2 * maximum, None, 0
    for u in range(n):
        for v in range(u + 1, n):
            if neighborhoods[u] & neighborhoods[v]:
                continue
            eligible += 1
            key = (n, u, v)
            if key not in PARITY_CACHE:
                PARITY_CACHE[key] = (((states >> u) ^ (states >> v)) & 1).astype(bool)
            opposite = PARITY_CACHE[key]
            same_value, opposite_value = int(values[~opposite].max()), int(values[opposite].max())
            numerator = same_value + opposite_value
            if numerator < best:
                best = numerator
                witness = {"vertices": [u, v], "same_partition_max_cut_integer": same_value,
                           "opposite_partition_max_cut_integer": opposite_value,
                           "same_partition_witness": int(states[(~opposite) & (values == same_value)][0]),
                           "opposite_partition_witness": int(states[opposite & (values == opposite_value)][0])}
    return {"p1_cut_upper_bound_numerator": best, "p1_cut_upper_bound_denominator": 2 * WEIGHT_SCALE,
            "p1_cut_upper_bound": best / (2 * WEIGHT_SCALE), "p1_energy_lower_bound": -best / (2 * WEIGHT_SCALE),
            "disjoint_lightcone_pairs": eligible, "locality_bound_witness": witness,
            "dominating_vertices": [vertex for vertex, mask in enumerate(neighborhoods) if mask == (1 << n) - 1]}


# Reject incomplete or contradictory mathematical evidence.
def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# Write complete JSON evidence atomically without accepting NaN.
def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


# Enumerate one representative of each globally complemented bit-string pair.
def exact_cut(row):
    started = time.perf_counter()
    n = int(row["n"])
    require(2 <= n <= 14, "Exact certification is limited to n <= 14")
    weights = []
    for weight in row["w_sorted"]:
        exact = Decimal(str(weight)) * WEIGHT_SCALE
        require(exact == exact.to_integral_value(), "An edge weight is not on the documented exact decimal grid")
        weights.append(int(exact))
    require(len(weights) == len(row["edges"]) and all(w > 0 for w in weights), "Invalid positive weights")
    if n not in STATE_CACHE:
        STATE_CACHE[n] = np.arange(1 << (n - 1), dtype=np.int64) << 1
    states = STATE_CACHE[n]
    values = np.zeros(len(states), dtype=np.int64)
    for (u, v), weight in zip(row["edges"], weights):
        values += weight * (((states >> u) ^ (states >> v)) & 1)
    maximum = int(values.max())
    winners = states[values == maximum]
    witness = int(winners[0])
    independently_summed = sum(w for (u, v), w in zip(row["edges"], weights)
                               if ((witness >> u) & 1) != ((witness >> v) & 1))
    require(independently_summed == maximum, "The maximum-cut witness has the wrong weight")
    require(abs(maximum / WEIGHT_SCALE - row["max_cut"]) <= BOUND_TOLERANCE,
            "Saved Max-Cut value differs from exact integer enumeration")
    locality = locality_bound(n, row["edges"], states, values, maximum)
    return {"max_cut_integer": maximum, "weight_scale": WEIGHT_SCALE, "exact_max_cut": maximum / WEIGHT_SCALE,
            "energy_lower_bound": -maximum / WEIGHT_SCALE, "witness_bitmask_little_endian": witness,
            "witness_bits_node_order": [(witness >> i) & 1 for i in range(n)],
            "optimal_complement_pairs": len(winners), "optimal_bitstrings": 2 * len(winners),
            "enumerated_complement_pairs": len(states), "canonical_optimal_bitmasks": winners.tolist(),
            "exact_solver_seconds": time.perf_counter() - started, **locality}


# Prepare an exact optimal-cut cat state whenever the graph contains a spanning star.
def optimal_star_witness(row, certificate):
    import check_rows
    center = certificate["dominating_vertices"][0]
    bits = certificate["witness_bits_node_order"]
    gamma = [np.pi / (4 * weight) if center in edge else 0.
             for edge, weight in zip(row["edges"], row["w_sorted"])]
    beta = [0. if vertex == center else (-np.pi / 4 if bits[vertex] != bits[center] else np.pi / 4)
            for vertex in range(row["n"])]
    physical = float(check_rows.energy(row["n"], row["edges"], row["w_sorted"], gamma, beta))
    require(abs(physical + certificate["exact_max_cut"]) <= 1e-10, "Constructive spanning-star witness is inconsistent")
    return {"center": center, "gamma": gamma, "beta": beta, "physical_energy": physical,
            "proof": "Activate only the spanning-star ZZ gates with w*gamma=pi/4, leave the center mixer zero and use leaf mixers +/-pi/4 to encode the exact optimal partition. Both global-complement bitstrings have that cut weight. Other cost-layer gates are zero, while the objective still includes every weighted edge."}


# Cross-check selected certificates with a separate integer-programming solver.
def milp_cut(row, certificate):
    started = time.perf_counter()
    n, m = row["n"], len(row["edges"])
    constraints = lil_matrix((4 * m, n + m), dtype=float)
    bounds = np.zeros(4 * m)
    for k, (u, v) in enumerate(row["edges"]):
        constraints[4 * k, [u, v, n + k]] = [-1, -1, 1]
        constraints[4 * k + 1, [u, v, n + k]] = [1, 1, 1]
        bounds[4 * k + 1] = 2
        constraints[4 * k + 2, [u, v, n + k]] = [1, -1, -1]
        constraints[4 * k + 3, [u, v, n + k]] = [-1, 1, -1]
    weights = np.asarray([int(Decimal(str(w)) * WEIGHT_SCALE) for w in row["w_sorted"]])
    objective = np.r_[np.zeros(n), -weights]
    lower, upper = np.zeros(n + m), np.ones(n + m)
    upper[0] = 0
    result = milp(objective, integrality=np.ones(n + m), bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(constraints.tocsr(), -np.inf, bounds),
                  options={"time_limit": 30.0, "mip_rel_gap": 0.0})
    require(result.success and result.status == 0, "MILP failed to certify optimality")
    require(abs(-result.fun - certificate["max_cut_integer"]) <= 1e-5
            and abs(-result.mip_dual_bound - certificate["max_cut_integer"]) <= 1e-5,
            "MILP primal/dual bounds disagree with exact enumeration")
    return {"status": int(result.status), "primal_integer": -float(result.fun),
            "dual_bound_integer": -float(result.mip_dual_bound), "mip_gap": float(result.mip_gap),
            "seconds": time.perf_counter() - started}


# Read every saved scientific record, preferring a completed expanded corpus.
def record_stream():
    completed = EXPANDED / "preflight" / "full_corpus_csv_gate.json"
    selected = EXPANDED if completed.exists() else ORIGINAL
    for tag, folder in [("full", "dataset"), ("large", "dataset_test")]:
        for path in sorted((selected / folder).glob("shard_*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    yield tag, json.loads(line)
    for path in sorted((selected / "ian_floors").glob("row_*.json")):
        yield "ian", json.loads(path.read_text())


# Classify empirical labels using an exact classical bound without assuming p=1 saturation.
def verify_records():
    OUT.mkdir(parents=True, exist_ok=True)
    seen, totals, sizes, families = set(), Counter(), defaultdict(Counter), defaultdict(Counter)
    checks, p1_counts, p1_usable_counts = Counter(), Counter(), Counter()
    p1_sizes, p1_families, star_findings = defaultdict(Counter), defaultdict(Counter), []
    started = time.perf_counter()
    output = OUT / "classical_certificates.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for tag, row in record_stream():
            key = (tag, row["id"])
            require(key not in seen, "Duplicate graph in verification population")
            seen.add(key)
            certificate = exact_cut(row)
            energy = row.get("energy", row.get("floor"))
            if row.get("pick") is None:
                energy = None
            gap = float(energy + certificate["exact_max_cut"]) if energy is not None else None
            require(gap is None or gap >= -BOUND_TOLERANCE, "A saved physical energy violates the classical ground-state bound")
            category = "no_pick" if gap is None else "attains_exact_bound_within_tolerance" if gap <= FLOOR_TOLERANCE else "positive_gap_no_variational_global_certificate"
            p1_gap = float(energy + certificate["p1_cut_upper_bound"]) if energy is not None else None
            require(p1_gap is None or p1_gap >= -BOUND_TOLERANCE, "A saved label violates the p=1 lightcone bound")
            p1_category = "no_pick" if p1_gap is None else "matches_classical_bound" if gap <= FLOOR_TOLERANCE else "matches_stronger_locality_bound" if p1_gap <= FLOOR_TOLERANCE else "uncertified_gap"
            result = {"corpus": tag, "id": row["id"], "n": row["n"], "type": row.get("type"),
                      "saved_usable": row.get("usable"), "reference_floor": row.get("floor"), "saved_label_energy": energy,
                      "label_energy_plus_exact_max_cut": gap, "certificate_status": category,
                      "label_energy_plus_p1_bound": p1_gap, "p1_certification_status": p1_category,
                      "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(), **certificate}
            if gap is not None and gap > FLOOR_TOLERANCE and certificate["dominating_vertices"]:
                result["explicit_better_star_witness"] = optimal_star_witness(row, certificate)
                p1_category = result["p1_certification_status"] = "explicit_better_star_witness"
                star_findings.append({"corpus": tag, "id": row["id"], "n": row["n"], "saved_usable": row.get("usable"),
                    "reference_energy": energy, "certified_p1_optimal_energy": -certificate["exact_max_cut"],
                    "independent_witness_energy": result["explicit_better_star_witness"]["physical_energy"],
                    "center": result["explicit_better_star_witness"]["center"], "row_sha256": result["row_sha256"]})
            p1_counts[tag + ":" + p1_category] += 1
            p1_sizes[tag + ":n=" + str(row["n"])][p1_category] += 1
            p1_families[tag + ":" + str(row.get("type"))][p1_category] += 1
            if row.get("usable") is True:
                p1_usable_counts[tag + ":" + p1_category] += 1
            if checks[row["n"]] < MILP_CHECKS_PER_N:
                result["independent_milp"] = milp_cut(row, certificate)
                checks[row["n"]] += 1
            handle.write(json.dumps(result, allow_nan=False) + "\n")
            totals[tag + ":" + category] += 1
            sizes[tag + ":n=" + str(row["n"])][category] += 1
            families[tag + ":" + str(row.get("type"))][category] += 1
            if len(seen) % 10000 == 0:
                print(json.dumps({"classically_verified": len(seen), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    temporary.replace(output)
    result = {"records": len(seen), "counts": dict(totals), "by_n": dict(sizes), "by_type": dict(families),
              "independent_milp_checks_by_n": dict(checks), "elapsed_seconds": time.perf_counter() - started,
              "local_cpu_only": True, "modal_compute": False, "platform": platform.platform(),
              "raw_file": str(output), "raw_sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
              "scope": "All saved records, including rejected records; no missing graph is silently counted as certified.",
              "mathematical_limit": "Exact Max-Cut certifies the Hamiltonian ground energy. A feasible p=1 label reaching that bound is variationally optimal within the stated numerical tolerance. A positive gap does not prove the p=1 optimizer is suboptimal. Neither test proves unique angles.",
              "floor_tolerance": FLOOR_TOLERANCE, "bound_tolerance": BOUND_TOLERANCE}
    result.update({"p1_counts_all_saved": dict(p1_counts), "p1_counts_saved_usable": dict(p1_usable_counts),
                   "p1_counts_by_n": dict(p1_sizes), "p1_counts_by_type": dict(p1_families),
                   "known_suboptimal_star_findings": star_findings,
                   "verification_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "locality_proof": "For p=1, backwards-evolved Zi is supported on the closed neighborhood N[i]. For disjoint N[u],N[v], expectations factorize on the product input. Global bit-flip symmetry gives <Zi>=0, so <ZuZv>=0. Same/opposite partitions each have probability 1/2; expected cut <= (C_same_max+C_opposite_max)/2. Conditional maxima are exact integers obtained by enumerating all complementary bitstring pairs. Minimize over eligible vertex pairs. Matching this bound certifies p=1 optimal energy within the numerical witness tolerance, not unique angles.",
                   "sources": ["https://www.nature.com/articles/s41534-023-00718-4", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9043219/"],
                   "novelty_claim": "No priority/novelty claim; apply the lightcone constraint and a constructive spanning-star witness to this dataset."})
    save_json(OUT / "classical_verification_summary.json", result)
    return result


# Apply the same exact bounds to the 21 pinned reference graphs, a separate corpus.
def verify_reference_graphs():
    import check_rows
    path = ROOT / "data/reference_picks.json"
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    require(source_hash == "a6c2831c21f940fdd50ed9bc1306ca2f89cb4eb9b91f5a023ded450c9cdd5345",
            "Reference file differs from the user-approved bytes")
    rows = json.loads(path.read_text())["graphs"]
    require(len(rows) == 21, "The full expected reference list is required")
    records, counts = [], Counter()
    for row in rows:
        certificate = exact_cut(row)
        m = len(row["edges"])
        value = float(check_rows.energy(row["n"], row["edges"], row["w_sorted"], row["pick"][:m], row["pick"][m:]))
        require(abs(value - row["floor"]) <= FLOOR_TOLERANCE, "Reference point does not reproduce its energy")
        require(value >= certificate["p1_energy_lower_bound"] - BOUND_TOLERANCE, "Reference violates the lightcone bound")
        category = "matches_classical_bound" if value <= -row["max_cut"] + FLOOR_TOLERANCE else "matches_stronger_locality_bound" if value <= certificate["p1_energy_lower_bound"] + FLOOR_TOLERANCE else "uncertified_gap"
        counts[category] += 1
        records.append({"corpus": "reference_21", "id": row["id"], "n": row["n"], "energy": value,
                        "p1_certification_status": category, **certificate})
    result = {"expected": 21, "received": len(records), "source_sha256": source_hash, "counts": dict(counts),
              "records": records, "note": "Reference IDs belong to their own corpus, not the production training ID namespace."}
    save_json(OUT / "reference_p1_certificates.json", result)
    return result


# Preserve original held-out polish data without inventing retrospective timings.
def export_existing_polishes():
    fields = ["experiment", "evaluation", "corpus", "model_seed", "strategy", "id", "candidate", "n",
              "energy_before", "energy_after", "nfev", "circuit_evaluations", "hit", "optimizer_wall_seconds"]
    target = OUT / "previous_heldout_polishes.csv"
    count = 0
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in sorted((ORIGINAL / "gnn" / "evaluations").glob("*.json")):
            value = json.loads(path.read_text())
            if value.get("set_name") not in ["test_split", "large", "ian"]:
                continue
            seed = next((part[1:] for part in path.stem.split("_") if part.startswith("s") and part[1:].isdigit()), "")
            for strategy in ["raw", "snap", "better"]:
                for row in value["results"].get(strategy, []):
                    for k, point in enumerate(row["polishes"]):
                        record = {name: point.get(name) for name in ["energy_before", "energy_after", "nfev", "circuit_evaluations", "hit"]}
                        record.update({"experiment": "original_63288_usable", "evaluation": path.stem,
                                       "corpus": value["corpus"], "model_seed": seed, "strategy": strategy,
                                       "id": row["id"], "candidate": k, "n": row["n"], "optimizer_wall_seconds": None})
                        writer.writerow(record)
                        count += 1
    result = {"rows": count, "csv": str(target), "per_polish_runtime": "not recorded in the prior study; blank values are missing measurements, not zero",
              "angles_and_all_fields": str(ORIGINAL / "gnn" / "evaluations"), "new_optimization": False}
    save_json(OUT / "previous_raw_data_manifest.json", result)
    return result


# Exhibit a rigorously solvable p=1 instance whose variational floor exceeds -MaxCut.
def p1_path_counterexample():
    from scipy.linalg import expm
    import check_rows
    row = {"n": 4, "edges": [[0, 1], [1, 2], [2, 3]], "w_sorted": [1.2, .7, 1.6], "max_cut": 3.5}
    certificate = exact_cut(row)
    gamma = [np.pi / (4 * 1.2), 0., np.pi / (4 * 1.6)]
    beta = [0., -np.pi / 4, 0., -np.pi / 4]
    witness_energy = check_rows.energy(4, row["edges"], row["w_sorted"], gamma, beta)
    require(abs(witness_energy + 3.15) <= 1e-12, "The analytic weighted p=1 path witness is incorrect")
    indices = np.arange(16)
    spins = 1 - 2 * ((indices[:, None] >> np.arange(4)) & 1)
    rng = np.random.default_rng(20261030)
    maximum_correlation = 0.
    for unused in range(50):
        point = rng.uniform(-np.pi, np.pi, 7)
        phases = sum(point[k] * row["w_sorted"][k] * spins[:, u] * spins[:, v] for k, (u, v) in enumerate(row["edges"]))
        state = np.exp(1j * phases) / 4
        unitary = np.ones((1, 1), dtype=complex)
        for qubit in reversed(range(4)):
            unitary = np.kron(unitary, expm(1j * point[3 + qubit] * np.array([[0., 1.], [1., 0.]])))
        probability = np.abs(unitary @ state) ** 2
        maximum_correlation = max(maximum_correlation, abs(float(probability @ (spins[:, 0] * spins[:, 3]))))
    require(maximum_correlation <= 1e-12, "The p=1 endpoint light-cone check failed")
    result = {"graph": row, "classical_certificate": certificate, "exact_p1_max_expected_cut": 3.15,
              "exact_p1_min_energy": -3.15, "classical_ground_energy": -3.5,
              "gamma_witness": gamma, "beta_witness": beta, "independent_witness_energy": witness_energy,
              "proof": "For the four-vertex weighted path with weights (1.2,0.7,1.6), the p=1 backwards light cones of endpoint Z0 and Z3 have disjoint supports {0,1} and {2,3}. Their expectations factorize on the product input |+>^4, and each one-point Z expectation is zero. Therefore <Z0 Z3>=0, so the endpoints disagree with probability 1/2. Equal endpoints require an even number of cut edges, hence cut weight <= 3.5-min(1.2,0.7,1.6)=2.8. Opposite endpoints allow cut weight <=3.5. Thus expected cut <= (2.8+3.5)/2=3.15 for every p=1 parameter vector. The explicit feasible angles attain 3.15, proving the exact p=1 optimum. The classical maximum is 3.5. Distinct weights also remove the path reversal automorphism.",
              "random_sanity_checks": 50, "max_endpoint_correlation_error": maximum_correlation,
              "scope": "Field-free weighted-Max-Cut Hamiltonian, distinct positive weights within [.5,2], one ZZ cost layer and node-wise X mixers. This proves that weighted p=1 MA-QAOA does not universally attain exact Max-Cut."}
    save_json(OUT / "p1_path_counterexample.json", result)
    return result


# Complete the free classical checks and export existing raw measurements.
# Summarize saved generation effort without equating elapsed graph time with billed core-hours.
def generation_statistics():
    grouped = defaultdict(list)
    for tag, row in record_stream():
        if tag not in ["full", "large"]:
            continue
        for group in ["all", "n=" + str(row["n"]), "type=" + row["type"],
                      "n=" + str(row["n"]) + ";type=" + row["type"]]:
            grouped[(tag, group)].append(row)
    output = []
    for (tag, group), rows in sorted(grouped.items()):
        result = {"corpus": tag, "group": group, "saved": len(rows),
                  "usable": sum(row.get("usable") is True for row in rows),
                  "pickless": sum(row.get("pick") is None for row in rows),
                  "audited": sum(row.get("audit") is True for row in rows),
                  "audit_disagreements": sum(row.get("audit") is True and row.get("audit_same") is not True for row in rows)}
        for name in ["secs_main", "secs", "nfev", "restarts", "m"]:
            values = [float(row[name]) for row in rows if type(row.get(name)) in [float, int] and np.isfinite(row[name])]
            result[name + "_records"] = len(values)
            if values:
                result.update({name + "_sum": float(np.sum(values)), name + "_mean": float(np.mean(values)),
                               name + "_median": float(np.median(values)), name + "_p95": float(np.quantile(values, .95)),
                               name + "_max": max(values)})
        output.append(result)
    value = {"groups": output, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "scope": "Saved graph records only. secs_main is the filter's reported elapsed time; secs includes graph preparation, checks and any unrelated-seed audit. Summed seconds exclude interrupted work, storage and scheduling and are not billed core-hours. nfev and restarts retain the unchanged filter's own definitions; they are separate from measured prediction-polish query counts."}
    save_json(OUT / "generation_statistics.json", value)
    fields = ["corpus", "group"] + sorted({name for row in output for name in row} - {"corpus", "group"})
    with (OUT / "generation_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(output)
    return value


def main():
    if MODE != "verify":
        raise ValueError("Unsupported verification mode")
    result = verify_records()
    verify_reference_graphs()
    counterexample = p1_path_counterexample()
    raw = export_existing_polishes()
    generation_statistics()
    print(json.dumps({"verification": result["counts"], "seconds": result["elapsed_seconds"], "raw_rows": raw["rows"],
                      "p1_path_floor": counterexample["exact_p1_min_energy"]}), flush=True)


if __name__ == "__main__":
    main()
