"""Independently audit every saved p=1 polish without running an optimizer."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import check_rows
from audit_evaluation_numerics import bounded_angles
from report_floor_review import Corpus, FOLDER, RESULTS, CHECKER_SHA, require


def energy_batch(row, points):
    """Exact p=1 ZZ expectations, including all shared-neighbor triangle terms."""
    points = np.atleast_2d(points).astype(float)
    weights = np.asarray(row["w_sorted"])
    m, n = len(weights), row["n"]
    edges = np.asarray(row["edges"], dtype=int)
    # Large-angle arguments are first reduced using exact saved inputs and
    # high-precision pi, independently of the optimizer's phase arithmetic.
    for index in np.flatnonzero(np.max(np.abs(points), axis=1) > 1000):
        points[index] = bounded_angles(points[index], weights, n)
    phases = points[:, :m] * weights
    theta = np.zeros((len(points), n, n))
    theta[:, edges[:, 0], edges[:, 1]] = phases
    theta[:, edges[:, 1], edges[:, 0]] = phases
    left = theta[:, edges[:, 0], :].copy()
    right = theta[:, edges[:, 1], :].copy()
    for coordinate in [0, 1]:
        left[:, np.arange(m), edges[:, coordinate]] = 0
        right[:, np.arange(m), edges[:, coordinate]] = 0
    c, s = np.cos(2 * points[:, m:]), np.sin(2 * points[:, m:])
    u, v = edges[:, 0], edges[:, 1]
    correlation = np.sin(2 * phases) * (c[:, u] * s[:, v] * np.cos(2 * right).prod(axis=2)
                                                    + s[:, u] * c[:, v] * np.cos(2 * left).prod(axis=2))
    correlation += .5 * s[:, u] * s[:, v] * (np.cos(2 * (left - right)).prod(axis=2)
                                            - np.cos(2 * (left + right)).prod(axis=2))
    return .5 * ((correlation - 1) * weights).sum(axis=1)


def validate_formula():
    import networkx as nx
    require(hashlib.sha256(Path(check_rows.__file__).read_bytes()).hexdigest() == CHECKER_SHA,
            "The independent checker differs from the approved byte copy")
    rng = np.random.default_rng(18423)
    maximum, count = 0.0, 0
    for n in range(2, 15):
        for trial in range(3):
            graph = nx.gnp_random_graph(n, .2 + trial * .3, seed=500 * n + trial)
            if not graph.number_of_edges():
                graph.add_edge(0, 1)
            edges = sorted(graph.edges())
            weights = rng.uniform(.5, 2, len(edges))
            row = {"n": n, "edges": edges, "w_sorted": weights.tolist()}
            points = rng.uniform(-3, 3, (5, len(edges) + n))
            fast = energy_batch(row, points)
            for point, energy in zip(points, fast):
                checked = check_rows.energy(n, edges, weights, point[:len(edges)], point[len(edges):])
                maximum = max(maximum, abs(energy - checked))
                count += 1
    require(maximum <= 1e-10, "Independent p=1 formula does not match the pinned checker")
    return {"points": count, "n_range": [2, 14], "maximum_error": maximum, "passed": True}


def audit():
    validation = validate_formula()
    corpus = Corpus()
    problems, original_problems, sources = [], [], {}
    counts = {"model_polish_visits": 0, "baseline_polishes": 0}
    maximum = 0.0

    def inspect(tag, graph_id, polishes, path, method):
        nonlocal maximum
        row = corpus.row(tag, graph_id)
        points = [point["angles_after"] for point in polishes]
        energies = energy_batch(row, points)
        for index, (point, energy) in enumerate(zip(polishes, energies)):
            error = float(energy - point["energy_after"])
            maximum = max(maximum, abs(error))
            hit = bool(energy <= row["floor"] + 1e-6)
            original_energy = point.get("optimizer_energy_after_before_numerical_repair", point["energy_after"])
            original_error = float(energy - original_energy)
            if abs(error) <= 1e-10 and abs(original_error) <= 1e-10 and hit == bool(point["hit"]):
                continue
            m = len(row["edges"])
            bounded = bounded_angles(point["angles_after"], row["w_sorted"], row["n"])
            checked = check_rows.energy(row["n"], row["edges"], row["w_sorted"], bounded[:m], bounded[m:])
            require(abs(checked - energy) <= 1e-10, "Flagged polish formula differs from the pinned checker")
            entry = {"path": path, "corpus": tag, "id": graph_id, "n": row["n"],
                             "method": method, "polish_index": index, "floor": row["floor"],
                             "stored_energy": point["energy_after"], "verified_energy": checked,
                             "error": checked - point["energy_after"], "original_hit": bool(point["hit"]),
                             "checked_hit": bool(checked <= row["floor"] + 1e-6),
                             "max_absolute_angle": max(map(abs, point["angles_after"])),
                             "angles_sha256": hashlib.sha256(json.dumps(point["angles_after"]).encode()).hexdigest()}
            if abs(error) > 1e-10 or hit != bool(point["hit"]):
                problems.append(entry)
            if abs(original_error) > 1e-10:
                original_problems.append(dict(entry, original_optimizer_energy=original_energy,
                                              original_optimizer_error=checked - original_energy))

    for path in sorted((FOLDER / "evaluations").glob("*.json")):
        raw = path.read_bytes()
        value = json.loads(raw)
        methods = {method: records for method, records in value.get("results", {}).items() if method in ["raw", "snap", "better"]}
        if not methods:
            continue
        relative = path.relative_to(RESULTS).as_posix()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        for method, records in methods.items():
            for record in records:
                counts["model_polish_visits"] += len(record["polishes"])
                inspect(value["corpus"], record["id"], record["polishes"], relative, method)
    print(json.dumps({"model_complete": True, "counts": counts, "flagged": len(problems)}), flush=True)
    for tag in ["val", "test_split", "large", "ian"]:
        for path in sorted((FOLDER / "baselines" / tag).glob("*.json")):
            raw = path.read_bytes()
            value = json.loads(raw)["baseline"]
            relative = path.relative_to(RESULTS).as_posix()
            sources[relative] = hashlib.sha256(raw).hexdigest()
            for method in ["random", "zero", "oracle"]:
                points = [value[method]] if method == "oracle" else value[method]
                counts["baseline_polishes"] += len(points)
                inspect("full" if tag in ["val", "test_split"] else tag, value["id"], points, relative, method)
    return {"created_epoch": time.time(), "formula_validation": validation, "checker_sha256": CHECKER_SHA,
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "counts": counts,
            "flagged_polish_visits": len(problems), "maximum_error": maximum,
            "original_optimizer_mismatch_visits": len(original_problems),
            "original_optimizer_mismatch_unique_points": len({(p["corpus"], p["id"], p["angles_sha256"]) for p in original_problems}),
            "maximum_original_optimizer_error": max((abs(p["original_optimizer_error"]) for p in original_problems), default=0),
            "changed_hit_decisions": sum(point["original_hit"] != point["checked_hit"] for point in problems),
            "problems": problems, "original_problems": original_problems,
            "input_sha256": sources, "new_optimizations": 0, "new_modal_compute": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.write:
        path = RESULTS / "preflight/all_polish_energy_audit.json"
        temporary = path.with_suffix(".audit.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({key: result[key] for key in ["formula_validation", "counts", "flagged_polish_visits", "maximum_error", "changed_hit_decisions", "original_optimizer_mismatch_visits", "original_optimizer_mismatch_unique_points", "maximum_original_optimizer_error"]}))


if __name__ == "__main__":
    main()
