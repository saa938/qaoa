"""Audit saved large-angle polishes without optimization or cloud submissions."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import mpmath as mp
import numpy as np

import check_rows
from report_floor_review import Corpus, FOLDER, RESULTS, CHECKER_SHA, require


def bounded_angles(point, weights, n):
    """Reduce exact saved binary64 inputs using an 80-digit value of pi."""
    m = len(weights)
    with mp.workdps(80):
        result = []
        for index, value in enumerate(point):
            scale = mp.mpf(float(weights[index])) if index < m else mp.mpf(1)
            phase = mp.mpf(float(value)) * scale
            reduced = (phase + mp.pi / 2) % mp.pi - mp.pi / 2
            result.append(float(reduced / scale))
    require(len(result) == m + n, "Invalid saved angle dimension")
    return np.asarray(result)


def audit():
    require(hashlib.sha256(Path(check_rows.__file__).read_bytes()).hexdigest() == CHECKER_SHA,
            "The independent checker differs from the approved byte copy")
    corpus = Corpus()
    cases, sources = {}, {}
    counts = {"model_polish_visits": 0, "baseline_polishes": 0}

    def inspect(tag, graph_id, point, source):
        angles = point["angles_after"]
        require(all(math.isfinite(value) for value in angles), "Non-finite saved angle")
        if max(map(abs, angles)) <= 1000:
            return
        original_energy = point.get("optimizer_energy_after_before_numerical_repair", point["energy_after"])
        key = (tag, graph_id, tuple(angles), original_energy)
        if key in cases:
            cases[key]["sources"].append(source)
            return
        row = corpus.row(tag, graph_id)
        m, n = len(row["edges"]), row["n"]
        weights = np.asarray(row["w_sorted"])
        bounded = bounded_angles(angles, weights, n)
        checked = check_rows.energy(n, row["edges"], weights, bounded[:m], bounded[m:])
        original = check_rows.energy(n, row["edges"], weights, angles[:m], angles[m:])
        cases[key] = {"corpus": tag, "id": graph_id, "n": n, "floor": row["floor"],
                      "original_optimizer_energy_after": original_energy, "current_energy_after": point["energy_after"],
                      "independently_checked_energy": checked, "unreduced_checker_energy": original,
                      "original_optimizer_energy_error": checked - original_energy,
                      "current_energy_error": checked - point["energy_after"],
                      "original_hit": bool(point["hit"]), "checked_hit": bool(checked <= row["floor"] + 1e-6),
                      "max_absolute_angle": max(map(abs, angles)), "original_angles": angles,
                      "bounded_angles": bounded.tolist(), "sources": [source]}

    for path in sorted((FOLDER / "evaluations").glob("*.json")):
        raw = path.read_bytes()
        value = json.loads(raw)
        if not any(method in value.get("results", {}) for method in ["raw", "snap", "better"]):
            continue
        sources[path.relative_to(RESULTS).as_posix()] = hashlib.sha256(raw).hexdigest()
        for method in ["raw", "snap", "better"]:
            for row in value.get("results", {}).get(method, []):
                for index, point in enumerate(row["polishes"]):
                    counts["model_polish_visits"] += 1
                    inspect(value["corpus"], row["id"], point,
                            {"path": path.relative_to(RESULTS).as_posix(), "method": method, "polish_index": index})
    for tag in ["val", "test_split", "large", "ian"]:
        for path in sorted((FOLDER / "baselines" / tag).glob("*.json")):
            raw = path.read_bytes()
            value = json.loads(raw)["baseline"]
            sources[path.relative_to(RESULTS).as_posix()] = hashlib.sha256(raw).hexdigest()
            for method in ["random", "zero", "oracle"]:
                points = [value[method]] if method == "oracle" else value[method]
                for index, point in enumerate(points):
                    counts["baseline_polishes"] += 1
                    inspect("full" if tag in ["val", "test_split"] else tag, value["id"], point,
                            {"path": path.relative_to(RESULTS).as_posix(), "method": method, "polish_index": index})
    rows = sorted(cases.values(), key=lambda row: abs(row["original_optimizer_energy_error"]), reverse=True)
    return {"captured_epoch": time.time(), "new_optimizations": 0, "new_modal_compute": False,
            "counts": counts, "large_angle_unique_polishes": len(rows),
            "original_optimizer_energy_errors_above_tolerance": sum(abs(row["original_optimizer_energy_error"]) > 1e-10 for row in rows),
            "current_energy_errors_above_tolerance": sum(abs(row["current_energy_error"]) > 1e-10 for row in rows),
            "hit_decisions_changed": sum(row["original_hit"] != row["checked_hit"] for row in rows),
            "max_original_optimizer_energy_error": max((abs(row["original_optimizer_energy_error"]) for row in rows), default=0),
            "max_current_energy_error": max((abs(row["current_energy_error"]) for row in rows), default=0),
            "period_reduction": "Exact saved binary64 inputs reduced modulo 80-digit pi in weighted edge phases and node angles; pinned checker evaluates bounded representatives.",
            "checker_sha256": CHECKER_SHA, "cases": rows, "input_sha256": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.write:
        path = RESULTS / "preflight/large_angle_energy_audit.json"
        temporary = path.with_suffix(".audit.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    print(json.dumps({key: result[key] for key in ["counts", "large_angle_unique_polishes", "original_optimizer_energy_errors_above_tolerance", "current_energy_errors_above_tolerance", "hit_decisions_changed", "max_original_optimizer_energy_error", "max_current_energy_error"]}))


if __name__ == "__main__":
    main()
