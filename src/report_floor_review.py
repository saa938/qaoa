"""Review existing evaluation evidence without submitting any additional solves."""

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

import check_rows


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FOLDER = RESULTS / "gnn"
TOLERANCE = 1e-6
CHECKER_SHA = "b5ab974f51775c3677a172c7678738fbbe86baee568bf903477101aa013d94a2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


class Corpus:
    def __init__(self):
        self.indices = {}
        self.rows = {}

    def row(self, corpus, graph_id):
        key = (corpus, graph_id)
        if key not in self.rows:
            if corpus not in self.indices:
                self.indices[corpus] = read(FOLDER / "corpus" / (corpus + "_index.json"))
            offset, length, sha = self.indices[corpus][str(graph_id)]
            with (FOLDER / "corpus" / (corpus + ".jsonl")).open("rb") as stream:
                stream.seek(offset)
                raw = stream.read(length)
            require(digest(raw) == sha, "Corpus record checksum differs")
            value = json.loads(raw)
            require(value["id"] == graph_id, "Corpus record id differs")
            self.rows[key] = value
        return self.rows[key]


def build(require_complete=False):
    require(digest((ROOT / "src/check_rows.py").read_bytes()) == CHECKER_SHA,
            "Independent checker differs from the approved byte copy")
    if require_complete:
        for filename in ["continuation_status.json", "learning_curve_continuation_status.json"]:
            require(read(RESULTS / "preflight" / filename)["stage"] == "complete",
                    "Finish the existing coordinators before the final review")
        names = [model + "_s42_val" for model in ["A", "B", "Bcls", "C03", "C10", "D", "E"]]
        names += ["final_s%d_%s" % (seed, tag) for seed in [42, 43, 44]
                  for tag in ["test_split", "large", "ian"]]
        names += ["curve%d_s%d_test" % (percent, seed) for percent in [25, 50] for seed in [42, 43, 44]]
        names += ["paper_ian"]
        require(all((FOLDER / "evaluations" / (name + ".json")).exists() for name in names),
                "A required completed model evaluation is missing")

    previous = read(RESULTS / "preflight/reference_floor_review.json")
    first_sources = {(row["corpus"], row["id"]): row.get("discovery_source", row["source"])
                     for row in previous["findings"]}
    baseline_path = RESULTS / "preflight/baseline_reference_floor_review.json"
    baseline_raw = baseline_path.read_bytes()
    baseline = json.loads(baseline_raw)
    require(baseline["complete_expected_baseline_cohorts"] and baseline["reviewed_graphs"] == 8935,
            "The complete baseline review is missing")
    corpus = Corpus()
    best = {}
    below_counts = Counter()
    sources = {}
    visits = Counter()

    def consider(tag, set_name, graph_id, polish, source):
        row = corpus.row(tag, graph_id)
        energy = polish["energy_after"]
        require(math.isfinite(energy), "Non-finite polished energy")
        if energy >= row["floor"] - TOLERANCE:
            return
        below_counts[source] += 1
        key = (tag, graph_id)
        if key in best and best[key]["energy_after"] <= energy:
            return
        point = np.asarray(polish["angles_after"], dtype=float)
        weights = np.asarray(row["w_sorted"], dtype=float)
        m = len(row["edges"])
        require(len(point) == m + row["n"] and np.isfinite(point).all(), "Invalid candidate angles")
        checked_point = point
        if np.max(np.abs(point)) > 1000:
            import mpmath as mp
            scales = np.r_[weights, np.ones(row["n"])]
            with mp.workdps(80):
                checked_point = np.asarray([
                    float(((mp.mpf(float(value)) * mp.mpf(float(scale)) + mp.pi / 2) % mp.pi - mp.pi / 2)
                          / mp.mpf(float(scale))) for value, scale in zip(point, scales)])
        checked = float(check_rows.energy(row["n"], row["edges"], weights, checked_point[:m], checked_point[m:]))
        pick = np.asarray(row["pick"], dtype=float)
        reference = float(check_rows.energy(row["n"], row["edges"], weights, pick[:m], pick[m:]))
        require(abs(checked - energy) <= 1e-10, "Lower-energy candidate fails the independent check")
        require(abs(reference - row["floor"]) <= TOLERANCE, "Reference label does not reproduce its target")
        require(checked >= -row["max_cut"] - TOLERANCE, "Candidate violates the exact Max-Cut bound")
        best[key] = {
            "corpus": tag, "set_name": set_name, "id": graph_id, "n": row["n"], "type": row["type"],
            "floor": row["floor"], "energy_after": checked, "improvement": row["floor"] - checked,
            "source": source, "discovery_source": first_sources.get(key, source),
            "angles_after": point.tolist(), "independent_energy_error": abs(checked - energy),
            "independent_reference_pick_energy": reference, "reference_pick_minus_floor": reference - row["floor"],
        }

    for row in baseline["findings"]:
        consider(row["corpus"], row["set_name"], row["id"], row, "saved " + row["method"] + " baseline")

    for path in sorted((FOLDER / "evaluations").glob("*.json")):
        raw = path.read_bytes()
        value = json.loads(raw)
        methods = {name: records for name, records in value.get("results", {}).items()
                   if name in ["raw", "snap", "better"]}
        if not methods:
            continue
        expected = value["expected_ids"]
        require(len(set(expected)) == len(expected), "Duplicate expected ids in model evaluation")
        sources[path.name] = digest(raw)
        for method, records in methods.items():
            require(len(records) == len(expected) and {row["id"] for row in records} == set(expected),
                    "Incomplete model evaluation: " + path.name + "/" + method)
            for row in records:
                for polish in row["polishes"]:
                    visits[method] += 1
                    consider(value["corpus"], value["set_name"], row["id"], polish, path.stem + "/" + method)

    findings = [best[key] for key in sorted(best)]
    training = set(read(FOLDER / "splits.json")["ids"]["train"])
    for row in findings:
        row["belongs_to_training_split"] = row["corpus"] == "full" and row["id"] in training
    return {
        "partial_model_review": not require_complete,
        "scope": "All complete raw, snapped and better-strategy evaluation aggregates present at capture; every independently executed strategy is reviewed.",
        "model_aggregates_reviewed": len(sources), "model_polish_visits": dict(visits),
        "baseline_expected_graphs": 8935, "baseline_reviewed_graphs": baseline["reviewed_graphs"],
        "baseline_polishes": sum(method["polishes"] for group in baseline["counts"].values() for method in group["methods"].values()),
        "known_counterexample_graphs": len(findings), "findings": findings,
        "below_reference_polish_visits_by_source": dict(below_counts),
        "interpretation": "Each listed original label is not on the true variational floor: an independently checked feasible point has lower energy. Neither improved points nor other reference labels are thereby certified globally optimal. Unequal searches do not estimate a population error rate.",
        "primary_targets_changed": False, "training_population_changed": False, "new_modal_solves": 0,
        "source_proofs": {"baseline": digest(baseline_raw), "checker": CHECKER_SHA, "model_evaluations": sources},
        "captured_epoch": time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = build(args.require_complete)
    if args.write:
        path = RESULTS / "preflight/reference_floor_review.json"
        temporary = path.with_suffix(".review.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    print(json.dumps({key: result[key] for key in ["partial_model_review", "model_aggregates_reviewed", "model_polish_visits", "known_counterexample_graphs"]}))


if __name__ == "__main__":
    main()
