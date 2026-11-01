"""Summarize saved evaluations for the requested final-report additions.

This script reads local evidence only. It never submits compute, changes a target,
or imports the active Modal runner. Rerun after all final seeds finish, then use
--append-report after the regular report generator has finished.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "gnn"
EXPECTED_SEEDS = (42, 43, 44)
START_MARKER = "<!-- BEGIN REQUESTED COMPARISON ADDITIONS -->"
END_MARKER = "<!-- END REQUESTED COMPARISON ADDITIONS -->"
PINS = {
    "src/modal_app.py": "816e648cb20a9e657f9f2db13a4d475716a014cf51c92153fc880832577f68e6",
    "src/gnn.py": "46eb804c6ee2918c1a82f39de5af6e830810eed460ed0b7ac228c49e8bbbb4b3",
    "src/make_dataset.py": "0cdb10df3c553dc24f12ac71de184a0fddd2f3d9728da2a78b58d0474cc8b197",
    "src/check_rows.py": "b5ab974f51775c3677a172c7678738fbbe86baee568bf903477101aa013d94a2",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        raw = path.read_bytes()
        self.hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)


def index_records(records, expected, label):
    indexed = {row["id"]: row for row in records}
    require(len(indexed) == len(records), label + ": duplicate ids")
    require(set(indexed) == set(expected), label + ": incomplete expected-id manifest")
    return indexed


def cost_summary(model, random, ids):
    """Sum charged model work plus estimated random fallback on every failure."""
    ratios, charged, fallback, reference = [], 0.0, 0.0, 0.0
    failures = zero_hit = 0
    for graph_id in ids:
        row, baseline = model[graph_id], random[graph_id]
        random_cost = baseline["expected_cost"]
        extra = 0.0 if row["hit"] else random_cost
        expected = row["circuit_evaluations"] + extra
        require(math.isclose(expected, row["expected_cost"], rel_tol=1e-12, abs_tol=1e-7),
                "Fallback accounting disagrees with saved expected_cost")
        require(math.isclose(expected / random_cost, row["cost_ratio"], rel_tol=1e-12),
                "Per-graph cost ratio disagrees with saved evidence")
        charged += row["circuit_evaluations"]
        fallback += extra
        reference += random_cost
        failures += not row["hit"]
        zero_hit += bool(baseline["zero_random_hit_lower_bound"])
        ratios.append(expected / random_cost)
    return {
        "graphs": len(ids), "failed_warm_starts": failures,
        "executed_model_circuit_evaluations": charged,
        "executed_optimizer_joint_calls": sum(model[i]["nfev"] for i in ids),
        "executed_ranking_energy_calls": sum(model[i]["ranking_evaluations"] for i in ids),
        "estimated_random_fallback_circuit_evaluations": fallback,
        "total_model_plus_fallback": charged + fallback,
        "total_random_reference": reference,
        "ratio_of_totals": (charged + fallback) / reference,
        "geometric_mean_cost_ratio": math.exp(statistics.mean(map(math.log, ratios))),
        "median_cost_ratio": statistics.median(ratios),
        "random_zero_hit_graphs": zero_hit,
    }


def baselines(evidence, tag, expected, aggregate):
    rows = {}
    for graph_id in expected:
        wrapper = evidence.read(FOLDER / "baselines" / tag / (str(graph_id) + ".json"))
        row = wrapper["baseline"]
        require(row["id"] == graph_id, "Baseline id mismatch")
        require(wrapper["restarts"] == len(row["random"]), "Random draw count mismatch")
        require(len(row["random"]) >= 4, "Matched four-start comparison needs four draws")
        hits = sum(bool(run["hit"]) for run in row["random"])
        total = sum(run["circuit_evaluations"] for run in row["random"])
        reference = total / hits if hits else total
        require(math.isclose(reference, aggregate[graph_id]["expected_cost"], rel_tol=1e-12),
                "Baseline aggregate differs from individual random draws")
        rows[graph_id] = row
    return rows


def matched_by_n(model_seeds, baseline_draws, random, expected):
    groups = defaultdict(list)
    first = next(iter(model_seeds.values()))
    for graph_id in expected:
        groups[first[graph_id]["n"]].append(graph_id)
    output = []
    for n, ids in sorted(groups.items()):
        seed_rates = {}
        for seed, model in model_seeds.items():
            top_hits = all_hits = 0
            for graph_id in ids:
                row = model[graph_id]
                runs = row["polishes"]
                require(1 <= len(runs) <= 4, "D must use up to four candidates")
                require(all(run["chosen"] == "snap" for run in runs), "Unexpected strategy")
                require(not any(run["hit"] for run in runs[:-1]), "Polishing did not stop at first hit")
                require(row["hit"] or len(runs) == 4, "Failed graph is missing candidate attempts")
                require(bool(row["hit"]) == any(run["hit"] for run in runs), "Model hit mismatch")
                require(row["ranking_evaluations"] == 4, "Ranking must include all four predictions")
                require(row["circuit_evaluations"] == 4 + sum(run["circuit_evaluations"] for run in runs),
                        "Charged model work omits a candidate or ranking evaluation")
                top_hits += bool(runs[0]["hit"])
                all_hits += bool(row["hit"])
            seed_rates[str(seed)] = {"top_one_hits": top_hits, "up_to_four_hits": all_hits,
                                     "top_one_hit_rate": top_hits / len(ids),
                                     "up_to_four_hit_rate": all_hits / len(ids)}
        one = sum(bool(baseline_draws[i]["random"][0]["hit"]) for i in ids)
        four = sum(any(run["hit"] for run in baseline_draws[i]["random"][:4]) for i in ids)
        # Also estimate single/four-start rates from all saved draws, using disjoint
        # consecutive groups of four. The fixed first draw/block remains primary.
        all_one, all_four = [], []
        for graph_id in ids:
            draws = baseline_draws[graph_id]["random"]
            all_one.append(statistics.mean(bool(run["hit"]) for run in draws))
            blocks = [draws[k:k + 4] for k in range(0, len(draws) - 3, 4)]
            all_four.append(statistics.mean(any(run["hit"] for run in block) for block in blocks))
        output.append({
            "n": n, "graphs": len(ids), "seeds": seed_rates,
            "model_top_one_hit_rate": statistics.mean(s["top_one_hit_rate"] for s in seed_rates.values()),
            "model_up_to_four_hit_rate": statistics.mean(s["up_to_four_hit_rate"] for s in seed_rates.values()),
            "random_first_one_hits": one, "random_first_four_hits": four,
            "random_first_one_hit_rate": one / len(ids), "random_first_four_hit_rate": four / len(ids),
            "random_all_draws_single_hit_rate": statistics.mean(all_one),
            "random_disjoint_four_blocks_hit_rate": statistics.mean(all_four),
        })
    return output


def mean_seed_model(models, expected):
    result = {}
    for graph_id in expected:
        rows = [model[graph_id] for model in models.values()]
        # hit is averaged only for descriptive output; expected_cost must be
        # averaged directly, never rederived from a binary version of this hit.
        result[graph_id] = {key: statistics.mean(row[key] for row in rows)
                            for key in ["expected_cost", "cost_ratio", "circuit_evaluations", "hit", "nfev", "ranking_evaluations"]}
    return result


def averaged_cost_summary(models, random, expected):
    individual = {str(seed): cost_summary(model, random, expected) for seed, model in models.items()}
    combined = mean_seed_model(models, expected)
    ratios = [combined[i]["cost_ratio"] for i in expected]
    charged = sum(combined[i]["circuit_evaluations"] for i in expected)
    total = sum(combined[i]["expected_cost"] for i in expected)
    reference = sum(random[i]["expected_cost"] for i in expected)
    numerator = np.asarray([combined[i]["expected_cost"] for i in expected])
    denominator = np.asarray([random[i]["expected_cost"] for i in expected])
    total_interval = paired_sum_ratio_interval(numerator, denominator)
    return {"per_seed": individual, "seed_averaged": {
        "graphs": len(expected), "seeds": len(models),
        "executed_model_circuit_evaluations": charged,
        "executed_optimizer_joint_calls": sum(combined[i]["nfev"] for i in expected),
        "executed_ranking_energy_calls": sum(combined[i]["ranking_evaluations"] for i in expected),
        "estimated_random_fallback_circuit_evaluations": total - charged,
        "total_model_plus_fallback": total, "total_random_reference": reference,
        "ratio_of_totals": total / reference,
        "ratio_of_totals_ci95": total_interval,
        "geometric_mean_cost_ratio": math.exp(statistics.mean(map(math.log, ratios))),
        "median_cost_ratio": statistics.median(ratios),
        "random_zero_hit_graphs": sum(bool(random[i]["zero_random_hit_lower_bound"]) for i in expected),
    }}


def paired_sum_ratio_interval(numerator, denominator):
    """Resample whole matched graphs, conditional on saved starts/model seeds."""
    require(len(numerator) == len(denominator) and len(numerator) > 0, "Invalid paired cost arrays")
    rng = np.random.default_rng(42)
    values = []
    for _ in range(1000):
        draw = rng.integers(len(numerator), size=len(numerator))
        values.append(float(numerator[draw].sum() / denominator[draw].sum()))
    return np.quantile(values, [.025, .975]).tolist()


def learning_curve_comparison(evidence, full_models, random, expected):
    """Reuse completed curve evaluations; do not infer an unfinished fit's result."""
    output, graph_costs = {}, {}
    subsets_path = FOLDER / "learning_curve_subsets.json"
    if not subsets_path.exists():
        return {"status": "pending existing approved fits", "fractions": output}
    subsets = evidence.read(subsets_path)["ids"]
    require(set(subsets["25"]) <= set(subsets["50"]) <= set(subsets["100"]),
            "Learning-curve training subsets are not nested")
    for percent in [25, 50, 100]:
        models = dict(full_models) if percent == 100 else {}
        if percent != 100:
            for seed in EXPECTED_SEEDS:
                path = FOLDER / "evaluations" / ("curve%d_s%d_test.json" % (percent, seed))
                if path.exists():
                    value = evidence.read(path)
                    require(value["expected_ids"] == expected, "Learning curve changes held-out ids")
                    models[seed] = index_records(value["results"]["snap"], expected, "Learning curve")
        item = {"train_graphs": len(subsets[str(percent)]), "seeds_available": list(models),
                "seeds_pending": [seed for seed in EXPECTED_SEEDS if seed not in models],
                "reuses_final_fits": percent == 100}
        if models:
            item["costs"] = averaged_cost_summary(models, random, expected)
            item["top_one_hit_rate"] = statistics.mean(bool(model[i]["polishes"][0]["hit"])
                                                       for model in models.values() for i in expected)
            item["up_to_four_hit_rate"] = statistics.mean(bool(model[i]["hit"])
                                                          for model in models.values() for i in expected)
            graph_costs[str(percent)] = np.asarray([statistics.mean(model[i]["expected_cost"] for model in models.values())
                                                    for i in expected])
        output[str(percent)] = item
    complete = all(not row["seeds_pending"] for row in output.values())
    paired = {}
    if complete:
        for earlier, later in [("25", "50"), ("50", "100"), ("25", "100")]:
            paired[earlier + "_to_" + later] = {
                "later_over_earlier_summed_cost": float(graph_costs[later].sum() / graph_costs[earlier].sum()),
                "ci95": paired_sum_ratio_interval(graph_costs[later], graph_costs[earlier])}
    return {"status": "complete" if complete else "pending existing approved fits", "fractions": output,
            "paired_total_cost_changes": paired}


def angle_counts(row, phase):
    """Same coordinate tolerance/period convention as gnn.label_metrics."""
    m = len(row["edges"])
    target = [a * w for a, w in zip(row["pick"][:m], row["w_sorted"])] + row["pick"][m:]
    step = math.pi / 8
    snapped = [round(value / step) * step for value in phase]
    correct = [abs((a - b + math.pi / 2) % math.pi - math.pi / 2) <= 1e-3
               for a, b in zip(snapped, target)]
    require(len(phase) == len(target), "Ian angle dimension mismatch")
    return {"edge_correct": sum(correct[:m]), "edges": m,
            "node_correct": sum(correct[m:]), "nodes": row["n"]}


def ian_comparison(evidence, models, expected):
    path = FOLDER / "corpus" / "ian.jsonl"
    evidence.hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    corpus = index_records([json.loads(line) for line in path.read_text().splitlines()], expected, "Ian corpus")
    output = {"graphs": len(expected), "seeds_available": list(models), "rows": []}
    old = {"name": "Old paper recorded predictions", "hypotheses": 1, "strategy": "raw",
           "initial_energy_over_reference": statistics.mean(corpus[i]["paper_csv_energy"] / corpus[i]["floor"] for i in expected),
           "initial_expected_cut_over_maxcut": statistics.mean(-corpus[i]["paper_csv_energy"] / corpus[i]["max_cut"] for i in expected)}
    paper_angles = []
    for i in expected:
        row = corpus[i]
        m = len(row["edges"])
        point = row["paper_prediction_raw"]
        phase = [a * w for a, w in zip(point[:m], row["w_sorted"])] + point[m:]
        paper_angles.append(angle_counts(row, phase))
    for label, key, total in [("edge_accuracy", "edge_correct", "edges"), ("node_accuracy", "node_correct", "nodes")]:
        old[label] = sum(row[key] for row in paper_angles) / sum(row[total] for row in paper_angles)
    paper_path = FOLDER / "evaluations" / "paper_ian.json"
    if paper_path.exists():
        paper = evidence.read(paper_path)
        records = index_records(paper["results"]["raw"], expected, "Old paper Ian")
        require(paper["expected_ids"] == expected, "Old paper comparison changes Ian ids")
        for i in expected:
            require(abs(records[i]["energy_before"] - corpus[i]["paper_csv_energy"]) <= 1e-12,
                    "Old paper energies differ from recorded CSV")
        random = index_records(paper["results"]["random"], expected, "Old paper Ian random")
        old.update({"post_refinement_status": "complete", "summary": paper["summaries"]["raw"]["metrics"],
                    "costs": cost_summary(records, random, expected)})
    else:
        old["post_refinement_status"] = "pending existing main-thread paper_ian evaluation; no substitute solves launched"
    output["rows"].append(old)
    for seed, records in models.items():
        counts = [angle_counts(corpus[i], records[i]["polishes"][0]["candidate_phase"]) for i in expected]
        output["rows"].append({
            "name": "New D seed " + str(seed), "seed": seed, "hypotheses": 4, "strategy": "snap",
            "initial_energy_over_reference": statistics.mean(records[i]["energy_before"] / corpus[i]["floor"] for i in expected),
            "initial_expected_cut_over_maxcut": statistics.mean(-records[i]["energy_before"] / corpus[i]["max_cut"] for i in expected),
            "edge_accuracy": sum(row["edge_correct"] for row in counts) / sum(row["edges"] for row in counts),
            "node_accuracy": sum(row["node_correct"] for row in counts) / sum(row["nodes"] for row in counts),
            "top_one_hit_rate": statistics.mean(records[i]["polishes"][0]["hit"] for i in expected),
            "up_to_four_hit_rate": statistics.mean(records[i]["hit"] for i in expected),
        })
    output["limits"] = [
        "Old paper uses its recorded predictions on exactly Ian's twenty rows; A in the bake-off is a separately retrained architecture.",
        "Ian graphs are unweighted eight-node graphs outside the weighted training distribution; this is a transfer check.",
        "Coordinate accuracy first snaps to pi/8 for both methods, matching the existing metric; it is not raw continuous-angle error.",
        "New D ranks four hypotheses; the old paper has one. Top-one and up-to-four outcomes must be distinguished.",
    ]
    return output


def build():
    evidence = Evidence()
    source_hashes = {name: digest(ROOT / name) for name in PINS}
    require(all(source_hashes[name] == value for name, value in PINS.items() if name != "src/modal_app.py"),
            "Active scientific sources changed: review the comparison conventions before regenerating")
    if source_hashes["src/modal_app.py"] != PINS["src/modal_app.py"]:
        snapshot = FOLDER / "source_snapshots" / ("modal_app_" + PINS["src/modal_app.py"] + ".py")
        require(snapshot.exists() and digest(snapshot) == PINS["src/modal_app.py"],
                "The frozen active-runner snapshot is missing or changed")
    splits = evidence.read(FOLDER / "splits.json")
    bakeoff = evidence.read(FOLDER / "bakeoff.json")
    strategy = bakeoff["strategy"]
    require(strategy == "snap", "Review the comparison after a strategy change")
    expected_sets = {"test_split": splits["ids"]["test"], "ian": list(range(20))}
    large_index = evidence.read(FOLDER / "corpus" / "large_index.json")
    # The complete saved baseline manifest fixes usable large-test ids, then the
    # corpus index verifies each one exists. Rejected saved rows are not silently
    # substituted into this already frozen usable evaluation population.
    large_baseline = evidence.read(FOLDER / "evaluations" / "large_baselines.json")
    expected_sets["large"] = large_baseline["expected_ids"]
    require(len(expected_sets["test_split"]) == 6331 and len(expected_sets["large"]) == 584,
            "Frozen test-population sizes changed")
    require(bool(large_index), "Large corpus index is empty")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "in progress: remaining approved fits, evaluations or final audits are pending",
        "strategy": strategy, "seeds_expected": list(EXPECTED_SEEDS),
        "new_modal_compute": False, "incremental_paid_compute_usd": 0,
        "missing_training_graphs_generated": 0, "spend_stop_usd": 280,
        "training_comparison_plan_unchanged": [25, 50, 100],
        "scientific_source_hashes": source_hashes,
        "matched_protocol": {
            "model_top_one": "Only the first (lowest initial energy) of four ranked predictions is polished for this derived hit statistic.",
            "random_one": "The first saved random draw, fixed before looking at hits.",
            "model_four": "Any success among up to four candidates, stopping at the first success; every failure has all four attempts saved.",
            "random_four": "Any success among the first four saved random draws, with no ranking or post-hoc selection.",
            "sensitivity": "Also report means over all random draws and over disjoint consecutive four-draw blocks; trailing incomplete blocks are unused.",
            "denominator": "All expected usable test ids per n; complete seed manifests are required. Missing seeds stay explicitly pending.",
        },
        "cost_convention": {
            "units": "Circuit evaluations under the existing 2*d+1 charge per joint energy/gradient call, plus ranking evaluations.",
            "execution_vs_charge": "The statevector simulator computes gradients by adjoint propagation. Circuit charges are parameter-shift-equivalent units assigned to actual optimizer calls, not literally executed shifted statevectors or hardware circuits. Joint-call and ranking-call totals are reported separately.",
            "model": "Sum every executed candidate attempt, including failures and ranking, plus per-graph estimated random fallback on warm-start failure.",
            "random": "Per graph, sum all saved random-start charges divided by its number of successes; sum these reference costs across the identical graph list.",
            "zero_hit": "When no saved random draw succeeds, use the existing lower-bound reference equal to the sum of all draw charges. This is not an exact expected cost.",
            "fallback": "Fallback is estimated from saved independent random draws. It was not physically executed after each failed model attempt.",
            "ratio_of_totals": "sum(model attempts + estimated fallback) / sum(random reference); unlike geometric mean, this weights graphs by their random reference cost.",
            "multi_seed": "Compute costs per seed and also average each graph's costs across available seeds before summing. Shared random baselines are not new independent draws.",
        },
        "sets": {},
    }
    saved_sets = {}
    for tag in ["test_split", "large", "ian"]:
        expected = expected_sets[tag]
        aggregate = evidence.read(FOLDER / "evaluations" / (tag + "_baselines.json"))
        require(aggregate["expected_ids"] == expected, tag + ": baseline ids changed")
        random = index_records(aggregate["results"]["random"], expected, tag + " random")
        model_seeds = {}
        all_strategies = {method: {} for method in ["raw", "snap", "better"]}
        for seed in EXPECTED_SEEDS:
            path = FOLDER / "evaluations" / ("final_s%d_%s.json" % (seed, tag))
            if path.exists():
                evaluation = evidence.read(path)
                require(evaluation["expected_ids"] == expected, "Model manifest differs from frozen population")
                model_seeds[seed] = index_records(evaluation["results"][strategy], expected, "Final model")
                for method in all_strategies:
                    all_strategies[method][seed] = index_records(evaluation["results"][method], expected, "Final " + method)
        require(model_seeds, tag + ": no completed model seed is available")
        draws = baselines(evidence, tag, expected, random)
        saved_sets[tag] = (model_seeds, random, expected)
        item = {"graphs": len(expected), "seeds_available": list(model_seeds),
                "seeds_pending": [seed for seed in EXPECTED_SEEDS if seed not in model_seeds],
                "matched_hit_rates_by_n": matched_by_n(model_seeds, draws, random, expected),
                "costs": averaged_cost_summary(model_seeds, random, expected),
                "all_postprocessing_strategies": {
                    method: {"costs": averaged_cost_summary(models, random, expected),
                             "hit_rate": statistics.mean(bool(model[i]["hit"]) for model in models.values() for i in expected)}
                    for method, models in all_strategies.items()},
                "all_saved_random_draws": {"starts": sum(len(row["random"]) for row in draws.values()),
                    "optimizer_joint_calls": sum(run["nfev"] for row in draws.values() for run in row["random"]),
                    "actual_circuit_evaluations": sum(run["circuit_evaluations"] for row in draws.values() for run in row["random"])}}
        if tag == "ian":
            result["ian_old_vs_new"] = ian_comparison(evidence, model_seeds, expected)
        result["sets"][tag] = item
    # The combined test population contains regular and larger weighted tests,
    # without treating Ian's separate unweighted transfer set as training-domain data.
    common_seeds = sorted(set(saved_sets["test_split"][0]) & set(saved_sets["large"][0]))
    combined_models = {seed: {} for seed in common_seeds}
    combined_random, combined_ids = {}, []
    for tag in ["test_split", "large"]:
        models, random, expected = saved_sets[tag]
        for graph_id in expected:
            key = (tag, graph_id)
            combined_ids.append(key)
            combined_random[key] = random[graph_id]
            for seed in common_seeds:
                combined_models[seed][key] = models[seed][graph_id]
    result["sets"]["all_weighted_test_graphs"] = {
        "graphs": len(combined_ids), "seeds_available": common_seeds,
        "seeds_pending": [seed for seed in EXPECTED_SEEDS if seed not in common_seeds],
        "matched_hit_rates_by_n": [],
        "costs": averaged_cost_summary(combined_models, combined_random, combined_ids),
    }
    result["learning_curve_comparison"] = learning_curve_comparison(evidence, *saved_sets["test_split"])
    validation_ids = splits["ids"]["val_selection"]
    require(len(validation_ids) == 2000, "Bake-off requires the same fixed 2000 validation ids")
    result["bakeoff"] = []
    descriptions = {
        "A": "Paper architecture, retrained on the new labels; one hypothesis",
        "B": "GPS regression with labels; one hypothesis",
        "Bcls": "GPS classification with labels; one hypothesis",
        "C03": "Selected base: 20 label epochs then 20 energy fine-tuning epochs, weight 0.3",
        "C10": "Selected base: 20 label epochs then 20 energy fine-tuning epochs, weight 1.0",
        "D": "GPS classification, four hypotheses, trained from scratch",
        "E": "GPS energy-only regression, no label loss; one hypothesis",
    }
    for name in ["A", "B", "Bcls", "C03", "C10", "D", "E"]:
        evaluation = evidence.read(FOLDER / "evaluations" / (name + "_s42_val.json"))
        require(evaluation["expected_ids"] == validation_ids, "Bake-off validation subset mismatch")
        selected = bakeoff["strategies"][name]["chosen"]
        model = index_records(evaluation["results"][selected], validation_ids, name)
        random = index_records(evaluation["results"]["random"], validation_ids, name + " random")
        metrics = evaluation["summaries"][selected]["metrics"]
        costs = cost_summary(model, random, validation_ids)
        require(math.isclose(costs["geometric_mean_cost_ratio"], metrics["geometric_mean_cost_ratio"]["value"], rel_tol=1e-12),
                "Bake-off geometric mean does not reproduce saved summary")
        result["bakeoff"].append({"name": name, "description": descriptions[name], "seed": 42,
                                 "epochs": 40, "strategy": selected, "metrics": metrics, "costs": costs})
    result["matched_label_ablation"] = evidence.read(FOLDER / "label_ablation.json")
    result["classifier_inference_precision"] = evidence.read(ROOT / "results/preflight/classifier_inference_precision.json")
    numerical_audit = ROOT / "results/preflight/all_polish_energy_audit.json"
    if numerical_audit.exists():
        result["all_polish_energy_audit"] = evidence.read(numerical_audit)
    for filename, key in [("billing_summary.json", "billing"), ("learning_curve.json", "learning_curve")]:
        path = FOLDER / filename
        if path.exists():
            result[key] = evidence.read(path)
    review = evidence.read(ROOT / "results/preflight/reference_floor_review.json")
    result["known_reference_counterexamples"] = {
        "partial_model_review": review["partial_model_review"],
        "labels_are_not_on_true_floor": True,
        "interpretation": "Each listed reference label is provably not on the true variational energy floor: an independently checked feasible point has lower energy. The improved point is not thereby certified globally optimal.",
        "targets_unchanged": True,
        "cases": [{key: row.get(key) for key in ["corpus", "set_name", "id", "n", "floor", "energy_after", "improvement", "source", "discovery_source", "belongs_to_training_split", "independent_energy_error"]}
                  for row in review["findings"]],
        "attribution": "The first four known graphs comprise three model-start discoveries and one saved random-baseline discovery; the complete review preserves discovery_source separately from the best saved point.",
    }
    result["input_sha256"] = evidence.hashes
    result["all_final_seed_evaluations_present"] = all(not item["seeds_pending"] for item in result["sets"].values())
    result["paper_ian_after_refinement_present"] = (FOLDER / "evaluations/paper_ian.json").exists()
    audit_hashes = result.get("all_polish_energy_audit", {}).get("input_sha256", {})
    model_paths = [path for path in (FOLDER / "evaluations").glob("*.json") if not path.stem.endswith("_baselines")]
    result["numerical_audit_covers_current_evaluations"] = bool(model_paths) and all(
        audit_hashes.get(path.relative_to(ROOT / "results").as_posix()) == digest(path) for path in model_paths)
    if (result["all_final_seed_evaluations_present"] and result["paper_ian_after_refinement_present"]
            and result["learning_curve_comparison"]["status"] == "complete" and not review["partial_model_review"]
            and result.get("all_polish_energy_audit", {}).get("flagged_polish_visits") == 0
            and result["all_polish_energy_audit"]["changed_hit_decisions"] == 0
            and result["numerical_audit_covers_current_evaluations"]):
        result["status"] = "complete"
    require(source_hashes == {name: digest(ROOT / name) for name in PINS}, "Scientific source changed during reporting")
    return result


def render(result):
    percent = lambda value: "%.2f%%" % (100 * value)
    lines = [START_MARKER, "", "## Additional comparisons requested before the final report", "",
             "Evidence snapshot: " + result["created_utc"] + ". Status: " + result["status"] + ".",
             ("These additions use saved evaluations only and launch no additional compute." if result["status"] == "complete" else
              "Regenerate these additions after the remaining approved fits and evaluations finish. This reporting script launches no compute."),
             "", "### Matched numbers of optimizer starts", "",
             "The model column uses the selected snapped strategy. Top one means the lowest-initial-energy prediction among four ranked hypotheses, followed by one polish. Up to four stops at the first successful polish. Random columns use the fixed first draw and first four draws, without choosing a favorable block. Thus optimizer-start counts match; model ranking and inference remain additional work, included in the cost accounting. Values average completed model seeds within each graph. Random baselines are shared across seeds.",
             "", "| Set | n | Graphs | Model seeds | Model top one | Random one | Model up to four | Random best of four |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for tag, item in result["sets"].items():
        for row in item["matched_hit_rates_by_n"]:
            lines.append("| %s | %d | %d | %s | %s | %s | %s | %s |" %
                         (tag, row["n"], row["graphs"], ", ".join(map(str, item["seeds_available"])),
                          percent(row["model_top_one_hit_rate"]), percent(row["random_first_one_hit_rate"]),
                          percent(row["model_up_to_four_hit_rate"]), percent(row["random_first_four_hit_rate"])))
    lines += ["", "The JSON also reports sensitivity estimates using all saved random draws for single-start success, and disjoint consecutive four-draw blocks for four-start success. Neither estimate uses 1-(1-p)^4 or assumes identical hit probability across graphs. All rates use the full frozen usable test-id population; pending model seeds are named explicitly.",
              "", "### Total circuit-evaluation accounting", "",
              "Each row sums across every graph in its set. Charges from executed attempts include ranking, successful polishes and failed polishes. The simulator uses adjoint gradients; multiplying each actual joint optimizer call by 2d+1 produces the prescribed parameter-shift-equivalent circuit charge. These are not literal counts of shifted statevectors or hardware circuits. Estimated random fallback is added on every warm-start failure; this fallback was not executed. The random total sums per-graph restart-to-reference cost estimates from saved random draws.",
              "", "| Set | Seed | Circuit charge from executed model attempts | Estimated fallback charge | Model + fallback total | Random reference total | Ratio of totals | Geometric mean ratio | Median ratio | Random zero-hit graphs |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, item in result["sets"].items():
        rows = list(item["costs"]["per_seed"].items()) + [("mean across available seeds", item["costs"]["seed_averaged"])]
        for seed, row in rows:
            lines.append("| %s | %s | %.2f | %.2f | %.2f | %.2f | %.6f | %.6f | %.6f | %d |" %
                         (tag, seed, row["executed_model_circuit_evaluations"], row["estimated_random_fallback_circuit_evaluations"],
                          row["total_model_plus_fallback"], row["total_random_reference"], row["ratio_of_totals"],
                          row["geometric_mean_cost_ratio"], row["median_cost_ratio"], row["random_zero_hit_graphs"]))
    lines += ["", "The following diagnostic table reports all three postprocessing strategies on the same held-out graphs. Snapping remains the strategy selected by the prespecified validation geometric-mean criterion. Raw postprocessing can have a lower ratio of total costs while snapping has a lower geometric mean; the objective and floating-point-sensitive optimizer paths both matter.",
              "", "| Set | Strategy | Hit rate | Ratio of total costs | Geometric mean ratio | Median ratio |",
              "|---|---|---:|---:|---:|---:|"]
    for tag, item in result["sets"].items():
        for method, data in item.get("all_postprocessing_strategies", {}).items():
            costs = data["costs"]["seed_averaged"]
            lines.append("| %s | %s | %s | %.6f | %.6f | %.6f |" %
                         (tag, method, percent(data["hit_rate"]), costs["ratio_of_totals"],
                          costs["geometric_mean_cost_ratio"], costs["median_cost_ratio"]))
    lines += ["", "The summed-cost intervals below resample whole graphs after averaging the available model seeds. They condition on the saved random-start estimates and trained seeds, and do not include uncertainty from additional random draws or training runs.",
              "", "| Set | Ratio of total costs [95% graph-bootstrap interval] |", "|---|---|"]
    for tag, item in result["sets"].items():
        row = item["costs"]["seed_averaged"]
        lines.append("| %s | %.6f [%.6f, %.6f] |" % (tag, row["ratio_of_totals"], *row["ratio_of_totals_ci95"]))
    lines += ["", "Zero-hit random graphs retain the existing conservative lower-bound reference cost. Ratios involving those graphs are estimates against that convention, not exact expected costs. Circuit charges follow the existing 2d+1 accounting per energy/gradient evaluation plus ranking calls; they are simulation-based operation counts, not measured quantum-hardware or wall-clock speedups.",
              "", "The next table distinguishes actual simulator calls from their circuit-equivalent charge for the selected snapped strategy. Model work sums all completed final seeds; the shared random baseline is counted once. Unselected raw/better strategies, validation, training and verification are additional work covered by the billing ledger. These measurement totals do not define a matched cost-to-reference ratio; no estimated fallback is included.",
              "", "| Set | Actual model joint optimizer calls | Actual model ranking calls | Model circuit-equivalent charge | Random starts executed | Actual random joint calls | Random circuit-equivalent charge |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for tag, item in result["sets"].items():
        if "all_saved_random_draws" in item:
            model_total = sum(row["executed_model_circuit_evaluations"] for row in item["costs"]["per_seed"].values())
            joint_calls = sum(row["executed_optimizer_joint_calls"] for row in item["costs"]["per_seed"].values())
            ranking_calls = sum(row["executed_ranking_energy_calls"] for row in item["costs"]["per_seed"].values())
            actual = item["all_saved_random_draws"]
            lines.append("| %s | %.0f | %.0f | %.0f | %d | %d | %.0f |" %
                         (tag, joint_calls, ranking_calls, model_total, actual["starts"], actual["optimizer_joint_calls"], actual["actual_circuit_evaluations"]))
    lines += [
              "", "### Complete bake-off on the same 2,000 validation graphs", "",
              "All rows use seed 42 and 40 training epochs on the same training ids. Each postprocessing strategy was selected on this shared validation subset. The A row is the paper architecture retrained on the new labels; the actual recorded old-paper predictions appear separately on Ian's rows.",
              "", "| Model | Recipe | Strategy | Hit rate | Ratio of totals | Geometric mean ratio | Median ratio |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in result["bakeoff"]:
        costs = row["costs"]
        lines.append("| %s | %s | %s | %s | %.6f | %.6f | %.6f |" %
                     (row["name"], row["description"], row["strategy"], percent(row["metrics"]["hit_rate"]["value"]),
                      costs["ratio_of_totals"], costs["geometric_mean_cost_ratio"], costs["median_cost_ratio"]))
    ablation = result["matched_label_ablation"]["supervised_regression_over_energy_only"]
    lines += ["", "The controlled label comparison is B versus E: matching regression architecture, training ids, seed and epoch budget, with label supervision versus energy-only loss. B/E geometric-mean cost ratio = %.6f, graph-bootstrap 95%% interval [%.6f, %.6f]. This is one matched training seed; it does not estimate training-seed variability. D versus E additionally changes classification and the number of hypotheses, so it cannot isolate the effect of unique labels." %
              (ablation["value"], ablation["low95"], ablation["high95"]),
              "B's periodic regression label loss also includes the prescribed 0.1 direction-norm penalty, which E's energy-only objective omits. This compares the complete supervised and energy-only recipes; it does not separate that regularizer's effect or compare canonical labels against other equally good minimum labels. E subtracts the reference energy only as an additive, parameter-independent logging offset; its training gradients do not depend on that reference value.",
              "", "### Reference labels that are not on the true floor", "",
              result["known_reference_counterexamples"]["interpretation"], "",
              result["known_reference_counterexamples"]["attribution"],
              "Primary success targets remain the original fixed reference energies, so results measure cost to reach those targets. The complete review records whether each counterexample belongs to the training split.",
              "", "| Id | n | Split | Stored reference energy | Independently checked lower energy | Improvement | Found by |",
              "|---:|---:|---|---:|---:|---:|---|"]
    for row in result["known_reference_counterexamples"]["cases"]:
        lines.append("| %d | %d | %s | %.8f | %.8f | %.8f | %s |" %
                     (row["id"], row["n"], row["set_name"], row["floor"], row["energy_after"], row["improvement"], row.get("discovery_source") or row["source"]))
    lines += ["", "### Recorded old paper GNN versus new D on Ian's twenty rows", "",
              "All twenty rows are unweighted eight-node graphs. Coordinate accuracies use the existing pi/8-snapped comparison to the recorded reference label; high-energy predictions can have matching coordinate classes. Initial energy quality is measured before any optimizer refinement. The old paper supplies one candidate; new D ranks four, so both top-one and up-to-four hit rates are shown.",
              "", "| Method | Initial expected cut / reference | Initial expected cut / exact Max-Cut | Edge class accuracy | Node class accuracy | Top-one hit after polish | Up-to-four hit after polish | Ratio of total costs |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["ian_old_vs_new"]["rows"]:
        if "seed" in row:
            top, four = percent(row["top_one_hit_rate"]), percent(row["up_to_four_hit_rate"])
        elif "summary" in row:
            top, four = percent(row["summary"]["hit_rate"]["value"]), "one candidate only"
        else:
            top, four = "pending existing evaluation", "one candidate only"
        costs = row.get("costs") if "seed" not in row else result["sets"]["ian"]["costs"]["per_seed"][str(row["seed"])]
        cost = "pending existing evaluation" if costs is None else "%.6f" % costs["ratio_of_totals"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" %
                     (row["name"], percent(row["initial_energy_over_reference"]), percent(row["initial_expected_cut_over_maxcut"]),
                      percent(row["edge_accuracy"]), percent(row["node_accuracy"]), top, four, cost))
    lines += ["", "Ian is a transfer check outside the weighted training distribution. Coordinate agreement with one recorded reference vector does not guarantee energy quality. Old-paper refinement comes from the planned paper_ian evaluation.",
              "", "### Training-set size and aggregate circuit cost", "",
              "Each fraction uses nested training ids and the same frozen 6,331 held-out graphs. Three model seeds are averaged within each graph. The 100% row reuses the final fits. All model attempts and estimated fallback are included in the cost ratios. With the same epoch cap and batch size, larger subsets receive more optimizer updates per epoch; this compares the fixed training recipe across data sizes and does not isolate data diversity from optimization exposure.",
              "", "| Training fraction | Training graphs | Completed seeds | Top-one hit | Up-to-four hit | Ratio of total costs [95% interval] | Geometric mean ratio | Median ratio |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for fraction, row in result["learning_curve_comparison"]["fractions"].items():
        if "costs" in row:
            costs = row["costs"]["seed_averaged"]
            interval = "%.6f [%.6f, %.6f]" % (costs["ratio_of_totals"], *costs["ratio_of_totals_ci95"])
            lines.append("| %s%% | %d | %s | %s | %s | %s | %.6f | %.6f |" %
                         (fraction, row["train_graphs"], ", ".join(map(str, row["seeds_available"])),
                          percent(row["top_one_hit_rate"]), percent(row["up_to_four_hit_rate"]),
                          interval, costs["geometric_mean_cost_ratio"], costs["median_cost_ratio"]))
        else:
            lines.append("| %s%% | %d | pending | pending | pending | pending | pending | pending |" %
                         (fraction, row["train_graphs"]))
    lines += ["", "Paired changes use the same held-out graphs and reference costs; a value below one favors the larger subset. These exploratory graph-bootstrap intervals use 1,000 resamples, are conditional on the saved starts and model seeds, and have no multiple-comparison adjustment.",
              "", "| Training change | Later / earlier summed cost [95% interval] |", "|---|---|"]
    for change, row in result["learning_curve_comparison"].get("paired_total_cost_changes", {}).items():
        lines.append("| %s%% → %s%% | %.6f [%.6f, %.6f] |" %
                     (*change.split("_to_"), row["later_over_earlier_summed_cost"], *row["ci95"]))
    if result["learning_curve_comparison"]["status"] == "complete":
        lines += ["", "![Total estimated circuit cost versus training data](figures/learning_curve_total_cost.png)",
                  "", "Error bars are whole-graph bootstrap 95% intervals. Costs include executed model attempts and estimated, unexecuted fallback. The dashed line is the shared random reference. This plot uses summed work; the earlier learning-curve plot uses the geometric mean of per-graph ratios."]
    precision = result["classifier_inference_precision"]
    lines += ["", "These subsets reflect the completed population. The missing 36,328 graphs include whole unstarted shards and slower shard tails; this curve alone cannot establish their value or justify extrapolating to that different mixture.",
              "", "### Classifier precision and optimizer sensitivity", "",
              precision["inference"], "", precision["interpretation"],
              "The selected snapped strategy was chosen on validation before final testing. No model code or training labels were changed in response to this check.",
              "", "| Classifier | Validation graphs | Maximum angle change (radians) | Raw geometric mean ratio | Snapped geometric mean ratio | Graphs with different polish hit |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, row in precision["results"].items():
        lines.append("| %s | %d | %.12g | %.6f | %.6f | %d |" %
                     (name, row["graphs"], row["maximum_phase_roundoff_radians"], row["raw_cost_ratio"],
                      row["snap_cost_ratio"], row["graphs_with_different_polish_hit"]))
    lines += ["", "[Independent precision evidence](../preflight/classifier_inference_precision.json).",
              "", "### Returned optimizer energies and final-state verification", "",
              "The independent energy gate also detected optimizer energy fields that did not match the returned final angles. The original evaluation did not retain optimizer termination status, so the cause of those mismatches is not established. This is separate from the small-angle class-constant rounding effect above.",
              "Every polish in completed model evaluation aggregates and saved reference baselines was audited with an exact p=1 expectation formula, validated against the unchanged pinned statevector checker. Flagged states were checked again with that checker. For very large angles, an 80-digit reduction by the exact circuit periods avoids matrix-exponential and binary64 phase-reduction errors. Original aggregates and optimizer energy values were retained. Final-energy metadata and its approximation-ratio summaries were corrected; every success decision, optimizer call count, candidate angle, cost statistic and model-selection result remained unchanged. The 1e-10 verification threshold was retained.",
              "Corrected values are in completed evaluation aggregates and the final-model prediction CSVs. Raw worker-batch evidence and byte-exact original aggregate backups remain available for audit; their original optimizer energy fields are not the authoritative physical final energies."]
    audit = result.get("all_polish_energy_audit")
    if audit is not None:
        lines += ["", "The final audit covers %s model-polish visits and %s baseline polishes. It finds %d remaining energy mismatches and %d changed success decisions. There were %d original optimizer mismatch visits (%d distinct final points); the largest original energy discrepancy was %.12g. [Complete audit](../preflight/all_polish_energy_audit.json)." %
                  (format(audit["counts"]["model_polish_visits"], ","), format(audit["counts"]["baseline_polishes"], ","),
                   audit["flagged_polish_visits"], audit["changed_hit_decisions"],
                   audit.get("original_optimizer_mismatch_visits", audit["flagged_polish_visits"]),
                   audit.get("original_optimizer_mismatch_unique_points", 0),
                   audit.get("maximum_original_optimizer_error", audit["maximum_error"]))]
    lines += [
              "", "The archive includes one environment proof per evaluation job, all scientific evaluation records and the selected model checkpoints. Repeated batch environment snapshots, temporary optimizer checkpoints, transport logs and operational diagnostics are omitted from delivery; the original local and Volume evidence remains preserved.",
              "", "The 25%/50%/100% learning-curve plan and $280 stop are unchanged. No missing training graphs were generated. Exact values, per-seed counts, matched-random sensitivities and input SHA-256s are in [the comparison evidence](requested_report_additions.json).", "", END_MARKER]
    return "\n".join(lines) + "\n"


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".comparison-additions.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


# Plot aggregate work separately from the geometric-mean selection score.
def draw_total_cost_learning_curve(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = result["learning_curve_comparison"]["fractions"]
    require(result["learning_curve_comparison"]["status"] == "complete", "Cannot plot unfinished fits")
    x = np.asarray([25, 50, 100])
    rows = [fractions[str(percent)]["costs"]["seed_averaged"] for percent in x]
    y = np.asarray([row["ratio_of_totals"] for row in rows])
    intervals = np.asarray([row["ratio_of_totals_ci95"] for row in rows])
    require(np.isfinite(intervals).all() and (intervals[:, 0] <= y).all()
            and (intervals[:, 1] >= y).all(), "Invalid total-cost plot intervals")
    figure, axis = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    axis.axhline(1, color="#777777", linestyle="--", linewidth=1, label="Random reference")
    axis.errorbar(x, y, yerr=np.vstack([y - intervals[:, 0], intervals[:, 1] - y]),
                  fmt="o-", color="#236789", linewidth=2, markersize=6, capsize=5,
                  label="D, snapped; 3 model seeds")
    for percent, value in zip(x, y):
        axis.annotate("%.3f" % value, (percent, value), xytext=(0, 10),
                      textcoords="offset points", ha="center", fontsize=10)
    axis.set_xticks(x, ["%d%%\n%s graphs" % (percent, format(fractions[str(percent)]["train_graphs"], ","))
                       for percent in x])
    axis.set_xlim(18, 107)
    axis.set_ylim(0, max(1.13, float(intervals.max()) * 1.2))
    axis.set_xlabel("Fraction of the frozen training split")
    axis.set_ylabel("Summed estimated cost / random reference")
    axis.set_title("Total estimated circuit cost versus training data")
    axis.grid(axis="y", alpha=.2)
    axis.legend(loc="lower left", frameon=False)
    destination = FOLDER / "figures"
    destination.mkdir(exist_ok=True)
    for extension in ["png", "pdf"]:
        figure.savefig(destination / ("learning_curve_total_cost." + extension), dpi=170)
    plt.close(figure)


def final_opening(result):
    regular = result["sets"]["test_split"]["costs"]["seed_averaged"]
    large = result["sets"]["large"]["costs"]["seed_averaged"]
    combined = result["sets"]["all_weighted_test_graphs"]["costs"]["seed_averaged"]
    raw_regular = result["sets"]["test_split"]["all_postprocessing_strategies"]["raw"]["costs"]["seed_averaged"]
    raw_large = result["sets"]["large"]["all_postprocessing_strategies"]["raw"]["costs"]["seed_averaged"]
    cases = result["known_reference_counterexamples"]["cases"]
    opening = (
        "The selected four-candidate classifier D uses snapped angles. On the 6,331 held-out n=5–12 graphs, "
        "its ratio of summed circuit-charge estimates to the random reference is **%.6f**; on the 584 n=13–14 graphs it is **%.6f**. "
        "The combined weighted-test ratio is **%.6f**. These totals include every executed model attempt and "
        "ranking call plus **estimated, unexecuted random fallback** after each model failure. They estimate cost "
        "to reach the fixed reference energies. They exclude classical inference and offline training and are not hardware runtime or dollar-cost measurements.\n\n"
        "The corresponding geometric mean of per-graph cost ratios is %.6f for n=5–12, with median %.6f. "
        "The model was selected using the prespecified geometric-mean criterion. That statistic gives graphs equal "
        "weight on a log scale; it does not measure the ratio of total work. The tables below report both, with "
        "three separately trained model seeds averaged within each graph and shared random baselines; this is not an ensemble. "
        "Raw postprocessing has lower summed-cost ratios (%.6f on n=5–12 and %.6f on n=13–14), "
        "while snapping wins the prespecified geometric-mean criterion. Both policies are reported.\n\n"
        "The dataset is frozen at 63,288 usable saved records. The 36,328 missing training-size graphs remain "
        "not run. The 25%%/50%%/100%% comparison uses nested subsets of the 50,628-graph training split, with "
        "the same held-out graphs at every fraction.\n\n"
        "**Reference-floor limitation:** %d original reference labels are demonstrably not on the true "
        "variational floor, because independently checked feasible points have lower energy. Primary success "
        "targets were kept fixed. These results therefore establish performance against empirical reference "
        "energies, not cost to a certified global minimum.\n\n" %
        (regular["ratio_of_totals"], large["ratio_of_totals"], combined["ratio_of_totals"],
         regular["geometric_mean_cost_ratio"], regular["median_cost_ratio"],
         raw_regular["ratio_of_totals"], raw_large["ratio_of_totals"], len(cases)))
    comparison = result["learning_curve_comparison"]
    if comparison["status"] == "complete":
        lines = ["Training-size results on the same 6,331 held-out graphs (three seeds per row):", "",
                 "| Training fraction | Training graphs | Total estimated cost / random | Geometric mean ratio | Warm-start hit, up to four |",
                 "|---|---:|---:|---:|---:|"]
        for percent in ["25", "50", "100"]:
            item = comparison["fractions"][percent]
            costs = item["costs"]["seed_averaged"]
            lines.append("| %s%% | %s | %.6f | %.6f | %.2f%% |" %
                         (percent, format(item["train_graphs"], ","), costs["ratio_of_totals"],
                          costs["geometric_mean_cost_ratio"], 100 * item["up_to_four_hit_rate"]))
        lines += ["", "Smaller cost ratios are better. Intervals, paired changes, seed variability and per-size results are below. The curve measures this training recipe within the completed population; it cannot establish the value of the different mixture among the 36,328 unrun graphs."]
        opening += "\n".join(lines) + "\n\n"
    return opening


def final_email(result):
    regular = result["sets"]["test_split"]["costs"]["seed_averaged"]
    large = result["sets"]["large"]["costs"]["seed_averaged"]
    ian = result["ian_old_vs_new"]["rows"]
    old = ian[0]
    new_hit = statistics.mean(row["up_to_four_hit_rate"] for row in ian[1:])
    new_one = statistics.mean(row["top_one_hit_rate"] for row in ian[1:])
    random = result["sets"]["ian"]["matched_hit_rates_by_n"][0]
    curve = result["learning_curve_comparison"]["fractions"]
    curve_text = "; ".join("%s%%: %s graphs, total-cost ratio %.3f (geometric mean %.3f)" %
                           (fraction, format(row["train_graphs"], ","), row["costs"]["seed_averaged"]["ratio_of_totals"],
                            row["costs"]["seed_averaged"]["geometric_mean_cost_ratio"])
                           for fraction, row in curve.items())
    count = len(result["known_reference_counterexamples"]["cases"])
    paragraphs = ["Hi Ian,",
        "I completed the weighted Max-Cut p=1 experiment on Presonance's tejas-16842 workspace. Of 63,672 saved records, 63,288 were usable (99.4%). The split has 50,628 training, 6,329 validation and 6,331 test graphs. The separate n=13–14 set has 584 usable graphs out of 600.",
        "D, the four-candidate classifier, won the seven-way validation comparison. Across three final seeds, its snapped policy used %.3f times the random reference's summed estimated circuit cost on n=5–12 and %.3f on n=13–14. These estimates include all model attempts and ranking calls, plus unexecuted random fallback on failures. Raw angles used less total estimated work; snapping won the prescribed geometric-mean selection score." %
        (regular["ratio_of_totals"], large["ratio_of_totals"]),
        "On your twenty unweighted graphs, the recorded old-paper predictions reached the fixed reference after one polish on %.1f%% of graphs. New D reached it on %.1f%% using its top-ranked candidate and %.1f%% with up to four. The matched random rates were %.1f%% for one start and %.1f%% for four." %
        (100 * old["summary"]["hit_rate"]["value"], 100 * new_one, 100 * new_hit,
         100 * random["random_first_one_hit_rate"], 100 * random["random_first_four_hit_rate"]),
        "The nested training-size comparison used three seeds per fraction and the same test graphs: " + curve_text + ". The 100% fits were reused. The 36,328 missing graphs remain unrun. Their size mix differs, so this curve cannot establish their value.",
        "There are %d independently confirmed counterexamples to the stored reference floors. I kept the original targets; these are empirical references, not certified global minima. The energy audit corrected inconsistent optimizer metadata without changing hits or costs. Very small changes from snapping can alter optimizer paths. The report separates these simulation results from hardware or runtime claims." % count,
        "Modal has posted $%.2f, with an upper reservation of $%.2f, below the $280 stop. The ledger includes failed runs. The report and verified archive contain the data, checks, weights, label ablation and comparisons. This email remains a draft." %
        (result["billing"]["reported_usd"], result["billing"]["upper_reservation_usd"]),
        "Thanks,\nAshay Sarda"]
    return "\n\n".join(paragraphs) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append-report", action="store_true", help="Replace only this supplement's marked section in the existing report")
    parser.add_argument("--require-complete", action="store_true", help="Reject a final report while any requested comparison is pending")
    args = parser.parse_args()
    result = build()
    if args.require_complete:
        require(result["status"] == "complete", "The requested final comparisons are not complete")
    if result["status"] == "complete":
        draw_total_cost_learning_curve(result)
    result["report_markdown"] = render(result)
    atomic_text(FOLDER / "requested_report_additions.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.append_report:
        path = FOLDER / "report.md"
        original = path.read_text(encoding="utf-8")
        if START_MARKER in original:
            prefix, rest = original.split(START_MARKER, 1)
            require(END_MARKER in rest, "Unterminated comparison supplement in existing report")
            _, suffix = rest.split(END_MARKER, 1)
            updated = prefix + result["report_markdown"] + suffix.lstrip("\r\n")
        else:
            updated = original.rstrip() + "\n\n" + result["report_markdown"]
        hit_note = "The overview hit-rate figure allows up to four model polishes; its random curve averages single-start success over the saved draws. Matched one-start and four-start comparisons are reported in the additional tables."
        if hit_note not in updated:
            target = "![Hit rate by size](figures/hit_rate_by_n.png)"
            updated = updated.replace(target, target + "\n\n" + hit_note, 1)
        histogram_note = "Initial-energy histograms show graph-level means: model seeds are averaged after ranking each model's candidates, while random starts are averaged within each graph. These are not distributions of individual random draws."
        if histogram_note not in updated:
            target = "![Initial energy distribution](figures/initial_energy_distribution.png)"
            updated = updated.replace(target, target + "\n\n" + histogram_note, 1)
        if result["all_final_seed_evaluations_present"] and result["paper_ian_after_refinement_present"]:
            header, separator, remainder = updated.partition("\n## Gates")
            require(separator and "Status:" in header, "Normal report opening is missing")
            title, _, status_and_intro = header.partition("\n\n")
            status = status_and_intro.split("\n", 1)[0]
            updated = title + "\n\n" + status + "\n\n" + final_opening(result) + separator + remainder
            if result["status"] == "complete":
                atomic_text(FOLDER / "email_draft.txt", final_email(result))
        atomic_text(path, updated)
    print(json.dumps({"status": result["status"], "sets": {
        tag: {"graphs": item["graphs"], "seeds_available": item["seeds_available"], "seeds_pending": item["seeds_pending"],
              "ratio_of_totals": item["costs"]["seed_averaged"]["ratio_of_totals"]}
        for tag, item in result["sets"].items()}, "incremental_paid_compute_usd": 0}))


if __name__ == "__main__":
    main()
