"""
Usage: python src/complete_full100k.py
Complete the original 100,000 graph experiment in an isolated results directory.
Preparation and all paid stages retain the existing $280 cumulative stop.
"""

import ast
import hashlib
import importlib
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = ROOT / "full_100k"
ORIGINAL_RESULTS = ROOT / "results"
GENERATION_RESERVE_USD = 75.0
TRAINING_PLAN_RESERVE_USD = 50.8
HISTORICAL_CURVE_USD = 13.75206555
HISTORICAL_FINAL_USD = 9.27183736
HISTORICAL_BAKEOFF_USD = 3.69005232
RUNTIME_HEADROOM = 1.15
VOLUME_NAME = "weighted-maxcut-full100k-20261030-cloudcopy"
SOURCE_VOLUME_NAME = "weighted-maxcut-unique-labels-20261028-threadfixed"
UPLOAD_BATCH_FILES = 64
PREPARE_ONLY = False
PREPARATION_WAIT_SECONDS = 900
LOG = logging.getLogger("qaoa.full100k")


# Preserve exact scientific source bytes while isolating all mutable experiment state.
def prepare_sources():
    original = (ROOT / "src" / "modal_app.py").read_text(encoding="utf-8")
    tree = ast.parse(original)
    replacements = {
        'APP_NAME = "weighted-maxcut-unique-labels"': 'APP_NAME = "weighted-maxcut-full100k"',
        'VOLUME_NAME = "weighted-maxcut-unique-labels-20261028-threadfixed"':
            'VOLUME_NAME = "' + VOLUME_NAME + '"',
        'TRAIN_EVAL_RESERVE_USD = 45.0': 'TRAIN_EVAL_RESERVE_USD = 50.8',
        'reserve = config["cost"]["full_reservation_usd" if mode == "full" else "test_reservation_usd"]':
            'reserve = 75.0 if mode == "full" else config["cost"]["test_reservation_usd"]',
        'elif not launched.exists():': 'elif not launched.exists() and not completed.exists():',
        'apps = {}\n    now = time.time()':
            'apps = dict(read_evidence(PREFLIGHT / "historical_billing.json")["apps"])\n    now = time.time()',
        'for name in [VOLUME_NAME, "weighted-maxcut-unique-labels-20261028"]:':
            'for name in [VOLUME_NAME, "weighted-maxcut-unique-labels-20261028", "weighted-maxcut-unique-labels-20261028-threadfixed", "weighted-maxcut-full100k-20261030", "weighted-maxcut-full100k-20261030-batched"]:',
        'require(total <= HARD_CAP_USD, "Projected total %.2f exceeds the $300 cap" % total)':
            'require(total <= SPEND_STOP_USD, "Projected total %.2f exceeds the $280 stop" % total)',
        'require(len(value["ids"]) == len(set(value["ids"])) == 63288 and len(value["not_run_ids"]) == 36328,':
            'require(len(value["ids"]) == len(set(value["ids"])) == value["usable"] >= 95000 and not value["not_run_ids"],',
        'coverage = dataset_gate(expected, records)\n    save_json(PREFLIGHT / (mode + "_corpus_coverage.json"), coverage)':
            'require(set(records) == set(expected), "Every planned graph must be saved before training")\n    coverage = dataset_gate(expected, records)\n    save_json(PREFLIGHT / (mode + "_corpus_coverage.json"), coverage)',
        'projection = 1.5 * blocks * ((maximum_block + 120) * hourly / 3600 + maximum_validation) + 5.0':
            'projection = 1.15 * read_evidence(PREFLIGHT / "continuation_cost_plan.json")["historical_curve_cost_usd"] * (len(load_training_population()["ids"]) / 63288) + 2.0',
        '"remaining_36328_graphs_generated": False': '"remaining_36328_graphs_generated": True',
    }
    patched = original
    for before, after in replacements.items():
        if before not in patched:
            raise RuntimeError("Continuation source anchor is missing: " + before[:90])
        patched = patched.replace(before, after)
    ast.parse(patched)
    source = EXPERIMENT / "src"
    source.mkdir(parents=True, exist_ok=True)
    data = EXPERIMENT / "data"
    data.mkdir(parents=True, exist_ok=True)
    runner = source / "modal_app.py"
    if runner.exists():
        if runner.read_text(encoding="utf-8") != patched:
            if (EXPERIMENT / "results" / "preflight" / "full_corpus_launch.json").exists():
                raise RuntimeError("Prepared continuation source changed after a paid launch")
            runner.write_text(patched, encoding="utf-8", newline="\r\n")
    else:
        runner.write_text(patched, encoding="utf-8", newline="\r\n")
    names = ["run_filter.py", "graph_sweep.py", "make_dataset.py", "check_rows.py", "gnn.py",
             "report_comparison_additions.py", "report_floor_review.py", "audit_evaluation_numerics.py",
             "audit_polish_energies.py"]
    for name in names:
        target = source / name
        expected = (ROOT / "src" / name).read_bytes()
        if target.exists() and target.read_bytes() != expected:
            raise RuntimeError("Continuation scientific source changed: " + name)
        if not target.exists():
            target.write_bytes(expected)
    for name in ["reference_picks.json", "MaxCutMAQAOAData.csv"]:
        target = data / name
        expected = (ROOT / "data" / name).read_bytes()
        if target.exists() and target.read_bytes() != expected:
            raise RuntimeError("Continuation input differs: " + name)
        if not target.exists():
            target.write_bytes(expected)
    return {"original_runner_sha256": hashlib.sha256((ROOT / "src" / "modal_app.py").read_bytes()).hexdigest(),
            "continuation_runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
            "continuation_runner_utf8_lf_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
            "original_function_count": sum(isinstance(n, ast.FunctionDef) for n in tree.body),
            "scientific_sources_byte_identical": True}


# Copy only hash-approved existing files between isolated cloud mounts.
def remote_copy_inputs(manifest, overrides):
    import hashlib
    import time
    from pathlib import Path
    import modal
    original = Path("/original").resolve()
    target = Path("/continuation").resolve()
    started = time.monotonic()
    total = 0
    for relative, expected in manifest.items():
        source_path = original / relative
        destination = target / relative
        if not destination.resolve().is_relative_to(target):
            raise ValueError("Input copy destination escapes its mount")
        if not source_path.resolve().is_relative_to(original):
            raise ValueError("Input copy source escapes its mount")
        data = overrides[relative] if relative in overrides else source_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Cloud input bytes differ: " + relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError("Cloud copy checksum differs: " + relative)
        total += len(data)
    modal.Volume.from_name(VOLUME_NAME).commit()
    return {"files": len(manifest), "bytes": total, "elapsed_seconds": time.monotonic() - started,
            "every_file_sha256_verified": True, "original_mount_writes": 0}


# Copy only immutable input evidence and saved records, without old training outputs.
def prepare_evidence(m, source_proof):
    import modal
    ready = m.PREFLIGHT / "continuation_inputs_ready.json"
    if ready.exists():
        prior = m.read_evidence(ready)
        if prior["source"] != source_proof:
            legacy = dict(source_proof)
            normalized = legacy.pop("continuation_runner_utf8_lf_sha256")
            legacy["continuation_runner_sha256"] = normalized
            m.require(prior["source"] == legacy and m.sha256(m.ROOT / "src/modal_app.py") == source_proof["continuation_runner_sha256"],
                      "Prepared source identity changed beyond the documented LF/byte hash correction")
            uploaded = [m.read_evidence(path)["upload_hashes"]["src/modal_app.py"]
                        for path in (m.PREFLIGHT / "training_launches").glob("*.json")]
            m.require(uploaded and set(uploaded) == {source_proof["continuation_runner_sha256"]},
                      "A recorded training upload used different runner bytes; inspect before correcting provenance")
            backup = m.PREFLIGHT / "continuation_inputs_ready_before_runner_hash_correction.json"
            if not backup.exists():
                shutil.copyfile(ready, backup)
            m.require(m.sha256(backup) == m.sha256(ready), "Preparation evidence backup differs")
            correction = {"legacy_record_sha256": m.sha256(backup),
                "legacy_field_interpretation": "continuation_runner_sha256 previously hashed derived UTF-8/LF text before Windows write_text converted line endings",
                "actual_runner_byte_sha256": source_proof["continuation_runner_sha256"],
                "normalized_utf8_lf_runner_sha256": normalized,
                "training_upload_manifests_checked": len(uploaded), "every_recorded_training_upload_matches_actual_bytes": True,
                "runner_and_pinned_input_bytes_changed": False, "new_compute": False, "corrected_epoch": time.time()}
            m.save_json(m.PREFLIGHT / "runner_hash_provenance_correction.json", correction)
            m.save_json(ready, dict(prior, source=source_proof,
                                   source_hash_correction="runner_hash_provenance_correction.json"))
        return
    m.require(not (m.PREFLIGHT / "full_corpus_launch.json").exists(),
              "Refusing to replace continuation inputs after generation has started")
    m.PREFLIGHT.mkdir(parents=True, exist_ok=True)
    names = ["frozen_plan.json", "frozen_plan_lock.json", "stopping_rule.json", "test_timings.json",
             "test_corpus_csv_gate.json", "test_corpus_coverage.json", "ian_floor_gate.json",
             "reference_build.json", "reference_results.json", "reference_stage_gate.json",
             "pilot_coverage.json", "pilot400_coverage.json", "stopping_rule.json", "modal_billing_rates.json"]
    uploads = []
    for name in sorted(set(names)):
        origin = ORIGINAL_RESULTS / "preflight" / name
        if origin.exists():
            target = m.PREFLIGHT / name
            shutil.copyfile(origin, target)
            uploads.append(target)
    for folder in ["dataset", "dataset_test", "ian_floors"]:
        for origin in sorted((ORIGINAL_RESULTS / folder).glob("*.json*")):
            target = m.RESULTS / folder / origin.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            m.require(m.sha256(target) == m.sha256(origin), "Saved record copy failed its hash")
            uploads.append(target)
    origin = ORIGINAL_RESULTS / "dataset_test" / "dataset.csv"
    shutil.copyfile(origin, m.RESULTS / "dataset_test" / "dataset.csv")
    uploads.append(m.RESULTS / "dataset_test" / "dataset.csv")
    for name in ["torch_energy_check.json", "torch_energy_check_cuda.json"]:
        target = m.RESULTS / "gnn" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ORIGINAL_RESULTS / "gnn" / name, target)
        uploads.append(target)
    historical = json.loads((ORIGINAL_RESULTS / "preflight" / "billing_reconciliation.json").read_text())
    m.save_json(m.PREFLIGHT / "historical_billing.json", historical)
    m.save_json(m.PREFLIGHT / "continuation_cost_plan.json", {
        "user_authorized_remaining_generation": True, "expected_new_graphs": 36328,
        "generation_reserve_usd": GENERATION_RESERVE_USD,
        "training_evaluation_projection_usd": TRAINING_PLAN_RESERVE_USD,
        "historical_curve_cost_usd": HISTORICAL_CURVE_USD,
        "historical_final_cost_usd": HISTORICAL_FINAL_USD,
        "historical_bakeoff_cost_usd": HISTORICAL_BAKEOFF_USD,
        "historical_upper_usd": historical["spend_reservation_usd"],
        "initial_total_projection_usd": historical["spend_reservation_usd"] + GENERATION_RESERVE_USD + TRAINING_PLAN_RESERVE_USD,
        "stop_usd": 280, "original_experiment_preserved": True, "source": source_proof})
    (m.RESULTS / "cost_ledger.md").write_text(
        "# Full 100,000 graph continuation\n\nThe user authorized the remaining 36,328 graphs and the full experiment. "
        "All historical task charges count toward the unchanged $280 stop. "
        "The original 63,288-usable experiment and its archive remain intact.\n", encoding="utf-8")
    uploads.extend([m.PREFLIGHT / "historical_billing.json", m.PREFLIGHT / "continuation_cost_plan.json"])
    volume = modal.Volume.from_name(m.VOLUME_NAME, create_if_missing=True, version=2)
    volume.hydrate()
    LOG.info("Copying %d hash-approved inputs between cloud Volumes", len(uploads))
    manifest = {path.relative_to(m.RESULTS).as_posix(): m.sha256(path) for path in uploads}
    overrides = {path.relative_to(m.RESULTS).as_posix(): path.read_bytes() for path in uploads
                 if path.name in ["historical_billing.json", "continuation_cost_plan.json"]}
    spending = m.reconciled_spend_reservation()
    m.budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"],
                  GENERATION_RESERVE_USD + TRAINING_PLAN_RESERVE_USD + .01)
    image = m.data_image().add_local_file(ROOT / "src" / "complete_full100k.py",
                                         m.REMOTE_ROOT + "/src/complete_full100k.py", copy=True)
    app = modal.App("weighted-maxcut-full100k-input-copy")
    function = app.function(image=image, cpu=(1, 1), memory=(1024, 2048), timeout=300,
                            retries=0, max_containers=1, scaledown_window=2, serialized=True,
                            volumes={"/original": modal.Volume.from_name(SOURCE_VOLUME_NAME),
                                     "/continuation": volume}, name="copy_verified_inputs")(remote_copy_inputs)
    hourly = .0473 + 2 * .008
    try:
        with app.run(detach=True):
            m.register_paid_app(app.app_id, "verified cloud input copy", hourly, budget_cap_usd=.01)
            call = function.spawn(manifest, overrides)
            m.save_json(m.PREFLIGHT / "cloud_input_copy_launch.json", {"app_id": app.app_id,
                        "call_id": call.object_id, "files": len(manifest), "reserve_usd": .01})
            proof = call.get()
    finally:
        if app.app_id:
            app.stop()
            m.register_paid_app(app.app_id, "verified cloud input copy", hourly, stopped=True, budget_cap_usd=.01)
    m.require(proof["files"] == len(manifest) and proof["every_file_sha256_verified"], "Cloud copy is incomplete")
    m.save_json(m.PREFLIGHT / "cloud_input_copy_result.json", proof)
    m.save_json(ready, {"source": source_proof, "files": len(uploads), "captured_epoch": time.time()})


# Keep the full expected-id denominator and every original record unchanged.
def freeze_complete_population(m):
    path = m.PREFLIGHT / "training_population.json"
    if path.exists():
        return m.load_training_population()
    config = m.load_frozen_plan()
    dataset = m.configured_dataset("full", config)
    expected = m.expected_manifest(dataset.selected("full"))
    records = m.load_records(m.RESULTS / "dataset", expected)
    m.require(len(expected) == 100000 and set(records) == set(expected), "The 100,000 graph corpus is incomplete")
    quality = m.dataset_gate(expected, records)
    previous = json.loads((ORIGINAL_RESULTS / "preflight" / "training_population.json").read_text())
    for graph_id in previous["ids"]:
        actual = hashlib.sha256(json.dumps(records[graph_id], sort_keys=True).encode()).hexdigest()
        m.require(actual == previous["row_hashes"][str(graph_id)], "An original usable label changed")
    saved_original = {}
    for origin in (ORIGINAL_RESULTS / "dataset").glob("shard_*.jsonl"):
        for line in origin.read_text().splitlines():
            row = json.loads(line)
            saved_original[row["id"]] = row
    m.require(len(saved_original) == 63672, "Original saved population differs")
    m.require(all(records[i] == row for i, row in saved_original.items()), "An original saved row changed")
    ids = sorted(i for i, row in records.items() if m.usable(row, expected[i]))
    value = {"ids": ids, "not_run_ids": [], "rejected_saved_ids": sorted(set(records) - set(ids)),
             "saved_ids": sorted(records), "expected": {i: expected[i] for i in ids}, "saved": len(records),
             "usable": len(ids), "not_run": 0, "quality": quality,
             "frozen_plan_sha256": m.sha256(m.PREFLIGHT / "frozen_plan.json"), "input_hashes": m.EXPECTED_HASHES,
             "row_hashes": {i: hashlib.sha256(json.dumps(records[i], sort_keys=True).encode()).hexdigest() for i in ids},
             "original_63672_saved_rows_unchanged": True, "newly_generated": 36328,
             "learning_curve": {"fractions": [.25, .5, 1.0], "status": "approved"}, "created_epoch": time.time()}
    m.save_json(path, value)
    m.save_json(m.PREFLIGHT / "training_population_lock.json", {"sha256": m.sha256(path)})
    m.save_json(m.PREFLIGHT / "learning_curve_approval.json", {"user_approved": True,
                "scope": "Full experiment including nested 25/50/100 percent fits; cumulative $280 stop"})
    return value


# Reconcile cumulative billing and cost all remaining work before each large phase.
def completed_curve_cost_credit(m, bill):
    applications = set()
    runner_hash = m.sha256(m.ROOT / "src/modal_app.py")
    for path in (m.PREFLIGHT / "training_launches").glob("curve*.json"):
        launch = m.read_evidence(path)
        paid = m.read_evidence(m.PREFLIGHT / "paid_apps" / (launch["app_id"] + ".json"))
        if (launch.get("upload_hashes", {}).get("src/modal_app.py") == runner_hash
                and paid and paid.get("name") == "training session learning_curve" and paid.get("stopped_epoch")):
            applications.add(launch["app_id"])
    for path in (m.PREFLIGHT / "evaluation_launches").glob("curve*.json"):
        launch = m.read_evidence(path)
        completed_result = (m.RESULTS / "gnn/evaluations" / path.name).is_file()
        paid = m.read_evidence(m.PREFLIGHT / "paid_apps" / (launch["app_id"] + ".json"))
        if (completed_result and launch.get("upload_hashes", {}).get("src/modal_app.py") == runner_hash
                and paid and paid.get("stopped_epoch")):
            applications.add(launch["app_id"])
    entries = [row for row in bill["entries"] if row["app_id"] in applications]
    return math.fsum(row["cost_usd"] for row in entries), sorted(applications)


# Adjust only the local coordinator's upfront cost expression; frozen worker bytes stay pinned.
def install_remaining_curve_projection(m, credit):
    import inspect
    source = inspect.getsource(m.train_learning_curve)
    tree = ast.parse(source)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "projection" for target in node.targets)]
    m.require(len(assignments) == 1, "The curve cost expression changed; refuse an ambiguous resume adjustment")
    assignment = assignments[0]
    expected = ast.parse('projection = 1.15 * read_evidence(PREFLIGHT / "continuation_cost_plan.json")["historical_curve_cost_usd"] * (len(load_training_population()["ids"]) / 63288) + 2.0').body[0]
    m.require(ast.dump(assignment.value) == ast.dump(expected.value), "Unexpected curve projection formula")
    assignment.value = ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
        args=[ast.Constant(0.0), ast.BinOp(left=assignment.value, op=ast.Sub(),
                                       right=ast.Name(id="CURVE_COMPLETED_COST_CREDIT_USD", ctx=ast.Load()))], keywords=[])
    ast.fix_missing_locations(tree)
    derived = ast.unparse(tree) + "\n"
    path = m.PREFLIGHT / "curve_resume_coordinator_function.py"
    path.write_text(derived, encoding="utf-8")
    m.CURVE_COMPLETED_COST_CREDIT_USD = credit
    namespace = {}
    exec(compile(tree, str(path), "exec"), m.__dict__, namespace)
    m.train_learning_curve = namespace["train_learning_curve"]
    return {"original_function_utf8_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "derived_coordinator_function_sha256": m.sha256(path), "frozen_worker_source_sha256": m.sha256(m.ROOT / "src/modal_app.py"),
            "only_semantic_change": "upfront remaining projection = max(0, whole-curve projection - posted completed curve charges)",
            "actual_spend_stop_changed": False, "per_allocation_reserve_changed": False}


def phase_budget(m, phase):
    bill = m.write_billing_summary()
    population = m.load_training_population()
    scale = population["usable"] / 63288
    costs = {"prepare": HISTORICAL_BAKEOFF_USD + HISTORICAL_FINAL_USD + HISTORICAL_CURVE_USD,
             "bakeoff": HISTORICAL_BAKEOFF_USD + HISTORICAL_FINAL_USD + HISTORICAL_CURVE_USD,
             "final": HISTORICAL_FINAL_USD + HISTORICAL_CURVE_USD,
             "curve": HISTORICAL_CURVE_USD}
    bakeoff = m.read_evidence(m.RESULTS / "gnn/bakeoff.json")
    reused = bool(bakeoff and bakeoff.get("selection_provenance", {}).get("reused_prior_selection"))
    if reused and phase in ["prepare", "bakeoff"]:
        costs[phase] -= HISTORICAL_BAKEOFF_USD
    remaining = RUNTIME_HEADROOM * costs[phase] * scale + 2.0
    credit, credited_applications, adjustment = 0., [], None
    if phase == "curve":
        credit, credited_applications = completed_curve_cost_credit(m, bill)
        m.require(0 <= credit < remaining, "Posted curve charges exceed the whole-curve projection; re-cost remaining work")
        if credit:
            adjustment = install_remaining_curve_projection(m, credit)
            whole = remaining
            remaining -= credit
            m.save_json(m.PREFLIGHT / "curve_resume_budget.json", {**adjustment,
                "whole_curve_projection_usd": whole, "posted_completed_cost_credit_usd": credit,
                "credited_application_ids": credited_applications, "remaining_curve_projection_usd": remaining,
                "prior_upper_usd": bill["upper_reservation_usd"], "checked_epoch": time.time(),
                "ledger_reported_usd_unchanged": bill["reported_usd"], "cumulative_stop_usd": m.SPEND_STOP_USD})
    m.require(bill["upper_reservation_usd"] + remaining < m.SPEND_STOP_USD,
              "The remaining experiment projection would reach the $280 stop")
    m.save_json(m.PREFLIGHT / ("phase_budget_" + phase + ".json"), {
        "historical_phase_costs_usd": costs[phase], "population_scale": scale,
        "runtime_headroom": RUNTIME_HEADROOM, "additional_overhead_usd": 2.0,
        "remaining_projection_usd": remaining, "prior_upper_usd": bill["upper_reservation_usd"],
        "completed_posted_curve_cost_credit_usd": credit, "curve_resume_adjustment": adjustment,
        "total_projection_usd": bill["upper_reservation_usd"] + remaining,
        "architecture_selection_reused": reused,
        "basis": "Observed full prior experiment costs scaled by usable population, plus 15% and $2; per-launch timeout bounds remain enforced"})
    m.save_json(m.PREFLIGHT / "training_pool_budget.json", {
        "model_source_sha256": m.sha256(m.ROOT / "src" / "gnn.py"),
        "maximum_worker_hourly_usd": m.TRAINING_MAX_GPU_HOURLY_USD + 4 * .0473 + 16 * .008,
        "GPU_overhead_seconds_per_input": m.TRAINING_IDLE_SECONDS + 30,
        "total_projection_usd": bill["upper_reservation_usd"] + remaining,
        "remaining_projection_usd": remaining, "one_GPU_at_a_time": True,
        "remaining_36328_generation_permitted": True})
    m.TRAIN_EVAL_RESERVE_USD = 8.0
    return bill


# Freeze the previously measured architecture choice before any new held-out comparison.
def reuse_previous_selection(m):
    decision = m.read_evidence(m.PREFLIGHT / "reuse_prior_architecture.json")
    if not decision or not decision.get("enabled"):
        return
    path = m.RESULTS / "gnn/bakeoff.json"
    if path.exists():
        return
    source = ORIGINAL_RESULTS / "gnn/bakeoff.json"
    value = json.loads(source.read_text())
    m.require(value["selection"]["chosen"] == "D", "Previous selected architecture differs from the budget plan")
    value["selection_provenance"] = {"reused_prior_selection": True, "original_usable_population": 63288,
        "source": str(source), "source_sha256": m.sha256(source), "decision": decision,
        "interpretation": "Architecture selected in the completed original seven-model bake-off, frozen for expanded-data retraining. No new expanded-data bake-off was run."}
    value["original_checkpoint_paths"] = value["checkpoint_paths"]
    value["checkpoint_paths"] = {}
    m.save_json(path, value)


# Repeat architecture selection only when explicitly requested, preserving prior fits.
def repeat_expanded_bakeoff(m):
    request = m.read_evidence(m.PREFLIGHT / "expanded_bakeoff_user_request.json")
    m.require(request and request.get("approved") is True,
              "The expanded bake-off requires the recorded user request")
    gate = m.PREFLIGHT / "stage3_gate.json"
    held_gate = m.PREFLIGHT / "stage3_gate_before_expanded_bakeoff.json"
    expected = request["stage3_scientific_gate_still_valid_sha256"]
    m.require(held_gate.exists() and m.sha256(held_gate) == expected,
              "The preserved scientific gate differs")
    if not gate.exists():
        status = m.read_evidence(m.RESULTS / "status.json")
        m.require(status and status.get("error") == "Stage 3 has not passed",
                  "Reconcile the prior coordinator before releasing the allocation hold")
        shutil.copyfile(held_gate, gate)
    m.require(m.sha256(gate) == expected, "The restored scientific gate differs")
    population = m.load_training_population()
    m.require(population["usable"] == 99368 and population["saved"] == 100000
              and population["not_run"] == 0, "Expanded population is incomplete")
    path = m.RESULTS / "gnn/bakeoff.json"
    backup = m.RESULTS / "gnn/bakeoff_prior_D.json"
    old = m.read_evidence(path)
    if old and old.get("selection_provenance", {}).get("reused_prior_selection"):
        if backup.exists():
            m.require(m.sha256(path) == m.sha256(backup), "Prior selection backup differs")
        else:
            shutil.copyfile(path, backup)
        path.unlink()
    decision_path = m.PREFLIGHT / "reuse_prior_architecture.json"
    prior_decision = m.PREFLIGHT / "reuse_prior_architecture_before_expanded_bakeoff.json"
    if not prior_decision.exists():
        shutil.copyfile(decision_path, prior_decision)
    m.save_json(decision_path, {"enabled": False, "reason": request["request"],
                              "user_approved": True, "updated_epoch": time.time()})
    bill = m.write_billing_summary()
    remaining = RUNTIME_HEADROOM * HISTORICAL_BAKEOFF_USD * population["usable"] / 63288 + .5
    minimum_comparison = 5.8
    m.require(bill["upper_reservation_usd"] + remaining + minimum_comparison < m.SPEND_STOP_USD,
              "Expanded bake-off plus the minimum matched comparison would reach $280")
    plan = {"historical_bakeoff_cost_usd": HISTORICAL_BAKEOFF_USD,
            "population_scale": population["usable"] / 63288,
            "runtime_headroom": RUNTIME_HEADROOM, "additional_overhead_usd": .5,
            "remaining_projection_usd": remaining,
            "minimum_matched_comparison_reserve_usd": minimum_comparison,
            "prior_upper_usd": bill["upper_reservation_usd"],
            "total_projection_usd": bill["upper_reservation_usd"] + remaining + minimum_comparison,
            "scope": "Seven models on the expanded training split; fixed 2000 validation graphs; no held-out selection",
            "completed_D_fits_preserved": True,
            "later_work": "Reconcile the selected architecture, completed fits and remaining budget before resuming curves and comparison"}
    m.save_json(m.PREFLIGHT / "phase_budget_expanded_bakeoff.json", plan)
    m.save_json(m.PREFLIGHT / "training_pool_budget.json", {
        "model_source_sha256": m.sha256(m.ROOT / "src/gnn.py"),
        "maximum_worker_hourly_usd": m.TRAINING_MAX_GPU_HOURLY_USD + 4 * .0473 + 16 * .008,
        "GPU_overhead_seconds_per_input": m.TRAINING_IDLE_SECONDS + 30,
        "total_projection_usd": plan["total_projection_usd"],
        "remaining_projection_usd": remaining, "one_GPU_at_a_time": True,
        "remaining_36328_generation_permitted": False})
    m.TRAIN_EVAL_RESERVE_USD = 8.0
    write_status(m, "expanded seven-model bakeoff")
    if not path.exists():
        with m.training_session("expanded_repeat"):
            result = m.run_bakeoff()
        result["selection_provenance"] = {
            "reused_prior_selection": False, "expanded_usable_population": population["usable"],
            "user_request": request["request"], "request_sha256": m.sha256(m.PREFLIGHT / "expanded_bakeoff_user_request.json"),
            "prior_selection_sha256": m.sha256(backup),
            "chronology": "Three full D fits and part of the D learning curve completed before the delayed request to repeat selection arrived. Preserved as prior work; any reuse requires matching the new winning configuration and strategy.",
            "selection_uses_heldout_test": False}
        m.save_json(path, result)
    m.write_billing_summary()
    write_status(m, "expanded seven-model bakeoff complete; winner reconciliation pending")
    return 0


# Reuse prior paid fits only when the repeated selection leaves their recipe intact.
def reconcile_expanded_selection(m):
    bakeoff = m.read_evidence(m.RESULTS / "gnn/bakeoff.json")
    provenance = (bakeoff or {}).get("selection_provenance", {})
    if not provenance or provenance.get("reused_prior_selection") is not False:
        return
    prior = m.read_evidence(m.RESULTS / "gnn/bakeoff_prior_D.json")
    fits = m.read_evidence(m.RESULTS / "gnn/final_training.json")
    final = m.read_evidence(m.RESULTS / "gnn/final_results.json")
    if not fits:
        return
    m.require(prior and bakeoff["selection"]["chosen"] == prior["selection"]["chosen"]
              and bakeoff["configuration"] == prior["configuration"]
              and bakeoff["strategy"] == prior["strategy"],
              "Expanded selection changed the winner or recipe; cost replacement fits before reusing any final artifacts")
    configuration = {key: value for key, value in bakeoff["configuration"].items()
                     if key not in ["seed", "maximum_epochs"]}
    training_hash = hashlib.sha256(json.dumps(m.read_evidence(m.RESULTS / "gnn/splits.json")["ids"]["train"]).encode()).hexdigest()
    m.require(set(fits["seeds"]) == {"42", "43", "44"}, "Preserved full fits are incomplete")
    for fit in fits["seeds"].values():
        trained = fit["best"]["training"]
        actual = {key: value for key, value in trained["config"].items()
                  if key not in ["seed", "maximum_epochs"]}
        m.require(actual == configuration and trained["training_ids_sha256"] == training_hash
                  and trained["source_hashes"]["src/gnn.py"] == m.sha256(m.ROOT / "src/gnn.py"),
                  "A preserved full fit differs in model, training IDs or scientific source")
    m.require(final and final["strategy"] == bakeoff["strategy"],
              "Existing held-out evaluation uses a different selected policy")
    m.save_json(m.PREFLIGHT / "expanded_winner_reuse_check.json", {
        "passed": True, "winner": bakeoff["selection"]["chosen"], "strategy": bakeoff["strategy"],
        "bakeoff_sha256": m.sha256(m.RESULTS / "gnn/bakeoff.json"),
        "preserved_prior_selection_sha256": m.sha256(m.RESULTS / "gnn/bakeoff_prior_D.json"),
        "full_training_sha256": m.sha256(m.RESULTS / "gnn/final_training.json"),
        "training_ids_sha256": training_hash, "full_fit_seeds": [42, 43, 44],
        "chronology": provenance["chronology"], "checked_epoch": time.time(),
        "new_heldout_results_used_for_reuse_decision": False})


# Verify all saved physical energies and report feasible improvements to references.
def final_numerical_review(m):
    import audit_polish_energies
    import numpy as np
    review = audit_polish_energies.audit()
    m.save_json(m.PREFLIGHT / "all_polish_energy_audit.json", review)
    m.require(review["flagged_polish_visits"] == 0 and review["changed_hit_decisions"] == 0,
              "The final independent physical-energy audit failed")
    corpus = audit_polish_energies.Corpus()
    findings, aggregates = {}, 0
    for path in sorted((m.RESULTS / "gnn" / "evaluations").glob("*.json")):
        value = m.read_evidence(path)
        aggregates += 1
        for method, records in value["results"].items():
            if method not in ["raw", "snap", "better"]:
                continue
            for record in records:
                row = corpus.row(value["corpus"], record["id"])
                for point in record["polishes"]:
                    if point["energy_after"] >= row["floor"] - m.ENERGY_TOL:
                        continue
                    key = (value["corpus"], row["id"])
                    if key in findings and findings[key]["energy"] <= point["energy_after"]:
                        continue
                    physical = audit_polish_energies.energy_batch(row, [point["angles_after"]])[0]
                    m.require(abs(physical - point["energy_after"]) <= 1e-10, "A feasible floor improvement is inconsistent")
                    findings[key] = {"corpus": key[0], "id": key[1], "n": row["n"], "reference": row["floor"],
                                     "energy": float(physical), "angles": point["angles_after"],
                                     "source": path.stem + "/" + method}
    for tag in ["val", "test_split", "large", "ian"]:
        for path in sorted((m.RESULTS / "gnn" / "baselines" / tag).glob("*.json")):
            baseline = m.read_evidence(path)["baseline"]
            key = ("full" if tag in ["val", "test_split"] else tag, baseline["id"])
            row = corpus.row(*key)
            for method in ["random", "zero", "oracle"]:
                points = [baseline[method]] if method == "oracle" else baseline[method]
                for point in points:
                    if point["energy_after"] >= row["floor"] - m.ENERGY_TOL:
                        continue
                    if key in findings and findings[key]["energy"] <= point["energy_after"]:
                        continue
                    physical = audit_polish_energies.energy_batch(row, [point["angles_after"]])[0]
                    m.require(abs(physical - point["energy_after"]) <= 1e-10, "A baseline floor improvement is inconsistent")
                    findings[key] = {"corpus": key[0], "id": key[1], "n": row["n"], "reference": row["floor"],
                                     "energy": float(physical), "angles": point["angles_after"],
                                     "source": "baseline/" + tag + "/" + method}
    result = {"model_aggregates_reviewed": aggregates, "known_counterexample_graphs": len(findings),
              "findings": [findings[key] for key in sorted(findings)], "primary_targets_changed": False,
              "interpretation": "Independently checked feasible points can refute an empirical floor. Neither reproducibility nor this review proves global optimality or uniqueness.",
              "new_modal_compute": False, "created_epoch": time.time()}
    m.save_json(m.PREFLIGHT / "reference_floor_review.json", result)
    return result


# Package selected model weights and evidence without logs, credentials or temporary data.
def archive_completed_experiment(m):
    import zipfile
    selected = m.delivery_checkpoints()
    expected_hashes = {}
    for path in (m.RESULTS / "gnn" / "training_jobs").glob("*.json"):
        value = m.read_evidence(path)
        relative = value["checkpoint"].replace("\\", "/")
        if relative in selected:
            expected_hashes[relative] = value["checkpoint_sha256"]
    final = m.read_evidence(m.RESULTS / "gnn" / "final_training.json")
    for value in final["seeds"].values():
        expected_hashes[value["final_checkpoint"]] = value["best"]["training"]["checkpoint_sha256"]
    curve = m.read_evidence(m.RESULTS / "gnn" / "learning_curve.json")
    for percent in ["25", "50"]:
        for value in curve["fractions"][percent]["seeds"].values():
            expected_hashes[value["checkpoint"]] = value["best"]["training"]["checkpoint_sha256"]
    m.require(set(expected_hashes) == selected, "A selected checkpoint lacks its training checksum")
    for relative in sorted(selected):
        target = m.download_artifact(relative)
        m.require(target.is_file() and m.sha256(target) == expected_hashes[relative],
                  "A selected checkpoint differs from its recorded training checksum")
    output = m.RESULTS / "delivery" / "weighted_maxcut_full100k.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    files["orchestration/complete_full100k.py"] = ROOT / "src" / "complete_full100k.py"
    for name in [".gitattributes", ".gitignore", "README.md"]:
        files[name] = ROOT / name
    for name in ["modal_app.py", "gnn.py", "run_filter.py", "make_dataset.py", "check_rows.py",
                 "graph_sweep.py", "report_comparison_additions.py", "report_floor_review.py",
                 "audit_polish_energies.py", "audit_evaluation_numerics.py", "complete_full100k.py",
                 "paper_model.py", "paper_comparison.py", "research_verification.py"]:
        files["original_source/src/" + name] = ROOT / "src" / name
    for name in ["reference_picks.json", "MaxCutMAQAOAData.csv"]:
        files["original_source/data/" + name] = ROOT / "data" / name
    readme = output.parent / "archive_readme.txt"
    readme.write_text(
        "Weighted MA-QAOA full-data experiment\n\n"
        "src/ and data/ contain the exact isolated Modal inputs. original_source/ contains the repository sources "
        "and small inputs from which the isolated runner is derived; orchestration/ contains the coordinator too. "
        "Root README.md and .gitattributes describe usage and exact-byte Git preservation.\n\n"
        "This is the completed core dataset/GNN/learning-curve experiment. The separate paper_comparison.zip "
        "contains the paper-based GNN and diffusion comparison; its live or pending state is excluded here.\n\n"
        "All scientific stages and numerical gates finish before this snapshot is packaged. Live status/report "
        "snapshots may say archive verification because the archive cannot contain its own final hash. "
        "The adjacent weighted_maxcut_full100k.json is written only after CRC, every manifest SHA-256 and "
        "credential checks pass. The live results/status.json becomes complete after that verification. "
        "manifest.json records every packaged file's exact bytes and SHA-256.\n",
        encoding="utf-8")
    files["ARCHIVE_README.txt"] = readme
    for folder in [m.ROOT / "src", m.ROOT / "data"]:
        for path in sorted(folder.glob("*")):
            if path.is_file() and path.suffix in [".py", ".json", ".csv"]:
                files[path.relative_to(m.ROOT).as_posix()] = path
    for path in sorted(m.RESULTS.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(m.RESULTS)
        name = relative.as_posix()
        if relative.parts[0] == "paper_comparison":
            continue
        if any(part in ["tmp", "logs", "delivery", "worker_proofs", "__pycache__"] for part in relative.parts):
            continue
        if path.suffix not in [".json", ".jsonl", ".csv", ".npz", ".png", ".pdf", ".md", ".pt", ".py"]:
            continue
        if relative.parts[0] == "gnn" and "checkpoints" in relative.parts and name not in selected:
            continue
        files["results/" + name] = path
    manifest = {name: {"sha256": m.sha256(path), "bytes": path.stat().st_size} for name, path in files.items()}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        m.require(archive.testzip() is None, "Archive CRC verification failed")
        for name, entry in manifest.items():
            with archive.open(name) as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            m.require(actual == entry["sha256"], "Archive entry checksum differs: " + name)
    credentials = m.verify_delivery_archive_secrets(output)
    proof = {"archive": str(output), "sha256": m.sha256(output), "files": len(files),
             "bytes": output.stat().st_size, "manifest_verified": True,
             "credential_scan": credentials, "completed_epoch": time.time()}
    m.save_json(output.with_suffix(".json"), proof)
    return proof


# Draw the held-out and learning-curve figures from completed numerical evidence.
def draw_completed_figures(m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    final = m.read_evidence(m.RESULTS / "gnn" / "final_results.json")
    m.draw_result_figures(final)
    curve = m.read_evidence(m.RESULTS / "gnn" / "learning_curve.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    x = [curve["fractions"][str(p)]["graphs"] for p in [25, 50, 100]]
    for axis, key, title in zip(axes, ["geometric_mean_cost_ratio", "hit_rate"],
                                ["Geometric mean work / random", "Hit rate within up to four proposals"]):
        metrics = [curve["fractions"][str(p)]["summary"]["metrics"][key] for p in [25, 50, 100]]
        y = [row["value"] for row in metrics]
        low = [value - row["low95"] for row, value in zip(metrics, y)]
        high = [row["high95"] - value for row, value in zip(metrics, y)]
        axis.errorbar(x, y, yerr=[low, high], marker="o", capsize=4)
        axis.set(xlabel="Training graphs", ylabel=title)
        axis.grid(alpha=.2)
    fig.suptitle("Nested training subsets; fixed held-out graphs; three model seeds")
    for extension in ["png", "pdf"]:
        fig.savefig(m.RESULTS / "gnn" / "figures" / ("learning_curve." + extension), dpi=180)
    plt.close(fig)


# Write measured completion status without changing the earlier report or email.
def write_status(m, stage, error=None):
    value = {"stage": stage, "completed": stage == "complete", "updated_epoch": time.time(),
             "error": (str(error) or type(error).__name__) if error is not None else None, "profile": m.MODAL_PROFILE,
             "original_results_preserved": True, "remaining_generation_authorized": True}
    population = m.read_evidence(m.PREFLIGHT / "training_population.json")
    if population:
        value.update({"saved_graphs": population["saved"], "usable_graphs": population["usable"], "not_run": 0})
    m.save_json(m.RESULTS / "status.json", value)
    lines = ["# Full 100,000 graph experiment", "", "Status: " + stage + ".", "",
             "This is a separate continuation of the original experiment. Its previous 63,288-usable results and archive are preserved.", "",
             "The earlier run stopped following a connection/completion-tracking issue. It did not hit the spend limit.", ""]
    if error:
        lines.extend(["Execution stopped: " + str(error), ""])
    if population:
        lines.extend(["Saved: %d/100000. Usable: %d (%.4f%%). Not run: 0. All 63,672 original saved records are unchanged." %
                      (population["saved"], population["usable"], population["usable"] / 1000), ""])
        statistics = m.read_evidence(m.PREFLIGHT / "stage3_gate.json")
        if statistics:
            lines.extend(["Training, validation, test and model-selection subset counts: " + json.dumps(statistics["split_counts"]) + ".", ""])
    bakeoff = m.read_evidence(m.RESULTS / "gnn/bakeoff.json")
    if bakeoff and bakeoff.get("selection_provenance", {}).get("reused_prior_selection"):
        lines.extend(["Architecture D was selected in the completed original seven-model bake-off on the 63,288-usable population. That architecture was frozen for expanded-data retraining; the seven-model bake-off was not repeated. The selection provenance and original evidence hash are in gnn/bakeoff.json.", ""])
    elif bakeoff and bakeoff.get("selection_provenance", {}).get("reused_prior_selection") is False:
        lines.extend(["The seven-model bake-off was repeated on the expanded training population and fixed 2,000-graph validation subset at the user's explicit request. Winner: %s; policy: %s. Full comparisons and selection uncertainty are in gnn/bakeoff.json. The prior selection is preserved as gnn/bakeoff_prior_D.json." %
                      (bakeoff["selection"]["chosen"], bakeoff["strategy"]), ""])
        reused = m.read_evidence(m.PREFLIGHT / "expanded_winner_reuse_check.json")
        if reused and reused.get("passed"):
            lines.extend(["The winning configuration and policy exactly matched the three full D fits completed before the delayed request arrived. Those fits were reused after checking training-ID and scientific-source hashes. The matching partial learning-curve work is preserved as well; no three additional full fits are implied.", ""])
    final = m.read_evidence(m.RESULTS / "gnn" / "final_results.json")
    if final:
        lines.extend(["Selected policy: " + final["strategy"] + ".", "",
                      "| Held-out set | Total estimated work / random | Geometric mean ratio | Hit rate |",
                      "|---|---:|---:|---:|"])
        for tag in ["test_split", "large", "ian"]:
            metrics = final["sets"][tag][final["strategy"]]["summary"]["metrics"]
            predicted = final["sets"][tag][final["strategy"]]["records"]
            reference = final["sets"][tag]["random"]["records"]
            total = math.fsum(row["expected_cost"] for row in predicted) / math.fsum(row["expected_cost"] for row in reference)
            lines.append("| %s | %.6f | %.6f | %.4f |" % (tag, total, metrics["geometric_mean_cost_ratio"]["value"], metrics["hit_rate"]["value"]))
        lines.extend(["", "Circuit-work estimates include executed warm starts and estimated unexecuted random fallback; they exclude offline training and inference. They measure reaching empirical reference energies, not certified global minima or hardware runtime.", ""])
        lines.extend(["The selected post-processing follows the prespecified geometric-mean validation criterion. It need not minimize the ratio of summed work. All raw, snapped and better-policy results are preserved in gnn/final_results.json and the figures.", "",
                      "![Hit rate by node count](gnn/figures/hit_rate_by_n.png)", "",
                      "![Geometric mean work by node count](gnn/figures/cost_ratio_by_n.png)", ""])
    curve = m.read_evidence(m.RESULTS / "gnn" / "learning_curve.json")
    if curve:
        lines.extend(["## Nested training subsets", "", "| Fraction | Training graphs | Total estimated work / random | Geometric mean work ratio | Hit rate |", "|---|---:|---:|---:|---:|"])
        denominator = math.fsum(row["expected_cost"] for row in final["sets"]["test_split"]["random"]["records"])
        for percent in [25, 50, 100]:
            row = curve["fractions"][str(percent)]
            metrics = row["summary"]["metrics"]
            if percent == 100:
                records = final["sets"]["test_split"][final["strategy"]]["records"]
                numerator = math.fsum(record["expected_cost"] for record in records)
            else:
                numerator = math.fsum(math.fsum(record["expected_cost"] for record in m.read_evidence(
                    m.RESULTS / "gnn" / "evaluations" / ("curve%d_s%d_test.json" % (percent, seed)))["results"][final["strategy"]]) for seed in [42, 43, 44]) / 3
            lines.append("| %d%% | %d | %.6f | %.6f | %.4f |" % (percent, row["graphs"], numerator / denominator,
                         metrics["geometric_mean_cost_ratio"]["value"], metrics["hit_rate"]["value"]))
        lines.extend(["", "The subsets share held-out graphs. Larger datasets also receive more updates under the common epoch cap; this is not a compute-matched causal data-volume experiment.", ""])
        lines.extend(["![Nested training subsets](gnn/figures/learning_curve.png)", ""])
    bill = m.read_evidence(m.RESULTS / "gnn" / "billing_summary.json")
    if bill:
        lines.extend(["Cumulative posted cost: $%.8f. Upper reservation: $%.8f. Stop: $280, including the original experiment." %
                      (bill["reported_usd"], bill["upper_reservation_usd"]), ""])
    floor_review = m.read_evidence(m.PREFLIGHT / "reference_floor_review.json")
    if floor_review:
        lines.extend(["Independent final-energy review found %d reference labels with verified feasible improvements. Primary targets were preserved; see preflight/reference_floor_review.json." % floor_review["known_counterexample_graphs"], ""])
    lines.extend(["Reproducibility checks do not prove mathematical uniqueness or global optimality. All expected-id gates remain automatic; missing and pickless rows count as failures.", ""])
    (m.RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


# Resume only the existing recorded generation calls after a host interruption.
def generate_remaining(m):
    if (m.PREFLIGHT / "full_corpus_csv_gate.json").exists():
        return
    if (m.PREFLIGHT / "tail_recovery_authorized.json").exists():
        return recover_generation_tail(m)
    launch = m.read_evidence(m.PREFLIGHT / "full_corpus_launch.json")
    if launch is None:
        return m.launch_sharded_mode("full")
    import modal
    calls = m.read_evidence(m.PREFLIGHT / "full_corpus_calls.json")
    m.require(calls is not None and len(calls) == len(launch["jobs"]), "Incomplete call manifest requires reconciliation")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results, resource_errors = [], []
    pool = ThreadPoolExecutor(max_workers=len(calls))
    try:
        waiting = {}
        for job, info in zip(launch["jobs"], calls):
            m.require(job["batch"] == info["batch"], "Recorded generation call order differs")
            future = pool.submit(modal.FunctionCall.from_id(info["call_id"]).get)
            waiting[future] = job
        for future in as_completed(waiting):
            job = waiting[future]
            try:
                result = future.result()
            except Exception as error:
                if "stage resource reservation is exhausted" not in str(error):
                    raise
                resource_errors.append({"batch": job["batch"], "error": str(error)})
                m.save_json(m.PREFLIGHT / "full_resource_exhaustion.json", resource_errors)
                continue
            m.require(result["batch"] == job["batch"] and result["config_hash"] == launch["config_hash"], "Generation identity differs")
            m.require({v["shard"] for v in result["finished"]} == set(job["shards"]), "Generation shard completions are missing")
            results.append(result)
            m.save_json(m.PREFLIGHT / "full_corpus_batches.json", results)
            LOG.info("full: %d/%d whole batch receipts collected", len(results), len(calls))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        import subprocess
        subprocess.run([sys.executable, "-m", "modal", "app", "stop", launch["app_id"], "--profile", m.MODAL_PROFILE, "--yes"], check=False)
        m.register_paid_app(launch["app_id"], "corpus full", 80 * 8 * (.0473 + .008), stopped=True, budget_cap_usd=75)
    if resource_errors:
        m.save_json(m.PREFLIGHT / "tail_recovery_authorized.json", {
            "enabled": True, "original_app": launch["app_id"], "cause": "Original generation stage resource allowance exhausted",
            "authorization": "User approved every remaining graph within the cumulative $280 stop",
            "new_stage": "Only missing IDs, one CPU per worker, independent resource tokens and future-core budget gate",
            "failed_calls": resource_errors, "created_epoch": time.time()})
        return recover_generation_tail(m)
    m.download_volume("dataset", m.RESULTS / "dataset")
    config = m.load_frozen_plan()
    dataset = m.configured_dataset("full", config)
    expected = m.expected_manifest(dataset.selected("full"))
    records = m.load_records(m.RESULTS / "dataset", expected)
    m.require(set(records) == set(expected), "Completed generation omits expected graphs")
    m.save_json(m.PREFLIGHT / "full_corpus_coverage.json", m.dataset_gate(expected, records))
    dataset.merge()
    m.save_json(m.PREFLIGHT / "full_corpus_csv_gate.json", m.csv_gate(dataset.MERGED, [i for i, row in records.items() if m.has_pick(row, expected[i])]))


# Run one child at a time on one reserved CPU, preserving the unchanged filter and shard format.
def remote_tail_batch(job, hashes, config_hash):
    import subprocess
    import sys
    import time
    from pathlib import Path
    import modal
    sys.path.insert(0, "/root/qaoa/src")
    import modal_app as m
    m.configure_threads()
    deadline = time.monotonic() + m.take_resource_token(job["budget_queue"])
    m.PREFLIGHT = m.RESULTS / "worker_proofs/tail" / job["name"]
    for relative, expected in hashes.items():
        m.require(m.sha256(Path("/root/qaoa") / relative) == expected, "Tail worker source differs")
    m.check_input_hashes()
    m.stage_zero()
    config = m.RESULTS / "preflight/frozen_plan.json"
    m.require(m.sha256(config) == config_hash, "Tail graph recipe differs")
    volume = modal.Volume.from_name(m.VOLUME_NAME)
    child, output, last_commit = None, None, time.monotonic()

    # Retry only bounded storage timeouts, reserving resource time before every attempt.
    def commit_with_reservation():
        nonlocal deadline
        for attempt in range(3):
            m.require(time.time() < job["deadline_epoch"], "Tail resource deadline reached")
            if time.monotonic() >= deadline - 45:
                deadline += m.take_resource_token(job["budget_queue"])
            try:
                m.commit_volume(volume)
                return
            except TimeoutError:
                if attempt == 2:
                    raise
                print("Retrying timed-out Volume commit within the reserved resource slice", flush=True)
    try:
        for shard in job["shards"]:
            path = m.RESULTS / "dataset" / ("shard_%04d_of_4000.jsonl" % shard)
            m.repair_partial_shard(path)
            logs = m.RESULTS / "dataset/logs"
            logs.mkdir(parents=True, exist_ok=True)
            output = (logs / ("tail_%04d.log" % shard)).open("a", encoding="utf-8")
            child = subprocess.Popen([sys.executable, "/root/qaoa/src/modal_app.py", "shard", "full", str(shard),
                                      "4000", str(config), config_hash], stdout=output, stderr=subprocess.STDOUT)
            while child.poll() is None:
                m.require(time.time() < job["deadline_epoch"], "Tail resource deadline reached")
                if time.monotonic() >= deadline - m.RESOURCE_LEAD_SECONDS:
                    deadline += m.take_resource_token(job["budget_queue"])
                if time.monotonic() - last_commit >= job["commit_interval"]:
                    commit_with_reservation()
                    last_commit = time.monotonic()
                time.sleep(.5)
            output.close()
            m.require(child.returncode == 0, "Tail shard subprocess failed; inspect its log")
            commit_with_reservation()
        return {"name": job["name"], "shards": job["shards"], "completed": True}
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if output is not None:
            output.close()
        m.commit_volume(volume)


# Download only committed numerical rows; retain evidence of any interrupted final fragment.
def download_tail_records(m, expected):
    from concurrent.futures import ThreadPoolExecutor
    import modal
    volume = modal.Volume.from_name(m.VOLUME_NAME)
    paths = [entry.path for entry in volume.iterdir("dataset", recursive=False)
             if entry.type == modal.volume.FileEntryType.FILE and entry.path.endswith(".jsonl")]

    # Copy each shard once, without the large, unnecessary restart logs.
    def copy(relative):
        path = m.RESULTS / relative
        m.require(path.resolve().is_relative_to((m.RESULTS / "dataset").resolve()), "Tail path escapes dataset")
        raw = b"".join(volume.read_file(relative))
        path.write_bytes(raw)
        removed = m.repair_partial_shard(path)
        if removed:
            evidence = m.PREFLIGHT / "interrupted_fragments" / (path.name + ".bin")
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_bytes(raw[-removed:])
        return len(raw)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(copy, paths))
    return m.load_records(m.RESULTS / "dataset", expected)


# Finish only unsaved IDs with one CPU per worker; wait for posted billing when needed.
def recover_generation_tail(m):
    import modal
    dataset = m.configured_dataset("full", m.load_frozen_plan())
    expected = m.expected_manifest(dataset.selected("full"))
    records = download_tail_records(m, expected)
    for attempt in range(1, 6):
        missing = sorted(set(expected) - set(records))
        m.save_json(m.PREFLIGHT / "tail_missing_ids.json", {"ids": missing, "saved": len(records),
                    "not_run": len(missing), "failed": 0, "captured_epoch": time.time()})
        if not missing:
            m.save_json(m.PREFLIGHT / "full_corpus_coverage.json", m.dataset_gate(expected, records))
            dataset.merge()
            m.save_json(m.PREFLIGHT / "full_corpus_csv_gate.json", m.csv_gate(dataset.MERGED,
                        [i for i, row in records.items() if m.has_pick(row, expected[i])]))
            return
        m.require(attempt <= 4, "Tail recovery remains incomplete after four bounded attempts")
        future = HISTORICAL_FINAL_USD + HISTORICAL_CURVE_USD
        if not (m.RESULTS / "gnn/bakeoff.json").exists():
            future += HISTORICAL_BAKEOFF_USD
        future = RUNTIME_HEADROOM * future * (100000 / 63288) + 2
        wait_until = time.monotonic() + 7200
        while True:
            bill = m.write_billing_summary()
            reserve = min(6., m.SPEND_STOP_USD - bill["upper_reservation_usd"] - future - .05)
            if reserve >= .75:
                break
            m.require(time.monotonic() < wait_until, "No budget headroom for the tail and remaining full experiment")
            LOG.info("Waiting for posted billing before funding the remaining %d IDs", len(missing))
            time.sleep(30)
        pending = {}
        for graph_id in missing:
            pending.setdefault(graph_id % 4000, []).append(graph_id)
        count = min(32, len(pending))
        buckets, loads = [[] for unused in range(count)], [0] * count
        for shard, ids in sorted(pending.items(), key=lambda pair: -sum(2 ** expected[i] for i in pair[1])):
            k = min(range(count), key=lambda index: loads[index])
            buckets[k].append(shard)
            loads[k] += sum(2 ** expected[i] for i in ids)
        quantum, hourly = 180, count * (.0473 + 2 * .008)
        tokens = int((reserve - .25) / (quantum * (.0473 + 2 * .008) / 3600))
        m.require(tokens > count, "Tail token allocation cannot cover its workers")
        queue_name = "qaoa-full-tail-" + str(int(time.time()))
        modal.Queue.from_name(queue_name, create_if_missing=True).put_many([quantum] * tokens, block=False)
        hashes = {name: m.sha256(m.ROOT / name) for name in m.UPLOAD_FILES}
        hashes["src/complete_full100k.py"] = m.sha256(ROOT / "src/complete_full100k.py")
        image = m.data_image().add_local_file(ROOT / "src/complete_full100k.py", m.REMOTE_ROOT + "/src/complete_full100k.py", copy=True)
        app = modal.App("weighted-maxcut-full100k-tail")
        function = app.function(image=image, cpu=(1, 1), memory=(1024, 2048), timeout=7200, retries=0,
                                max_containers=count, scaledown_window=2, serialized=True,
                                volumes={m.REMOTE_ROOT + "/results": modal.Volume.from_name(m.VOLUME_NAME)},
                                name="one_cpu_tail")(remote_tail_batch)
        job_name = "tail_" + str(int(time.time()))
        jobs = [{"name": job_name + "_" + str(k), "shards": shards, "budget_queue": queue_name,
                 "commit_interval": 90 + (k % 16) * 3,
                 "deadline_epoch": time.time() + 7200} for k, shards in enumerate(buckets)]
        retry = False
        try:
            with app.run(detach=True):
                m.register_paid_app(app.app_id, job_name, hourly, budget_cap_usd=reserve)
                calls = []
                for job in jobs:
                    call = function.spawn(job, hashes, m.sha256(m.PREFLIGHT / "frozen_plan.json"))
                    calls.append({"call_id": call.object_id, "job": job})
                    m.save_json(m.PREFLIGHT / (job_name + "_launch.json"), {"app_id": app.app_id, "calls": calls,
                                "reserve_usd": reserve, "future_core_projection_usd": future, "prior_upper_usd": bill["upper_reservation_usd"],
                                "initial_tokens": tokens, "token_seconds": quantum, "hashes": hashes})
                for info in calls:
                    try:
                        result = modal.FunctionCall.from_id(info["call_id"]).get()
                        m.require(result["completed"], "Tail batch is incomplete")
                    except Exception as error:
                        if "resource reservation is exhausted" not in str(error) and "Tail resource deadline reached" not in str(error):
                            raise
                        retry = True
                        LOG.warning("Tail resource allowance ended; preserving completed rows")
                        break
        finally:
            if app.app_id:
                app.stop()
                m.register_paid_app(app.app_id, job_name, hourly, stopped=True, budget_cap_usd=reserve)
        m.write_billing_summary()
        records = download_tail_records(m, expected)
        m.require(retry or set(records) == set(expected), "Tail jobs returned without complete expected-ID coverage")
    raise RuntimeError("Tail recovery remains incomplete after four bounded attempts")


# Run every approved phase with fresh outputs and cumulative billing gates.
def main():
    if os.environ.get("QAOA_WAIT_FOR_PREPARATION") == "1":
        deadline = time.monotonic() + PREPARATION_WAIT_SECONDS
        ready = EXPERIMENT / "results" / "preflight" / "continuation_inputs_ready.json"
        while not ready.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("Input preparation did not finish; no compute was launched")
            time.sleep(5)
    source = prepare_sources()
    sys.path.insert(0, str(EXPERIMENT / "src"))
    m = importlib.import_module("modal_app")
    m.require(m.ROOT == EXPERIMENT and m.VOLUME_NAME == VOLUME_NAME, "Continuation imports escaped its root")
    m.PREFLIGHT.mkdir(parents=True, exist_ok=True)
    temporary = m.RESULTS / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary.resolve())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(m.RESULTS / "continuation.log", encoding="utf-8")])
    for name in ["hpack", "h2", "grpclib", "modal", "asyncio"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    stage = "preflight"
    try:
        m.lock_profile()
        m.check_input_hashes()
        m.stage_zero()
        m.verify_workspace()
        prepare_evidence(m, source)
        if sys.argv[1:] == ["expanded-bakeoff"]:
            stage = "expanded seven-model bakeoff"
            return repeat_expanded_bakeoff(m)
        m.require(not sys.argv[1:], "Unknown coordinator arguments")
        reuse_previous_selection(m)
        reconcile_expanded_selection(m)
        if PREPARE_ONLY:
            return 0
        spending = m.reconciled_spend_reservation()
        launch = m.read_evidence(m.PREFLIGHT / "full_corpus_launch.json")
        if launch is None:
            m.budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"],
                          GENERATION_RESERVE_USD + TRAINING_PLAN_RESERVE_USD)
        else:
            m.require(spending["spend_reservation_usd"] < m.SPEND_STOP_USD,
                      "Cumulative spend reservation reached the $280 stop")
        stage = "generation"
        write_status(m, stage)
        generate_remaining(m)
        freeze_complete_population(m)
        stage = "data verification"
        write_status(m, stage)
        if not (m.PREFLIGHT / "stage3_gate.json").exists():
            phase_budget(m, "prepare")
            config = m.load_frozen_plan()
            m.describe_corpus("full", config)
            m.describe_corpus("test", config)
            m.prepare_training_artifacts()
        stage = "bakeoff"
        write_status(m, stage)
        if not (m.RESULTS / "gnn" / "bakeoff.json").exists():
            phase_budget(m, "bakeoff")
            with m.training_session("bakeoff"):
                m.run_bakeoff()
        stage = "final training and evaluation"
        write_status(m, stage)
        if not (m.RESULTS / "gnn" / "final_results.json").exists():
            phase_budget(m, "final")
            m.run_final_training()
            m.run_final_evaluation()
        stage = "25/50/100 percent comparison"
        write_status(m, stage)
        if not (m.RESULTS / "gnn" / "learning_curve.json").exists():
            phase_budget(m, "curve")
            m.run_learning_curve()
        stage = "final independent numerical review"
        write_status(m, stage)
        final_numerical_review(m)
        draw_completed_figures(m)
        m.write_billing_summary()
        stage = "archive verification"
        write_status(m, stage)
        archive_completed_experiment(m)
        m.write_billing_summary()
        write_status(m, "complete")
        return 0
    except BaseException as error:
        LOG.exception("Full experiment stopped during %s", stage)
        m.stop_task_apps()
        try:
            m.write_billing_summary()
        except Exception:
            LOG.exception("Billing refresh failed after stop; keep existing reservations")
        write_status(m, "stopped during " + stage, error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
