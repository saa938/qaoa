"""Weighted p=1 MA-QAOA comparison, raw evidence, technical report and email draft.

Usage: python src/paper_comparison.py [prepare|run|report|selfcheck]
No arguments run the approved dependency-aware continuation on tejas-16842 only.
"""

import csv
import hashlib
import importlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = ROOT / "full_100k"
OUTPUT = EXPERIMENT / "results" / "paper_comparison"
SEEDS = [42, 43, 44]
EXTRA_RESERVE_USD = 12.0
GPU_SECONDS = 4 * 3600
CPU_SECONDS = 3600
CPU_CONTAINERS = 40
CALL_BUDGET = 2000
HIT_TOL = 1e-6
PAPER_PDF_SHA256 = "10b3a21ef02b8ba138fb2acb6d90fc8ee4161b1920a6fba982cbcdd5e7b6c50f"
LOG = logging.getLogger("paper_comparison")


# Read optional evidence without treating missing files as successful experiments.
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


# Atomically replace small coordination records on the same filesystem.
def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


# Hash exact source or output bytes, including their newline convention.
def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# Import the isolated approved runner rather than the original completed experiment.
def runner():
    sys.path.insert(0, str(EXPERIMENT / "src"))
    module = importlib.import_module("modal_app")
    module.require(module.ROOT == EXPERIMENT and module.MODAL_PROFILE == "tejas-16842", "Comparison root/profile differs")
    module.lock_profile()
    return module


# Record the scientific questions and all adaptations before looking at new held-out results.
def prepare():
    protocol = {
        "task": "Predict p=1 weighted MA-QAOA parameters from one filter-selected label per graph",
        "weighted_only": True, "new_graph_generation": False,
        "seeds": SEEDS, "epochs": 30, "batch_size": 256,
        "seed_completion_policy": "Complete seed 42 for all three models first; add 43/44 only when measured runtime plus 30% projects a whole matched trio within the remaining GPU window. Use runtime/budget, never test performance, for this decision. Report the actual seed count.",
        "optimizer": "AdamW, constant lr .001, weight decay .0001, clip norm 1",
        "selection": "minimum model-native validation loss; test labels never select checkpoints",
        "dataset": "completed expanded corpus; exactly the core training/validation/test IDs and physical labels",
        "paper_gnn": {"architecture": "5 GINE layers; hidden 192; residual; BatchNorm; dropout .1",
                      "loss": "1-cos(raw angle error) plus .02 abs(raw vector norm-1), beta plus directed gamma blocks",
                      "representation": "actual RZZ rotation r and paper mixer b represented by cos/sin; atan2 decoding",
                      "angle_units": "Weighted extension: r=-2*w*gamma_filter and b=-beta_filter; RZZ(r)=exp(-i*r*ZZ/2), RX(2*b)=exp(-i*b*X). Decode theta_filter=-r/2 and beta_filter=-b. Absorb weight into the independent edge gate rotation; original paper experiments were unweighted.",
                      "adaptations": ["append edge weight", "same grouped 80/10/10 split as all comparison models",
                                      "canonical ordering unspecified in paper: use degree/weighted-degree rank with averaged ties",
                                      "deterministic eigenvector/eigenspace orientation from the existing weighted feature code",
                                      "mean raw vectors of both edge directions at inference",
                                      "aggregate active angles within each minibatch; average directed edge losses"],
                      "status": "documented paper-based reproduction; original implementation unavailable"},
        "ours_30": "winning core architecture, native supervised loss, 30-epoch common optimizer; no energy fine-tuning",
        "core_final": "Inference from the saved full-recipe winning GNN checkpoints on the same graphs and query budgets; training-ID and checkpoint hashes checked; no additional full-recipe fit",
        "historical_selection": "Repeat the seven-model bake-off on the expanded training split using the fixed validation-selection IDs, as explicitly requested. Preserve and reuse completed D fits only if the new winning configuration and strategy match. Disclose prior historical validation exposure and report the newly-generated test cohort separately.",
        "diffusion": {"source": "https://link.springer.com/article/10.1140/epjqt/s40507-025-00397-4",
                      "status": "MA-QAOA adaptation, not a reproduction of the fixed-six-angle standard-QAOA experiment",
                      "graph_denoiser": "4 GPS layers; hidden 128; weighted features; noisy circular coordinates; timestep embedding",
                      "coordinates": "theta=w*gamma and beta, each embedded as [cos(2 angle),sin(2 angle)]",
                      "training": "1000 noise levels; linear beta .0001 to .02; predict Gaussian noise; graph-normalized node+edge MSE",
                      "sampling": "50 deterministic DDIM steps, four independent initial noise draws; no test-label guidance",
                      "single_label": "one clean parameter vector per graph; noise augmentation does not create new independent labels"},
        "evaluation": {"sets": ["weighted held-out test split", "weighted n=13-14 extrapolation set"],
                       "large_denominator": "All 600 planned n=13-14 graphs; the 16 unusable references count as failed reference/label hits while physical energies remain measurable",
                       "primary": "raw head/sample 0, exactly one proposal; direct energy and local polish reported separately",
                       "secondary": "K=4 proposals ranked by measured initial energy, charge all ranking and polish evaluations",
                       "paper_K4": "one prediction plus three independent fixed sigma=.05 phase perturbations; explicitly a jitter control",
                       "postprocessing": ["raw", "snap each phase to nearest pi/8"],
                       "budget": "2000 energy/joint-energy-gradient queries per graph/method/policy/K, shared across all proposals",
                       "optimizer": "unchanged run_filter.polish L-BFGS-B; ftol=1e-15, gtol=1e-12, maxiter=4000, wrapped query cap",
                       "baselines": ["seeded uniform random K=1,4", "exact zero", "noisy zero K=1,4", "reference label oracle"],
                       "reference_success": "physical energy <= saved empirical reference floor + 1e-6",
                       "label_success": "period-aware physical-phase L2 distance <= 1e-3; raw output and final polished point",
                       "physics": "independently recompute every final polish energy; preserve result.fun separately",
                       "runtime": "same one-thread CPU image; objective, optimizer and independent verification timing kept separate",
                       "missing": "any missing expected graph/model/policy is a failed gate; never shrink the denominator"},
        "budget": {"cumulative_stop_usd": 280, "additional_projection_usd": EXTRA_RESERVE_USD,
                   "one_GPU_at_a_time": True, "dependency": "core pipeline completes or stops before additional paid experiments"},
        "no_claims": ["filter agreement is not a mathematical uniqueness/global-optimality certificate",
                      "positive classical gap does not establish failed p=1 optimization",
                      "equal epochs and proposal counts are not equal training FLOPs",
                      "estimated parameter-shift circuit work is not actual simulator/hardware runtime"],
    }
    protocol["paper_pdf_sha256"] = PAPER_PDF_SHA256
    # The verified paper identity is portable; redistribution of the PDF is unnecessary.
    paper_path = os.environ.get("QAOA_PAPER_PDF")
    if paper_path and digest(Path(paper_path)) != PAPER_PDF_SHA256:
        raise RuntimeError("QAOA_PAPER_PDF differs from the verified reference paper")
    path = OUTPUT / "protocol.json"
    if path.exists() and read(path) != protocol:
        previous = read(path)
        changed = {key for key in set(previous) | set(protocol) if previous.get(key) != protocol.get(key)}
        request = read(EXPERIMENT / "results/preflight/expanded_bakeoff_user_request.json")
        allowed = {"seed_completion_policy"}
        if request and request.get("approved") is True:
            allowed.add("historical_selection")
        review = read(OUTPUT / "paper_angle_convention_review.json")
        if "paper_gnn" in changed and review and review.get("roundtrip_passed") is True:
            old_paper, new_paper = dict(previous["paper_gnn"]), dict(protocol["paper_gnn"])
            for field in ["representation", "angle_units"]:
                old_paper.pop(field, None)
                new_paper.pop(field, None)
            if old_paper == new_paper and review.get("previous_protocol_sha256") == digest(path):
                allowed.add("paper_gnn")
        if (OUTPUT / "training_session.json").exists() or not changed <= allowed:
            raise RuntimeError("A comparison protocol already exists with different settings; preserve evidence before changing it")
        backup = OUTPUT / ("protocol_before_paper_unit_correction.json" if "paper_gnn" in changed else
                           "protocol_before_expanded_selection.json" if "historical_selection" in changed
                           else "protocol_before_runtime_policy.json")
        if not backup.exists():
            save(backup, previous)
    save(path, protocol)
    save(OUTPUT / "status.json", {"stage": "prepared", "completed": False, "updated_epoch": time.time(),
                                 "note": "No diffusion/paper comparison compute is launched by prepare"})
    return protocol


# Stop this worker process at its funded deadline, including blocked storage/import calls.
def resource_watchdog(deadline_epoch):
    import threading
    timer = threading.Timer(max(0., deadline_epoch - time.time()), lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    return timer


# Import GPU-only dependencies under a deadline that also covers input and checkpoint IO.
def training_worker(job, hashes, manifest_hash):
    timer = resource_watchdog(job["deadline_epoch"])
    try:
        import paper_model
        return paper_model.remote_train(job, hashes, manifest_hash)
    finally:
        timer.cancel()


# Count objective calls and keep every trajectory point without claiming a failed polish hit.
def measured_polish(row, start, allowance, deadline, reference_valid):
    import run_filter
    import modal_app as m
    setup_started = time.perf_counter()
    energy, unused, gradient, dimension = run_filter.make_energy(row["n"], row["edges"], row["w_sorted"], 1)
    setup_seconds = time.perf_counter() - setup_started
    history, best = [], [float("inf"), np.asarray(start, float)]
    started, cpu_started = time.perf_counter(), time.process_time()

    # Interrupt before spending an unreserved function evaluation.
    class Exhausted(Exception):
        pass

    # The pinned optimizer requests joint energy and adjoint gradient in one simulator call.
    def objective(point, with_energy=False):
        if len(history) >= allowance or time.time() >= deadline:
            raise Exhausted()
        value, derivative = gradient(point, True)
        history.append([len(history) + 1, float(value), time.perf_counter() - started])
        if value < best[0]:
            best[:] = [float(value), np.array(point, copy=True)]
        return (value, derivative) if with_energy else derivative

    try:
        result = run_filter.polish(energy, objective, start)
        point = result.x
        reported_energy = float(result.fun)
        status, message, success = int(result.status), str(result.message), bool(result.success)
        optimizer_nfev, optimizer_njev = int(result.nfev), int(result.njev)
        exhausted = False
    except Exhausted:
        point, reported_energy = best[1], best[0]
        status, message, success, exhausted = -1, "shared query or resource deadline reached", False, True
        optimizer_nfev = optimizer_njev = len(history)
    elapsed, cpu_elapsed = time.perf_counter() - started, time.process_time() - cpu_started
    if not history:
        raise RuntimeError("No reserved objective evaluation remained")
    verify_started = time.perf_counter()
    physical = float(m.independent_p1_final_energies(row, [point])[0])
    return {"angles_before": np.asarray(start).tolist(), "angles_after": np.asarray(point).tolist(),
            "energy_before": history[0][1], "energy_after": physical, "optimizer_reported_energy": reported_energy,
            "nfev": optimizer_nfev, "njev": optimizer_njev, "joint_simulator_calls": len(history),
            "optimizer_wall_seconds": elapsed, "optimizer_cpu_seconds": cpu_elapsed,
            "circuit_setup_wall_seconds": setup_seconds,
            "verification_wall_seconds": time.perf_counter() - verify_started,
            "status": status, "message": message, "optimizer_success": success, "budget_exhausted": exhausted,
            "reference_hit": bool(reference_valid and physical <= row["floor"] + HIT_TOL), "trajectory": history}


# Compute exact ideal-state sampling probabilities independently from optimizer metadata.
def measurement_statistics(row, point):
    import modal_app as mrunner
    point = mrunner.period_equivalent_angles(point, row["w_sorted"])
    n, edges = row["n"], row["edges"]
    weights = np.asarray(row["w_sorted"])
    m = len(edges)
    states = np.arange(1 << n, dtype=np.uint32)
    phase, cut = np.zeros(len(states)), np.zeros(len(states))
    for k, ((u, v), weight) in enumerate(zip(edges, weights)):
        crossing = ((states >> u) ^ (states >> v)) & 1
        cut += weight * crossing
        phase += point[k] * weight * (1 - 2 * crossing.astype(float))
    psi = np.exp(1j * phase) / np.sqrt(len(states))
    for j, beta in enumerate(point[m:]):
        view = psi.reshape(-1, 2, 1 << j)
        a, b = view[:, 0].copy(), view[:, 1].copy()
        view[:, 0] = np.cos(beta) * a + 1j * np.sin(beta) * b
        view[:, 1] = np.cos(beta) * b + 1j * np.sin(beta) * a
    probabilities = np.abs(psi) ** 2
    expected = float(probabilities @ cut)
    optimum = np.abs(cut - row["max_cut"]) <= 1e-8
    success = float(probabilities[optimum].sum())
    return {"expected_cut": expected, "cut_standard_deviation": float(np.sqrt(max(0., probabilities @ (cut * cut) - expected ** 2))),
            "optimal_cut_probability_one_shot": success,
            "optimal_cut_probability_100_shots": float(-np.expm1(100 * np.log1p(-min(success, 1 - 1e-16))))}


# Evaluate fixed candidates and include their energy ranking queries in every cost total.
def evaluate_case(row, phases, policy, count, deadline, reference_valid=None):
    import run_filter
    import modal_app as mrunner
    case_started = setup_started = time.perf_counter()
    weights = np.asarray(row["w_sorted"])
    m, n = len(weights), row["n"]
    if reference_valid is None:
        reference_valid = mrunner.usable(row, n)
    energy, unused, gradient, dimension = run_filter.make_energy(n, row["edges"], weights, 1)
    setup_seconds = time.perf_counter() - setup_started
    phases = np.asarray(phases, float)[:count]
    if phases.shape != (count, dimension) or not np.isfinite(phases).all():
        raise RuntimeError("Missing, malformed or nonfinite proposal")
    if policy == "snap":
        phases = np.round(phases / (np.pi / 8)) * (np.pi / 8)
    scales = np.r_[weights, np.ones(n)]
    started, ranked = time.perf_counter(), []
    for k, phase in enumerate(phases):
        if time.time() >= deadline:
            raise RuntimeError("CPU evaluation resource deadline reached")
        point = phase / scales
        ranked.append((float(energy(point)), k, point))
    ranking_seconds = time.perf_counter() - started
    ranked.sort(key=lambda item: item[0])
    polishes, calls = [], len(ranked)
    for initial, k, point in ranked:
        if calls >= CALL_BUDGET:
            break
        result = measured_polish(row, point, CALL_BUDGET - calls, deadline, reference_valid)
        result["proposal_index"] = k
        polishes.append(result)
        calls += result["joint_simulator_calls"]
        if result["reference_hit"]:
            break
    if not polishes:
        raise RuntimeError("No proposal was polished")
    best = min(polishes, key=lambda value: value["energy_after"])
    target = np.asarray(row["pick"]) * scales
    delta = (phases[ranked[0][1]] - target + np.pi / 2) % np.pi - np.pi / 2
    final_delta = (np.asarray(best["angles_after"]) * scales - target + np.pi / 2) % np.pi - np.pi / 2
    continuous_label = bool((np.abs(target[:m, None] - np.asarray([0., np.pi / 4])).min(1) > 1e-3).any()
                            or (np.abs(target[m:, None] - np.arange(-2, 3) * np.pi / 8).min(1) > 1e-3).any())
    joint = sum(value["joint_simulator_calls"] for value in polishes)
    # The first-hit index includes every previously attempted proposal, not only the final one.
    offset = len(ranked)
    first_hit = min((index + 1 for value, index, point in ranked if reference_valid and value <= row["floor"] + HIT_TOL), default=None)
    for result in polishes:
        for call, value, elapsed in result["trajectory"]:
            if reference_valid and value <= row["floor"] + HIT_TOL and first_hit is None:
                first_hit = offset + call
        offset += result["joint_simulator_calls"]
    sampling_started = time.perf_counter()
    sampling = measurement_statistics(row, np.asarray(best["angles_after"]))
    if abs(-sampling["expected_cut"] - best["energy_after"]) > 1e-8:
        raise RuntimeError("Independent ideal-state sampling calculation disagrees with physical energy")
    thresholds = {"shared_999": .999 * row["floor"], "self_999": .999 * best["energy_after"]}
    first_queries = {}
    for key, threshold in thresholds.items():
        if key == "shared_999" and not reference_valid:
            first_queries["first_" + key + "_query"] = None
            continue
        offset = len(ranked)
        first = min((index + 1 for value, index, point in ranked if value <= threshold), default=None)
        for result in polishes:
            for call, value, elapsed in result["trajectory"]:
                if value <= threshold and first is None:
                    first = offset + call
            offset += result["joint_simulator_calls"]
        first_queries["first_" + key + "_query"] = first
    return {"id": row["id"], "n": n, "type": row["type"], "policy": policy, "proposals": count,
            "continuous_label": continuous_label,
            "reference_valid": reference_valid,
            "reference_energy": row["floor"], "max_cut": row["max_cut"],
            "energy_before": ranked[0][0], "energy_after": best["energy_after"],
            "reference_gap_before": ranked[0][0] - row["floor"],
            "reference_gap_after": best["energy_after"] - row["floor"],
            "cut_ratio_before": -ranked[0][0] / row["max_cut"],
            "cut_ratio_after": -best["energy_after"] / row["max_cut"],
            "direct_reference_hit": bool(reference_valid and ranked[0][0] <= row["floor"] + HIT_TOL),
            "reference_hit": best["reference_hit"],
            "classical_bound_hit": best["energy_after"] <= -row["max_cut"] + HIT_TOL,
            "phase_l2_before": float(np.linalg.norm(delta)), "phase_l2_after": float(np.linalg.norm(final_delta)),
            "label_hit_before": bool(reference_valid and np.linalg.norm(delta) <= 1e-3),
            "label_hit_after": bool(reference_valid and np.linalg.norm(final_delta) <= 1e-3),
            "ranking_energy_calls": len(ranked), "joint_simulator_calls": joint, "total_function_queries": calls,
            "parameter_shift_equivalent_calls": len(ranked) + (2 * dimension + 1) * joint,
            "first_reference_hit_query": first_hit, "ranking_wall_seconds": ranking_seconds,
            "optimizer_wall_seconds": sum(v["optimizer_wall_seconds"] for v in polishes),
            "optimizer_cpu_seconds": sum(v["optimizer_cpu_seconds"] for v in polishes),
            "circuit_setup_wall_seconds": setup_seconds + sum(v["circuit_setup_wall_seconds"] for v in polishes),
            "verification_wall_seconds": sum(v["verification_wall_seconds"] for v in polishes),
            "sampling_verification_wall_seconds": time.perf_counter() - sampling_started,
            "evaluation_wall_seconds": time.perf_counter() - case_started,
            **sampling, **first_queries, "polishes": polishes}


# Read only hash-verified model predictions and use every expected held-out graph.
def evaluation_body(job, hashes, manifest_hash):
    import platform
    import modal
    import modal_app as m
    m.configure_threads()
    for relative, expected in hashes.items():
        m.require(m.sha256(m.ROOT / relative) == expected, "Evaluation source hash differs")
    m.require(m.sha256(m.RESULTS / "gnn/corpus_manifest.json") == manifest_hash, "Evaluation corpus differs")
    rows = read_large_all(m, job["large_all"], job["ids"]) if job["corpus"] == "large" else m.read_corpus_rows(job["corpus"], job["ids"])
    predictions = {}
    for name, info in job["predictions"].items():
        path = m.RESULTS / info["path"]
        m.require(m.sha256(path) == info["sha256"], "Evaluation predictions differ")
        predictions[name] = {r["id"]: r for r in json.loads(path.read_text())}
    path = m.RESULTS / "paper_comparison/evaluations" / job["corpus"] / ("batch_%04d.jsonl" % job["batch"])
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            m.require(time.time() < job["deadline_epoch"], "Evaluation reservation expired")
            rng = np.random.default_rng([42, row["id"], 90491])
            dimension = row["n"] + len(row["edges"])
            candidates = {
                "random": rng.uniform(-np.pi / 2, np.pi / 2, (4, dimension)),
                "noisy_zero": rng.normal(0, .05, (4, dimension)), "zero": np.zeros((1, dimension)),
                "oracle": np.asarray([np.r_[np.asarray(row["pick"][:len(row["edges"])]) * row["w_sorted"],
                                                row["pick"][len(row["edges"]):]]])}
            for name, records in predictions.items():
                m.require(row["id"] in records, "Prediction is missing an expected graph")
                proposed = records[row["id"]]["phase_candidates"]
                if len(proposed) == 1:
                    jitter_rng = np.random.default_rng([int(name.rsplit("s", 1)[-1]), row["id"], 192])
                    proposed = np.r_[np.asarray(proposed), np.asarray(proposed) + jitter_rng.normal(0, .05, (3, dimension))]
                candidates[name] = proposed
            cases = [(name, policy, k) for name, proposed in candidates.items()
                     for policy in (["raw", "snap"] if name in predictions else ["raw"])
                     for k in ([1, 4] if len(proposed) >= 4 else [1])]
            order = np.random.default_rng([row["id"], 20261031]).permutation(len(cases))
            for position, index in enumerate(order):
                name, policy, k = cases[int(index)]
                result = evaluate_case(row, candidates[name], policy, k, job["deadline_epoch"])
                result.update({"model": name, "corpus": job["corpus"], "batch": job["batch"], "case_execution_order": position})
                if name in predictions:
                    result["inference_amortized_seconds"] = predictions[name][row["id"]].get("inference_amortized_seconds")
                    result["inference_candidates_generated"] = predictions[name][row["id"]].get(
                        "generated_candidates", len(predictions[name][row["id"]]["phase_candidates"]))
                handle.write(json.dumps(result, allow_nan=False) + "\n")
                count += 1
            handle.flush()
    m.commit_volume(modal.Volume.from_name(m.VOLUME_NAME))
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"batch": job["batch"], "corpus": job["corpus"], "ids": job["ids"], "records": count,
            "path": path.relative_to(m.RESULTS).as_posix(), "sha256": m.sha256(path),
            "cpu": cpu, "platform": platform.platform(), "threads": 1}


# Include final Volume writes in the same global CPU resource deadline.
def evaluation_worker(job, hashes, manifest_hash):
    timer = resource_watchdog(job["deadline_epoch"])
    try:
        return evaluation_body(job, hashes, manifest_hash)
    finally:
        timer.cancel()


# Require a fresh all-phase reservation before allocating any additional paid container.
def budget(m, remaining):
    value = m.write_billing_summary()
    m.require(value["upper_reservation_usd"] + remaining < m.SPEND_STOP_USD,
              "Comparison would exceed the cumulative $280 stop")
    save(OUTPUT / "cost_plan.json", {"checked_epoch": time.time(), "prior_upper_usd": value["upper_reservation_usd"],
                                   "additional_projection_usd": remaining, "cumulative_stop_usd": 280})
    return value


# Copy both extension modules into the exact core scientific image without editing pinned files.
def image(m, gpu):
    result = m.gpu_image() if gpu else m.data_image().pip_install("mpmath==1.3.0")
    for name in ["paper_model.py", "paper_comparison.py"]:
        result = result.add_local_file(ROOT / "src" / name, m.REMOTE_ROOT + "/src/" + name, copy=True)
    return result


# Pin extension bytes as well as every unchanged scientific source copied by the runner.
def source_hashes(m):
    result = {name: m.sha256(m.ROOT / name) for name in m.UPLOAD_FILES}
    result.update({"src/" + name: digest(ROOT / "src" / name) for name in ["paper_model.py", "paper_comparison.py"]})
    return result


# Freeze all 600 planned larger graphs, retaining unusable-label flags rather than dropping IDs.
def prepare_large_all(m):
    import modal
    expected = {i: 13 if i < 300 else 14 for i in range(600)}
    rows = m.load_records(m.RESULTS / "dataset_test", expected)
    m.require(set(rows) == set(expected), "All 600 planned larger graphs must be saved")
    m.require(all(m.has_pick(row, expected[i]) and isinstance(row.get("floor"), (int, float))
                  and math.isfinite(row["floor"]) for i, row in rows.items()),
              "A missing/pickless larger reference cannot disappear from the expected-ID list")
    path = OUTPUT / "large_all.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i in sorted(rows):
            handle.write(json.dumps(rows[i], allow_nan=False) + "\n")
    info = {"path": path.relative_to(m.RESULTS).as_posix(), "sha256": digest(path), "expected_ids": sorted(expected),
            "label_usable": sum(m.usable(row, expected[i]) for i, row in rows.items()), "planned_graphs": 600}
    with modal.Volume.from_name(m.VOLUME_NAME).batch_upload(force=True) as batch:
        batch.put_file(path, info["path"])
    save(OUTPUT / "large_all_manifest.json", info)
    return info


# Load the same hash-verified complete larger-test corpus on CPU and GPU workers.
def read_large_all(m, info, ids=None):
    path = m.RESULTS / info["path"]
    m.require(path.resolve().is_relative_to(m.RESULTS.resolve()) and m.sha256(path) == info["sha256"],
              "Complete larger-test input bytes differ")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    m.require(len(rows) == len(by_id) == 600 and set(by_id) == set(range(600)), "Complete larger-test coverage differs")
    m.require(all(row["n"] == (13 if row["id"] < 300 else 14) for row in rows), "Wrong larger-test node count")
    return [by_id[i] for i in (sorted(by_id) if ids is None else ids)]


# Fit nine fixed-seed models in one bounded GPU session and persist calls before waiting.
# Record earlier selection exposure and a fresh-generation test cohort without changing any split.
def historical_selection_provenance(m):
    original_population = read(ROOT / "results/preflight/training_population.json")
    original_splits = read(ROOT / "results/gnn/splits.json")
    current_splits = read(m.RESULTS / "gnn/splits.json")
    m.require(original_population is not None and original_splits is not None,
              "Original selection evidence is missing")
    saved = set(original_population["saved_ids"])
    fresh = set(original_population["not_run_ids"])
    test = set(current_splits["ids"]["test"])
    selection = set(original_splits["ids"]["val_selection"])
    m.require(not saved & fresh and saved | fresh == set(range(100000)), "Original generation cohorts differ")
    value = {"original_population_sha256": digest(ROOT / "results/preflight/training_population.json"),
             "original_splits_sha256": digest(ROOT / "results/gnn/splits.json"),
             "current_splits_sha256": digest(m.RESULTS / "gnn/splits.json"),
             "previously_saved_ids": sorted(saved), "newly_generated_ids": sorted(fresh),
             "historical_selection_ids": sorted(selection),
             "test_previously_saved": len(test & saved), "test_newly_generated": len(test & fresh),
             "test_historical_selection_overlap": sorted(test & selection),
             "current_bakeoff_sha256": digest(m.RESULTS / "gnn/bakeoff.json"),
             "current_selection": read(m.RESULTS / "gnn/bakeoff.json")["selection"]["chosen"],
             "current_selection_provenance": read(m.RESULTS / "gnn/bakeoff.json").get("selection_provenance", {}),
             "interpretation": "The expanded bake-off uses its frozen validation subset. D had also been selected in the earlier study, and its full expanded fits completed before the delayed request to repeat selection arrived. The expanded test split is held out from each new fit, but is not wholly unseen by the historical research process. The newly_generated subgroup was unavailable during that earlier selection; no additional fits or optimizer calls are needed for this subgroup report."}
    save(OUTPUT / "historical_selection_provenance.json", value)
    return value


def train_models(m):
    import modal
    hashes = source_hashes(m)
    manifest_hash = m.sha256(m.RESULTS / "gnn/corpus_manifest.json")
    historical_selection_provenance(m)
    completed = read(OUTPUT / "training_complete.json")
    if completed:
        m.require(all(value["source_hashes"] == hashes and value["corpus_manifest_sha256"] == manifest_hash
                      and len(value["history"]) == 30 and value["predictions"]["large"]["graphs"] == 600
                      for value in completed.values()), "Cached comparison uses different source, data or expected IDs")
        return completed
    large_all = prepare_large_all(m)
    bakeoff = m.read_evidence(m.RESULTS / "gnn/bakeoff.json")
    winner = bakeoff["selection"]["chosen"]
    configuration = {key: value for key, value in bakeoff["configuration"].items()
                     if key in ["baseline", "classification", "hypotheses", "hidden", "layers", "theta_classes"]}
    core_fits = m.read_evidence(m.RESULTS / "gnn/final_training.json")["seeds"]
    train_hash = hashlib.sha256(json.dumps(m.read_evidence(m.RESULTS / "gnn/splits.json")["ids"]["train"]).encode()).hexdigest()
    m.require(all(value["best"]["training"]["training_ids_sha256"] == train_hash for value in core_fits.values()),
              "The full-recipe models used a different training graph list")
    signature = {key: value for key, value in bakeoff["configuration"].items()
                 if key not in ["seed", "maximum_epochs"]}
    m.require(all({key: value for key, value in fit["best"]["training"]["config"].items()
                   if key not in ["seed", "maximum_epochs"]} == signature for fit in core_fits.values()),
              "The preserved full-recipe fits do not match the expanded bake-off winner")
    jobs = [{"name": kind + "_s" + str(seed), "kind": kind, "seed": seed,
             "configuration": configuration if kind == "ours30" else {"baseline": False}, "large_all": large_all}
            for seed in SEEDS for kind in ["paper_weighted", "ours30", "diffusion"]]
    for job in jobs:
        if job["kind"] == "ours30":
            core = core_fits[str(job["seed"])]
            job["core_checkpoint"] = {"path": core["final_checkpoint"], "sha256": core["best"]["training"]["checkpoint_sha256"],
                                      "selected_epoch": core["best"]["epoch"], "new_training": False}
    hourly = m.TRAINING_MAX_GPU_HOURLY_USD + 4 * .0473 + 16 * .008
    bill = m.write_billing_summary()
    cpu_reserve = CPU_CONTAINERS * (.0473 + 2 * .008) * CPU_SECONDS / 3600 + .9
    available = min(EXTRA_RESERVE_USD, m.SPEND_STOP_USD - bill["upper_reservation_usd"] - .05)
    gpu_seconds = min(GPU_SECONDS, int((available - cpu_reserve) / hourly * 3600))
    m.require(gpu_seconds >= 3600, "Remaining cumulative budget cannot fund even the first matched model trio")
    budget(m, gpu_seconds * hourly / 3600 + cpu_reserve)
    app = modal.App("weighted-maxcut-paper-diffusion-training")
    function = app.function(image=image(m, True), gpu=m.TRAINING_GPU_TYPES, cpu=(4, 4), memory=(16384, 16384),
                            timeout=3600, retries=0, max_containers=1, scaledown_window=45,
                            volumes={m.REMOTE_ROOT + "/results": modal.Volume.from_name(m.VOLUME_NAME)},
                            serialized=True, name="matched_training")(training_worker)
    results = {}
    deadline = time.time() + gpu_seconds
    session_started = time.perf_counter()
    launch = OUTPUT / "training_session.json"
    if launch.exists():
        previous = read(launch)
        m.require(previous["source_hashes"] == hashes, "Recorded comparison source differs; cannot resubmit")
        raise RuntimeError("A recorded comparison session must be reconciled before allocating a replacement GPU")
    try:
        with app.run(detach=True):
            m.register_paid_app(app.app_id, "matched paper/diffusion training", hourly, budget_cap_usd=hourly * gpu_seconds / 3600 + .25)
            save(launch, {"app_id": app.app_id, "deadline_epoch": deadline, "source_hashes": hashes,
                          "maximum_hourly_usd": hourly, "jobs": jobs, "manifest_hash": manifest_hash})
            for job in jobs:
                if job["seed"] != SEEDS[0] and job["kind"] == "paper_weighted":
                    finished_trios = len(results) // 3
                    projected_trio_seconds = 1.3 * (time.perf_counter() - session_started) / finished_trios
                    if time.time() + projected_trio_seconds >= deadline:
                        save(OUTPUT / "seed_budget_decision.json", {"completed_seeds": SEEDS[:finished_trios],
                            "requested_seeds": SEEDS, "projected_next_trio_seconds": projected_trio_seconds,
                            "remaining_seconds": deadline - time.time(), "basis": "runtime and budget only, no test outcomes"})
                        LOG.warning("Retaining %d complete matched seed trios within the existing budget", finished_trios)
                        break
                m.require(time.time() < deadline, "Comparison GPU deadline reached")
                result_path = OUTPUT / "training" / (job["name"] + ".json")
                if result_path.exists():
                    result = read(result_path)
                    m.require(result["source_hashes"] == hashes and result["corpus_manifest_sha256"] == manifest_hash
                              and result["job"]["seed"] == job["seed"] and result["job"]["kind"] == job["kind"]
                              and len(result["history"]) == 30 and result["predictions"]["large"]["graphs"] == 600,
                              "Cached fit uses a different comparison protocol")
                else:
                    job = dict(job, deadline_epoch=deadline)
                    call = function.spawn(job, hashes, manifest_hash)
                    save(OUTPUT / "calls" / (job["name"] + ".json"), {"call_id": call.object_id, "app_id": app.app_id, "job": job})
                    result = call.get()
                    save(result_path, result)
                for info in result["predictions"].values():
                    path = m.download_artifact(info["path"])
                    m.require(m.sha256(path) == info["sha256"], "Downloaded comparison prediction differs")
                for info in result.get("core_predictions", {}).values():
                    path = m.download_artifact(info["path"])
                    m.require(m.sha256(path) == info["sha256"], "Downloaded full-recipe prediction differs")
                results[job["name"]] = result
                m.write_billing_summary()
                LOG.info("Completed comparison fit %s (%d/9 requested)", job["name"], len(results))
    finally:
        if app.app_id:
            app.stop()
            m.register_paid_app(app.app_id, "matched paper/diffusion training", hourly, stopped=True)
    used = [seed for seed in SEEDS if all(kind + "_s" + str(seed) in results for kind in ["paper_weighted", "ours30", "diffusion"])]
    m.require(bool(used) and len(results) == 3 * len(used), "A matched model trio is incomplete")
    save(OUTPUT / "completed_seeds.json", {"requested": SEEDS, "used": used, "budget_limited": len(used) < len(SEEDS)})
    save(OUTPUT / "training_complete.json", results)
    return results


# Reuse the final model's weights for the same measured comparisons, without another fit.
def prediction_models(fits):
    models = dict(fits)
    for value in fits.values():
        if value["job"]["kind"] == "ours30":
            if set(value.get("core_predictions", {})) != {"test_split", "large"}:
                raise RuntimeError("Full-recipe model inference is incomplete")
            models["core_final_s%d" % value["job"]["seed"]] = {"predictions": value["core_predictions"],
                "checkpoint": value["job"]["core_checkpoint"], "new_training": False}
    return models


# Evaluate all weighted held-out records using one bounded CPU wave and strict ID coverage.
def evaluate_models(m, fits):
    import modal
    if read(OUTPUT / "evaluation_complete.json"):
        return
    budget(m, CPU_CONTAINERS * (.0473 + 2 * .008) * CPU_SECONDS / 3600 + .25)
    hashes = source_hashes(m)
    manifest_hash = m.sha256(m.RESULTS / "gnn/corpus_manifest.json")
    splits = m.read_evidence(m.RESULTS / "gnn/splits.json")
    models = prediction_models(fits)
    save(OUTPUT / "prediction_models.json", models)
    all_jobs = []
    for tag, corpus, ids, size in [("test_split", "full", splits["ids"]["test"], 20),
                                   ("large", "large", list(range(600)), 5)]:
        predictions = {name: value["predictions"][tag] for name, value in models.items()}
        for start in range(0, len(ids), size):
            all_jobs.append({"batch": start // size, "corpus": corpus, "ids": ids[start:start + size],
                             "predictions": predictions, "large_all": read(OUTPUT / "large_all_manifest.json")})
    app = modal.App("weighted-maxcut-paper-diffusion-evaluation")
    hourly = CPU_CONTAINERS * (.0473 + 2 * .008)
    function = app.function(image=image(m, False), cpu=(1, 1), memory=(1024, 2048), timeout=1800,
                            retries=0, max_containers=CPU_CONTAINERS, scaledown_window=2,
                            volumes={m.REMOTE_ROOT + "/results": modal.Volume.from_name(m.VOLUME_NAME)},
                            serialized=True, name="matched_evaluation")(evaluation_worker)
    launch = OUTPUT / "evaluation_session.json"
    m.require(not launch.exists(), "Reconcile unfinished comparison evaluation before resubmission")
    metadata = []
    try:
        with app.run(detach=True):
            m.register_paid_app(app.app_id, "matched paper/diffusion evaluation", hourly,
                                budget_cap_usd=hourly * CPU_SECONDS / 3600 + .25)
            deadline = time.time() + CPU_SECONDS
            jobs = [dict(job, deadline_epoch=deadline) for job in all_jobs]
            calls = []
            for job in jobs:
                call = function.spawn(job, hashes, manifest_hash)
                calls.append({"call_id": call.object_id, "corpus": job["corpus"], "batch": job["batch"]})
                save(OUTPUT / "evaluation_calls.json", calls)
            save(launch, {"app_id": app.app_id, "jobs": jobs, "source_hashes": hashes, "manifest_hash": manifest_hash})
            for info in calls:
                result = modal.FunctionCall.from_id(info["call_id"]).get()
                path = m.download_artifact(result["path"])
                m.require(m.sha256(path) == result["sha256"], "CPU evidence checksum differs")
                metadata.append(result)
                save(OUTPUT / "evaluation_batches.json", metadata)
                LOG.info("Received comparison evaluation %d/%d", len(metadata), len(calls))
    finally:
        if app.app_id:
            app.stop()
            m.register_paid_app(app.app_id, "matched paper/diffusion evaluation", hourly, stopped=True)
    save(OUTPUT / "evaluation_complete.json", {"batches": metadata, "jobs": len(all_jobs),
                                             "completed_epoch": time.time(), "source_hashes": hashes})


# Wait for the existing pipeline; an earlier interruption remains explicit, never relabeled complete.
def wait_for_core():
    last = None
    while True:
        status = read(EXPERIMENT / "results/status.json")
        stage = status.get("stage") if status else "unknown"
        if stage != last:
            LOG.info("Core experiment stage: %s", stage)
            last = stage
        if status and status.get("completed"):
            return
        if status and status.get("error"):
            raise RuntimeError("Core experiment stopped: " + status["error"])
        time.sleep(20)


# Bootstrap graph-level means after averaging seeds within each held-out graph.
def interval(values, seed=42):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError("A summary has missing or nonfinite graph metrics")
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, 1000, 50):
        indices = rng.integers(len(values), size=(50, len(values)))
        estimates.extend(values[indices].mean(1).tolist())
    return {"mean": float(values.mean()), "low95": float(np.quantile(estimates, .025)),
            "high95": float(np.quantile(estimates, .975)), "graphs": len(values)}


# Export compact tables, full per-polish evidence and strict expected-ID checks.
def summarize():
    m = runner()
    splits = read(m.RESULTS / "gnn/splits.json")
    expected = {"full": set(splits["ids"]["test"]),
                "large": set(range(600))}
    fits = read(OUTPUT / "training_complete.json")
    models = prediction_models(fits)
    provenance = read(OUTPUT / "historical_selection_provenance.json")
    fresh_ids = set(provenance["newly_generated_ids"])
    selection_ids = set(provenance["historical_selection_ids"])
    certificates = {}
    with (OUTPUT / "classical_certificates.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value["corpus"] in expected and value["id"] in expected[value["corpus"]]:
                certificates[(value["corpus"], value["id"])] = value
    for corpus, ids in expected.items():
        rows = read_large_all(m, read(OUTPUT / "large_all_manifest.json")) if corpus == "large" else m.read_corpus_rows(corpus, sorted(ids))
        for row in rows:
            certificate = certificates.get((corpus, row["id"]))
            m.require(certificate is not None and certificate["row_sha256"] == hashlib.sha256(
                json.dumps(row, sort_keys=True).encode()).hexdigest(), "Classical certificate belongs to a different graph or label")
    methods = [(name, policy, k) for name in models for policy in ["raw", "snap"] for k in [1, 4]]
    methods += [(name, "raw", k) for name in ["random", "noisy_zero"] for k in [1, 4]]
    methods += [(name, "raw", 1) for name in ["zero", "oracle"]]
    metric_names = ["energy_before", "energy_after", "reference_gap_before", "reference_gap_after",
                    "cut_ratio_before", "cut_ratio_after", "direct_reference_hit", "reference_hit",
                    "classical_bound_hit", "phase_l2_before", "phase_l2_after", "label_hit_before", "label_hit_after",
                    "total_function_queries", "parameter_shift_equivalent_calls", "optimizer_wall_seconds",
                    "optimizer_cpu_seconds", "ranking_wall_seconds", "circuit_setup_wall_seconds",
                    "inference_amortized_seconds",
                    "evaluation_wall_seconds", "optimal_cut_probability_one_shot", "p1_bound_ratio_before",
                    "p1_bound_ratio_after", "p1_bound_hit_before", "p1_bound_hit_after"]
    groups, seen, seed_values = {}, set(), {}
    cases_path, polishes_path = OUTPUT / "per_graph_metrics.csv", OUTPUT / "per_polish_metrics.csv"
    case_fields = ["corpus", "id", "n", "type", "generation_cohort", "historical_selection_exposure", "reference_valid", "continuous_label", "certified_reference", "p1_reference_certification_status",
                   "p1_cut_upper_bound", "model", "policy", "proposals"] + metric_names + [
        "ranking_energy_calls", "joint_simulator_calls", "first_reference_hit_query",
        "first_shared_999_query", "first_self_999_query", "verification_wall_seconds", "sampling_verification_wall_seconds",
        "inference_candidates_generated", "case_execution_order"]
    polish_fields = ["corpus", "id", "model", "policy", "proposals", "proposal_index", "energy_before", "energy_after",
                     "optimizer_reported_energy", "nfev", "njev", "joint_simulator_calls", "optimizer_wall_seconds",
                     "optimizer_cpu_seconds", "reference_hit", "optimizer_success", "status", "message", "budget_exhausted"]
    with cases_path.open("w", newline="", encoding="utf-8") as case_handle, polishes_path.open("w", newline="", encoding="utf-8") as polish_handle:
        case_writer, polish_writer = csv.DictWriter(case_handle, case_fields), csv.DictWriter(polish_handle, polish_fields)
        case_writer.writeheader()
        polish_writer.writeheader()
        for path in sorted((OUTPUT / "evaluations").glob("*/batch_*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    certificate = certificates[(row["corpus"], row["id"])]
                    upper = certificate["p1_cut_upper_bound"]
                    m.require(row["energy_before"] >= -upper - 1e-8 and row["energy_after"] >= -upper - 1e-8,
                              "A model energy violates the independent p=1 upper bound")
                    row.update({"p1_cut_upper_bound": upper, "p1_bound_ratio_before": -row["energy_before"] / upper,
                                "p1_bound_ratio_after": -row["energy_after"] / upper,
                                "p1_bound_hit_before": row["energy_before"] <= -upper + HIT_TOL,
                                "p1_bound_hit_after": row["energy_after"] <= -upper + HIT_TOL,
                                "p1_reference_certification_status": certificate["p1_certification_status"],
                                "certified_reference": certificate["p1_certification_status"] in ["matches_classical_bound", "matches_stronger_locality_bound"]})
                    row["generation_cohort"] = ("larger_test" if row["corpus"] == "large" else
                                               "newly_generated" if row["id"] in fresh_ids else "previously_saved")
                    row["historical_selection_exposure"] = row["corpus"] == "full" and row["id"] in selection_ids
                    if row["model"] in models:
                        elapsed = row.get("inference_amortized_seconds")
                        m.require(type(elapsed) in [float, int] and math.isfinite(elapsed) and elapsed >= 0,
                                  "A learned model is missing measured inference timing")
                        row["inference_amortized_seconds"] = float(elapsed)
                    else:
                        row["inference_amortized_seconds"] = 0.
                    identity = (row["corpus"], row["id"], row["model"], row["policy"], row["proposals"])
                    if identity in seen or row["id"] not in expected[row["corpus"]]:
                        raise RuntimeError("Unexpected or duplicate comparison evaluation record")
                    seen.add(identity)
                    case_writer.writerow({key: row.get(key) for key in case_fields})
                    for polish in row["polishes"]:
                        values = dict(row, **polish)
                        polish_writer.writerow({key: values.get(key) for key in polish_fields})
                    family = row["model"].rsplit("_s", 1)[0] if row["model"] in models else row["model"]
                    key = (row["corpus"], family, row["policy"], row["proposals"])
                    metrics = {name: float(row[name]) for name in metric_names}
                    for subgroup in ["all", "n=" + str(row["n"]), "type=" + row["type"], "continuous_label=" + str(row["continuous_label"]),
                                     "certified_reference=" + str(row["certified_reference"]), "reference_valid=" + str(row["reference_valid"]),
                                     "generation_cohort=" + row["generation_cohort"],
                                     "historical_selection_exposure=" + str(row["historical_selection_exposure"])]:
                        groups.setdefault((*key, subgroup), {}).setdefault(row["id"], []).append(metrics)
                    seed_values.setdefault((row["corpus"], row["model"], row["policy"], row["proposals"]), []).append(metrics)
    expected_count = sum(len(ids) * len(methods) for ids in expected.values())
    if len(seen) != expected_count:
        raise RuntimeError("Comparison coverage is incomplete: %d/%d records" % (len(seen), expected_count))
    for corpus, ids in expected.items():
        for name, policy, k in methods:
            if any((corpus, graph_id, name, policy, k) not in seen for graph_id in ids):
                raise RuntimeError("An expected comparison graph is missing")
    summary, rows_out = {}, []
    for (corpus, family, policy, k, subgroup), records in groups.items():
        required_seeds = sum(name.rsplit("_s", 1)[0] == family for name in models) if family in ["paper_weighted", "ours30", "diffusion", "core_final"] else 1
        if any(len(values) != required_seeds for values in records.values()):
            raise RuntimeError("A graph is missing a required model seed")
        means = {name: [float(np.mean([value[name] for value in records[graph_id]])) for graph_id in sorted(records)]
                 for name in metric_names}
        metrics = {name: interval(values) if subgroup == "all" else {"mean": float(np.mean(values)), "graphs": len(values)}
                   for name, values in means.items()}
        label = "|".join(map(str, [corpus, family, policy, k, subgroup]))
        summary[label] = {"corpus": corpus, "model": family, "policy": policy, "proposals": k,
                          "group": subgroup, "metrics": metrics}
        for name, value in metrics.items():
            rows_out.append({"corpus": corpus, "model": family, "policy": policy, "proposals": k,
                             "group": subgroup, "metric": name, **value})
    with (OUTPUT / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["corpus", "model", "policy", "proposals", "group", "metric", "mean", "low95", "high95", "graphs"])
        writer.writeheader()
        writer.writerows(rows_out)
    spread = {"|".join(map(str, key)): {name: float(np.mean([row[name] for row in records])) for name in metric_names}
              for key, records in seed_values.items()}
    result = {"groups": summary, "individual_seed_means": spread, "expected_case_records": expected_count,
              "received_case_records": len(seen), "coverage_passed": True,
              "completed_seeds": read(OUTPUT / "completed_seeds.json"),
              "bootstrap": "1000 graph resamples after averaging the completed matched model seeds within each graph; subgroup means exploratory"}
    paired = {}
    for corpus in expected:
        for policy in ["raw", "snap"]:
            for k in [1, 4]:
                for left, right in [("ours30", "paper_weighted"), ("diffusion", "paper_weighted"), ("diffusion", "ours30"),
                                    ("core_final", "paper_weighted"), ("core_final", "diffusion"), ("core_final", "ours30")]:
                    a, b = groups[(corpus, left, policy, k, "all")], groups[(corpus, right, policy, k, "all")]
                    ids = sorted(expected[corpus])
                    differences = {}
                    for metric in ["cut_ratio_before", "reference_hit", "total_function_queries", "optimizer_wall_seconds", "inference_amortized_seconds"]:
                        values = [np.mean([v[metric] for v in a[i]]) - np.mean([v[metric] for v in b[i]]) for i in ids]
                        differences[metric] = interval(values)
                    paired["|".join(map(str, [corpus, left + "_minus_" + right, policy, k]))] = differences
    result["paired_differences"] = paired
    save(OUTPUT / "summary.json", result)
    return result


# Draw publication-friendly comparisons without substituting estimated circuit work for time.
def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["random", "paper_weighted", "ours30", "diffusion", "core_final"]
    labels = ["Random", "Paper GINE", "Our GNN (30 epochs)", "Graph diffusion", "Our GNN (full recipe)"]
    for corpus in ["full", "large"]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for axis, metric, title in zip(axes, ["cut_ratio_before", "reference_hit", "optimizer_wall_seconds"],
                                       ["Direct expected cut / exact maximum", "Reference energy hit after polish", "CPU optimizer seconds / graph"]):
            values = [summary["groups"]["|".join(map(str, [corpus, model, "raw", 1, "all"]))]["metrics"][metric] for model in models]
            y = [value["mean"] for value in values]
            axis.bar(range(len(y)), y, color=["#777777", "#d98b2b", "#3679b9", "#5a9d68", "#8756a5"])
            axis.errorbar(range(len(y)), y, yerr=[[v - a["low95"] for v, a in zip(y, values)],
                                                [a["high95"] - v for v, a in zip(y, values)]], fmt="none", color="black", capsize=3)
            axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
            axis.set_title(title, fontsize=10)
            axis.grid(axis="y", alpha=.2)
        fig.suptitle("Weighted MA-QAOA p=1; one raw proposal; " + ("held-out training-size graphs" if corpus == "full" else "unseen n=13–14 graphs"))
        for extension in ["png", "pdf"]:
            fig.savefig(OUTPUT / ("comparison_" + corpus + "." + extension), dpi=180)
        plt.close(fig)


# Write an honest research account and an unsent email from completed numerical evidence.
def report():
    completed = read(OUTPUT / "evaluation_complete.json")
    summary = summarize() if completed else None
    if summary:
        figures(summary)
    protocol = read(OUTPUT / "protocol.json") or {}
    core = read(EXPERIMENT / "results/status.json") or {}
    original = read(ROOT / "results/gnn/final_results.json")
    classical = read(OUTPUT / "classical_verification_summary.json") or {}
    fits = read(OUTPUT / "training_complete.json") or {}
    population = read(EXPERIMENT / "results/preflight/training_population.json") or {}
    final = read(EXPERIMENT / "results/gnn/final_results.json")
    curve = read(EXPERIMENT / "results/gnn/learning_curve.json")
    billing = read(EXPERIMENT / "results/gnn/billing_summary.json") or {}
    lines = ["Weighted MA-QAOA parameter prediction: technical account", "=" * 65, "",
             "Scope: weighted graphs only, p=1. One filter-selected reference vector per graph. Learn graph -> parameters; evaluate direct predictions and equally budgeted local refinement on held-out weighted graphs.", "",
             "Current core status: " + core.get("stage", "unknown") + ". Comparison complete: " + str(bool(summary)) + ".", "",
             "1. Dataset, labels and leakage controls", "",
             "The generator samples ten families: ER p=.3/.4/.5/.6, 3/4-regular, trees, bipartite, Barabasi-Albert m=2 and Watts-Strogatz k=4,p=.3. It permutes vertex labels and draws weights in [.5,2], rounded to four decimals. Random weights usually break graph automorphisms; this does not remove every circuit symmetry or prove uniqueness of a minimum.",
             "The unchanged filter harvests minima, applies its ordered selection, settles/slides flat directions, and confirms the pick on two seeds. The production stopping rule uses 100 unchanged restarts and a 5% unrelated-seed audit. The supplied 21-reference timing study reports agreement between STALL=100 and 400 and on unrelated seeds; the Modal n=6-9 regression rerun covers 12 references. All 320 production-pilot audits passed. These are empirical reproducibility assurances.",
             "CSV gamma/weight order is NetworkX node-insertion edge order, whereas filter internals use sorted edges. Explicit conversion was verified on Ian's 20 rows to about 9e-15; confusing these orders silently evaluates a different circuit.",
             "The original interrupted run saved 63,672 records, of which 63,288 passed (99.3969%). The 36,328 missing IDs were not run, rather than failed. The continuation fills only those IDs and preserves every original saved record byte-for-byte. Rejected records never become training examples merely to improve coverage.",
             "The original 105 Max-Cut plus 69 spin-glass landscape instances must not be described as 174 Max-Cut graphs. Relabel 19/20 and 2% weight 16/20 were pooled over 10 of each problem class. The 21 reference graphs are all weighted Max-Cut.",
             "Train/validation/test IDs are grouped by the core WL hash protocol; extrapolation graphs are excluded from training. All comparison models reuse identical physical labels and split IDs. Model-native validation loss selects 30-epoch checkpoints; held-out test energies do not select them. Core full-recipe training separately uses validation warm-start performance. Check splits.json for actual group memberships and caveats about WL equivalence.", "",
             "2. Physical conventions and what is certified", "",
             "The implemented objective is E=1/2 sum_e w_e(<Z_u Z_v>-1)=-expected cut. With theta_e=w_e gamma_e, the circuit uses exp(+i sum theta_e Z_u Z_v) then exp(+i sum beta_v X_v). Every theta and beta has energy period pi. The predictor's common representation is [theta edges in sorted order, beta nodes], with raw gamma=theta/w used by the pinned simulator.",
             "Exact integer enumeration fixes one vertex to quotient the complementary cut symmetry and uses integer weights at scale 10,000. It verifies the classical maximum C* independently; E >= -C*. Independent MILP spot checks cross-check the enumerator. Matching -C* certifies p=1 variational optimal energy within tolerance, but never unique angles. A positive gap alone cannot distinguish ansatz restrictions from an optimizer missing a lower p=1 minimum.",
             "A stronger p=1 bound uses disjoint radius-one lightcones. For vertices u,v at distance at least three, <ZuZv>=0, so same/opposite partition relations each occur with probability 1/2. Exact conditional classical maxima C_same,C_opposite give expected cut <= U=(C_same+C_opposite)/2. We minimize U over eligible pairs. A feasible label matching U certifies its p=1 optimal energy within 1e-6 even when U<C*. Ratios -E/U give a lower bound on the achieved fraction of the unknown p=1 optimum. A point attaining U certifies its own global p=1 energy; failing to attain a possibly loose bound does not establish failed optimization. These are applications of locality, not a novelty claim.",
             "For graphs with a dominating vertex, activate just the spanning-star cost gates and set leaf mixers according to an exact optimal cut. This constructs a feasible p=1 state at -C*. We independently evaluate the resulting angles. When a saved label is worse, this is a concrete counterexample to its global optimality, rather than just an unresolved gap. The frozen selected-reference labels remain unchanged for the matched benchmark; known weaker labels are flagged.",
             "A four-node path with distinct edge weights (1.2,.7,1.6) has C*=3.5 but exact p=1 MA-QAOA maximum expected cut 3.15. Endpoint backward lightcones are disjoint, so <Z0 Z3>=0. Endpoints agree half the time, allowing cut weight at most 3.5-.7=2.8 then; otherwise the bound is 3.5. Their average is 3.15. The angles gamma=(pi/(4*1.2),0,pi/(4*1.6)), beta=(0,-pi/4,0,-pi/4) attain 3.15. These unequal positive weights also remove path reversal symmetry. This is a weighted analytic illustration, not a new training-label generation project. It proves that p=1 energy need not equal the classical bound even with perfect optimization.",
             "A lower feasible energy than a saved reference disproves that reference's global optimality. It does not invalidate reproducible selection as a supervised task. Preserve original targets for paired evaluation and report such counterexamples separately.", "",
             "3. Models and the two papers", "",
             "Paper-based GNN: five residual GINE layers, hidden 192, BatchNorm/ReLU, dropout .1; circular gate-angle prediction with 1-cos(error)+.02*abs(norm-1). The weighted extension predicts actual RZZ rotations r=-2*w*gamma_filter and paper mixer b=-beta_filter; RZZ(r)=exp(-i*r*ZZ/2), RX(2*b)=exp(-i*b*X). Decoding theta_filter=-r/2 and beta_filter=-b preserves the physical state. Weight is absorbed into each independent edge rotation. Original experiments were unweighted and original code is unavailable. Weighted features, angle units, ordering/eigenspace conventions and the common split are documented adaptations in protocol.json. This is not a claim of bit-for-bit reproduction.",
             "Core global models use six hidden-256 GPS layers, each combining local GINE aggregation with eight-head full-graph attention, residual LayerNorm blocks and a GELU feed-forward block. Their node input has 46 channels: six degree/weighted-degree/size/triangle/clustering scalars, 32 unweighted/weighted random-walk return probabilities for steps 1-16, and eight weighted-Laplacian coordinates. Edge input has 35 channels: 11 weight/rank/endpoint/common-neighbor scalars and 24 symmetric Laplacian endpoint combinations. The scalar wdegree/mean_incident is algebraically the unweighted degree; it is retained as implemented, not a distinct extra statistic. Graph-invariant anchors orient simple and repeated eigenspaces, with unresolved directions zeroed; coherent eigenvector-sign augmentation is applied during training. The paper baseline instead has ten node and 19 directed-edge channels, including unweighted-Laplacian coordinates and edge weights. Padding masks exclude artificial nodes and edges.",
             "At the user's request, architecture selection is repeated across all seven models on the expanded training split and the fixed 2000 validation graphs. Three full D fits and part of its learning curve had completed before that request arrived. They are preserved; reuse requires matching the new winning configuration and post-processing strategy. See bakeoff.json and its preserved prior-selection evidence for the measured winner and chronology. The controlled comparison uses a common 30-epoch constant-AdamW supervised recipe; the full core model has a different validation/epoch recipe and is reported separately. Epoch matching is not FLOP, GPU-time or parameter-count matching.",
             "For core regression, each circular phase contributes 1-cos(2*(prediction-target)) plus the .02 raw-vector norm penalty. Classification uses cross-entropy on edge and mixer classes; training-only inverse-square-root class weights activate when the largest/smallest nonzero class count exceeds three. Losses average active edges and nodes separately, sum the two blocks, and take the best hypothesis per graph before the minibatch mean. The full core fits use AdamW and, for global models, warmup for up to five epochs followed by cosine decay. They train at most 150 epochs, validate warm-start work every five epochs, and stop after four checks without improvement. Seeds 42/43/44 are retained separately. Classification rounds to discrete class representatives; rare continuous targets can therefore remain imperfect even with otherwise correct classifications.",
             "The earlier usable dataset had 0.8387% of edge phases outside the {0,pi/4} classes and 0.01423% of beta labels outside the pi/8 grid within the chosen tolerance. This motivated classification, while leaving rare continuous-label cases to examine separately. The core vocabulary trigger reads a full-population summary, although class centroids/weights use training labels. The controlled comparison recomputes the entire vocabulary trigger and centroids from training rows only, and records whether it matches the core configuration.",
             "The diffusion source studied standard QAOA p=3 with six shared angles and unweighted graphs. Its graph-conditioned denoising approach is adapted here to a variable number of MA-QAOA edge and node parameters. Our denoiser is a four-layer hidden-128 GPS network, not their fixed-vector MLP/Graph2Vec. It consumes weighted graph features, noisy circular coordinates and timestep embeddings, predicts Gaussian noise at 1000 levels, and generates four proposals using 50 DDIM steps. A GNN and diffusion are compatible: the GNN is the denoising network.",
             "Concretely, training embeds the selected angles as circular coordinate pairs, chooses a noise level t, adds Gaussian noise, and asks the graph-conditioned network to recover the added noise. At inference it starts from random coordinates and repeatedly applies the learned denoiser before decoding angles with atan2. This provides a stochastic graph-to-parameters predictor. Four initial noise draws give four proposals without searching for four extra reference minima. DDIM uses fewer sampling steps than the 1000-level training schedule; the saved protocol fixes that choice before evaluation.",
             "Each graph still has exactly one clean filter label. Added noise is training augmentation, not additional independent minima or new graph labels. A diffusion model may capture uncertainty and offer diverse guesses, but superiority over direct regression must be measured; it is not assumed.", "",
             "4. Evaluation definitions and fairness", "",
             "Primary: raw head/sample 0, one proposal per method; report direct energy, phase label error and reference hit before refinement. Then use the unchanged L-BFGS-B polish and record final physical energy, reference hit, cut ratio, objective/gradient calls and wall/process CPU time. Secondary: four proposals, charge every energy ranking call, share a 2000-query budget across polishes and stop on a reference-energy hit. For the deterministic paper predictor, K=4 means one prediction plus three fixed sigma=.05 phase perturbations, explicitly a jitter control.",
             "The saved reference floor supplies a benchmark stopping target. It is not available for an arbitrary new deployment graph. Query counts until that target measure known-target benchmark work; deployment would need a fixed budget or a separately justified stopping certificate. These comparisons also change features, representations and losses, so any performance difference concerns the complete predictor configuration rather than an isolated causal effect of architecture.",
             "Raw and nearest-pi/8 snapped policies are both retained; the test set never chooses a winning policy. Random, exact-zero, noisy-zero and oracle-label controls use the same optimizer and budget. Ranking costs and post-processing choices can materially affect comparisons. A four-proposal model against one random proposal is not the primary fair comparison.",
             "Each joint adjoint energy/gradient evaluation is one simulator query here. A hardware parameter-shift-equivalent estimate charges 2D+1 circuits per joint call plus energy ranking calls; it is an estimate, not circuits actually executed. The core expected restart-work metric includes an estimated unexecuted random fallback. The new comparison reports executed calls and measured time directly.",
             "Energy hit means E <= reference+1e-6; label hit means period-aware phase L2 distance <=1e-3. They are different. Exact-classical-bound attainment and ideal one-shot optimal-cut sampling probability are also distinct; the latter is computed from the final statevector. First-query-to-shared-99.9% and first-query-to-own-final-99.9% are saved separately because an easy self-relative target can reward a worse final solution.",
             "Independent verification times are separate from optimization/inference times. GPU inference timings describe the complete saved prediction artifact (four candidates for D/diffusion and one for the paper GNN); K=1 evaluation takes its first candidate, so those GPU times are not a separately optimized single-sample latency benchmark. Each graph's case order is deterministically randomized for paired CPU timing; hardware metadata and execution order are retained. This evaluates the original paper's local-optimization arm under a common optimizer; its complete MLS/dual-annealing study is outside this predictor comparison. Older experiments did not record per-polish runtime, so their missing values remain missing. An earlier review corrected optimizer-reported energy metadata discrepancies using independently evaluated final angles, without altering those angles or evaluation counts; the underlying cause was not established.",
             "Every expected graph/model/seed/policy must be present. Missing/pickless records never disappear from a percentage denominator. Completed matched model seeds are averaged within graph before graph bootstrap intervals; seed means and graph uncertainty are reported separately. These intervals describe held-out graph variation conditional on the completed fits and fixed split; they do not estimate uncertainty over new training datasets or all possible training seeds. The protocol requests three seeds, but adds whole seed trios only when measured runtime projects that they fit within the budget; the actual count is recorded. Runtime, rather than test outcomes, determines any reduction. Node/family subgroup means are exploratory.", ""]
    generation = read(OUTPUT / "generation_statistics.json")
    if generation:
        data_lines = ["Completed graph population and recorded generation time", "",
                      "Corpus | Nodes | Saved | Usable | Audits | Audit disagreements | Mean main seconds"]
        for row in sorted((row for row in generation["groups"] if row["group"].startswith("n=")
                           and ";" not in row["group"]), key=lambda row: (row["corpus"], int(row["group"][2:]))):
            data_lines.append("%s | %s | %d | %d | %d | %d | %.2f" % (row["corpus"], row["group"][2:],
                row["saved"], row["usable"], row["audited"], row["audit_disagreements"], row["secs_main_mean"]))
        data_lines.extend(["", "All planned IDs are saved. Usability is measured over saved records, including every rejected one in the denominator. Three-regular graphs require an even node count; families are sampled only where the generator permits them. generation_statistics.csv includes each family and n-by-family count and timing distribution. These times come from successful saved records and do not replace billed resource-hours.", ""])
        split = read(EXPERIMENT / "results/gnn/splits.json")
        if split:
            data_lines.extend(["The frozen split contains %d training, %d validation and %d held-out test graphs. A fixed %d-graph subset of validation supplies the optimizer-based architecture and early-stopping comparisons. The paper/diffusion fits select checkpoints using native loss on the same validation split. The 25/50/100%% learning curve keeps that validation/test split fixed and uses nested subsets of the training IDs." %
                tuple(len(split["ids"][name]) for name in ["train", "val", "test", "val_selection"]), ""])
            data_lines.extend(["Training-size test performance is conditional on the frozen usable-label population. Every expected test ID remains in its model-performance denominator. The separate larger-graph evaluation covers all 600 planned IDs, including unusable references; its denominator therefore answers a broader coverage question.", ""])
        stats = read(EXPERIMENT / "results/gnn/label_statistics.json")
        if stats:
            data_lines.extend(["Expanded usable-label statistics: %d edge phases and %d mixer coordinates. %.6f%% of edge phases lie outside {0,pi/4}; %.6f%% of mixers lie outside the pi/8 grid at the prescribed tolerance. The correct box test uses the filter's raw-gamma tolerance, scaled by each weight in phase coordinates. It reports %d violations. A fixed phase tolerance would instead flag %d small excesses, so it is not interchangeable with the specified test." %
                (stats["edge_count"], stats["node_count"], 100 * stats["theta_other_fraction"],
                 100 * stats["beta_other_fraction"], stats["box_violations"], stats["fixed_phase_tolerance_violations"]), ""])
        bias = read(ROOT / "results/preflight/saved_data_bias_check.json")
        if bias:
            data_lines.extend(["Earlier interrupted-run bias check, computed from saved records without Modal compute", "",
                               "Nodes | Completed mean main seconds | Random pilot mean seconds | Ratio | Bootstrap 95% ratio interval"])
            for n, value in sorted(bias["by_n"].items(), key=lambda pair: int(pair[0])):
                data_lines.append("%s | %.3f | %.3f | %.4f | [%.4f, %.4f]" % (n, value["saved_mean_secs_main"],
                    value["pilot_mean_secs_main"], value["mean_ratio"], *value["bootstrap_mean_ratio_95ci"]))
            shards = bias["shards"]
            data_lines.extend(["", "The original 36328 missing IDs comprised %d in %d wholly unstarted shards and %d in %d partially completed shard tails. Those partial shards contained ascending-ID prefixes; the data do not identify a globally runtime-sorted slow tail. The partial sample overrepresented small n, but no within-n interval met the recorded clearly-easy rule (%s). With only 40 pilot graphs per n, this is a coarse timing check, not proof of label/topology representativeness. bias_independent_mean_check.json independently reproduces the completed means. The resumed experiment now has all planned IDs, while rejected-label selection remains visible in its usability table." %
                (shards["missing_in_whole_shards"], shards["unstarted"], shards["missing_in_started_shard_tails"],
                 shards["partial"], bias["practical_signal_rule"]), ""])
        index = lines.index("2. Physical conventions and what is certified")
        lines[index:index] = data_lines
    numeric_lines = ["Independent numerical regression gates", ""]
    rebuilt = read(EXPERIMENT / "results/preflight/reference_build.json")
    reruns = read(EXPERIMENT / "results/preflight/reference_results.json")
    csv_gate = read(EXPERIMENT / "results/preflight/full_corpus_csv_gate.json")
    ian_gate = read(EXPERIMENT / "results/preflight/ian_floor_gate.json")
    gpu_gate = read(EXPERIMENT / "results/gnn/torch_energy_check_cuda.json")
    if rebuilt and reruns:
        numeric_lines.append("Reference graph rebuild: %d/%d pass. Modal filter regressions: %d/%d pass, maximum pick-vector distance %.3g." %
            (sum(v["passed"] for v in rebuilt["results"]), rebuilt["expected"],
             sum(v["passed"] for v in reruns), len(reruns), max(v["pick_distance"] for v in reruns)))
    if csv_gate:
        numeric_lines.append("All-saved CSV edge-order/energy check: %d rows, maximum independent energy error %.3g, threshold %.3g." %
            (csv_gate["rows"], csv_gate["max_energy_error"], csv_gate["threshold"]))
    if ian_gate:
        numeric_lines.append("Legacy Ian regression gate passed=%s: %d rows rebuilt, %d known floor targets, maximum CSV energy error %.3g." %
            (ian_gate["passed"], ian_gate["expected"], len(ian_gate["known_floor_rows"]), ian_gate["max_paper_energy_error"]))
    if gpu_gate:
        numeric_lines.append("Independent GPU circuit gate passed=%s: %d random parameter points, maximum energy error %.3g and maximum phase-gradient coordinate error %.3g, complex128 on %s. This A100 allocation was a separately capped numerical check; model-training hardware is recorded per fit." %
            (gpu_gate["passed"], gpu_gate["points"], gpu_gate["max_energy_error"], gpu_gate["max_gradient_error"], gpu_gate["gpu"]))
    if len(numeric_lines) > 2:
        numeric_lines.append("")
        index = lines.index("3. Models and the two papers")
        lines[index:index] = numeric_lines
    lines.extend(["The new matched comparison evaluates all 600 planned larger graphs. Sixteen did not pass reference-label usability checks, so their reference-hit and label-hit outcomes are failures; their predicted physical energies and ideal sampling statistics are still evaluated. The oracle reference-hit ceiling is therefore 584/600. Conditional reference_valid=True tables allow comparisons on the 584 valid-label graphs. Earlier core-only reports retain their original conditional 584-graph metrics; they must not be read as full-600 success percentages.",
                  "The core_final rows reuse the saved full-recipe GNN checkpoints and evaluate them under exactly the same cases as the 30-epoch models. Their training-ID hashes are checked against the shared training split. No extra full-recipe training is performed by the comparison.", ""])
    bakeoff = read(EXPERIMENT / "results/gnn/bakeoff.json")
    if bakeoff and bakeoff.get("selection_provenance", {}).get("reused_prior_selection") is False:
        lines.extend(["Expanded seven-model validation comparison", "",
                      "All variants use the expanded training split, seed 42, a 40-epoch budget and the same 2000 validation-selection graphs. C03/C10 reuse 20 supervised epochs of the selected B/Bcls base, then apply 20 energy fine-tuning epochs. A uses five hidden-192 GINE layers; B/Bcls/C/D/E use six hidden-256 GPS layers. This A baseline uses the core phase representation/loss, so it is distinct from the documented raw-angle paper reproduction in the matched comparison.",
                      "A/B regress circular phases; Bcls classifies phases; E uses energy-only training; C03/C10 combine label loss with energy weights .3/1.0; D has four classification or regression heads, inheriting the validation-selected B/Bcls base. D minimizes the best head's per-graph label loss. Its heads are alternative predictions learned from one target, not four independent reference minima. The raw K=1 comparison always takes head 0; the full core recipe ranks four heads by initial energy and charges those queries.", "",
                      "Variant | Selected policy | Geometric mean estimated work/random | Summed estimated work/random | Reference hit"])
        for name in ["A", "B", "Bcls", "E", "C03", "C10", "D"]:
            evaluation = read(EXPERIMENT / ("results/gnn/evaluations/" + name + "_s42_val.json"))
            if not evaluation or len(evaluation["expected_ids"]) != 2000:
                raise RuntimeError("A completed expanded bake-off lacks its full validation evidence")
            strategy = bakeoff["strategies"][name]["chosen"]
            metrics = evaluation["summaries"][strategy]["metrics"]
            ratio = math.fsum(v["expected_cost"] for v in evaluation["results"][strategy]) / math.fsum(
                v["expected_cost"] for v in evaluation["results"]["random"])
            lines.append("%s | %s | %.6f | %.6f | %.4f" % (name, strategy,
                metrics["geometric_mean_cost_ratio"]["value"], ratio, metrics["hit_rate"]["value"]))
        lines.extend(["", "Selected expanded winner: %s, policy %s. Selection follows paired validation log-work comparisons with the prespecified simplicity rule. These are validation results, not held-out performance. bakeoff.json preserves the comparisons, uncertainty rule and exact winning configuration." %
                      (bakeoff["selection"]["chosen"], bakeoff["strategy"]), ""])
        reused = read(EXPERIMENT / "results/preflight/expanded_winner_reuse_check.json")
        if reused:
            lines.extend(["Reuse of the completed full fits passed exact checks for the winning configuration, selected policy, all three seeds, training-ID hash and scientific source hash. expanded_winner_reuse_check.json records the evidence; this reuse does not represent three additional paid fits.", ""])
    provenance = read(OUTPUT / "historical_selection_provenance.json")
    if provenance:
        lines.extend(["The expanded architecture choice uses the fixed expanded validation set. Historical model development also used the earlier corpus: of the expanded test graphs, %d were newly generated, %d were previously saved, and %d appeared in the earlier validation-selection list. All expanded test graphs are excluded from these new fits, but the entire expanded test set cannot be described as untouched by historical model development. The generation_cohort=newly_generated tables provide a separate cohort unavailable during that earlier selection. These are descriptive subgroup results; no new hyperparameters are selected from them." %
                      (provenance["test_newly_generated"], provenance["test_previously_saved"], len(provenance["test_historical_selection_overlap"])), ""])
    if classical:
        p1 = classical.get("p1_counts_saved_usable", {})
        usable = sum(value for key, value in p1.items() if key.startswith("full:"))
        certified = p1.get("full:matches_classical_bound", 0) + p1.get("full:matches_stronger_locality_bound", 0)
        if usable:
            lines.extend(["Usable weighted training-size labels with certified p=1 optimal energy: %d/%d (%.4f%%). Other labels remain uncertified unless a constructive counterexample is listed." %
                          (certified, usable, 100 * certified / usable), ""])
        lines.extend(["Classical and p=1 certificate counts among usable labels", "",
                      "Corpus | Classical-bound match | Stronger locality-bound match | Uncertified gap | Explicit better witness"])
        for corpus in ["full", "large"]:
            lines.append("%s | %d | %d | %d | %d" % (corpus, *[p1.get(corpus + ":" + key, 0) for key in [
                "matches_classical_bound", "matches_stronger_locality_bound", "uncertified_gap", "explicit_better_star_witness"]]))
        lines.extend(["", "Verified counterexamples to global optimality of a frozen usable reference", "",
                      "ID | Nodes | Split | Saved reference energy | Certified p=1 optimum | Dominating center"])
        fixed_split = read(EXPERIMENT / "results/gnn/splits.json")["ids"]
        for value in classical.get("known_suboptimal_star_findings", []):
            if value["corpus"] == "full" and value["saved_usable"]:
                membership = next(name for name in ["train", "val", "test"] if value["id"] in fixed_split[name])
                lines.append("%d | %d | %s | %.8f | %.8f | %d" % (value["id"], value["n"], membership, value["reference_energy"],
                    value["certified_p1_optimal_energy"], value["center"]))
        lines.extend(["", "The verification manifest covers %d saved records: the 100000 generated training-size graphs, all 600 larger graphs and 20 legacy Ian CSV checks. Those legacy checks are auxiliary and do not enter weighted model training. The 21 supplied weighted reference graphs have a separate certificate file. The usable-label table above excludes rejected labels; all-saved counts by n and family remain in classical_verification_summary.json." % classical["records"],
                      "Independent integer-program checks passed on %d records (%s by node count). classical_certificates.jsonl retains each graph's exact optimal cut witnesses, degeneracy, conditional maxima, p=1 bound and any constructive star parameters. Its SHA-256 is %s. A certificate of energy optimality does not certify a unique parameter vector." %
                      (sum(classical["independent_milp_checks_by_n"].values()),
                       json.dumps(classical["independent_milp_checks_by_n"], sort_keys=True), classical["raw_sha256"]), ""])
    if summary:
        lines.extend(["5. Measured matched comparison", "",
                      "Requested and completed seed counts: " + json.dumps(summary["completed_seeds"]) + ".", "",
                      "Set | Model | Direct cut ratio | Direct reference hit | After-polish hit | Function queries | CPU polish seconds | Amortized GPU inference seconds",
                      "----|-------|------------------|----------------------|------------------|------------------|-------------------|-------------------------------"])
        for corpus in ["full", "large"]:
            for model in ["random", "zero", "noisy_zero", "oracle", "paper_weighted", "ours30", "diffusion", "core_final"]:
                row = summary["groups"]["|".join(map(str, [corpus, model, "raw", 1, "all"]))]["metrics"]
                values = [row[key]["mean"] for key in ["cut_ratio_before", "direct_reference_hit", "reference_hit", "total_function_queries", "optimizer_wall_seconds", "inference_amortized_seconds"]]
                lines.append("%s | %s | %.6f | %.4f | %.4f | %.2f | %.6f | %.6f" % (corpus, model, *values))
        lines.extend(["", "Full raw/snap K=1/K=4, 95% intervals, each seed, each node count and each family are in summary.json and summary_metrics.csv. The primary table above does not select the best method using held-out outcomes.", ""])
        lines.extend(["Agreement with the single filter-selected parameter vector, raw K=1", "",
                      "Set | Model | Direct label hit | Polished label hit | Direct phase L2 | Polished phase L2",
                      "A label hit requires period-aware L2 <= 1e-3 across the complete [w*gamma,beta] vector. Only individual pi periods are quotiented; further circuit symmetries and flat-valley equivalences are not quotiented. A prediction can achieve the same energy while failing this stricter coordinate test. Larger graphs have more coordinates, so their unnormalized L2 is not directly size comparable."])
        for corpus in ["full", "large"]:
            for model in ["paper_weighted", "ours30", "diffusion", "core_final"]:
                row = summary["groups"]["|".join(map(str, [corpus, model, "raw", 1, "all"]))]["metrics"]
                lines.append("%s | %s | %.4f | %.4f | %.6f | %.6f" % (corpus, model,
                    *[row[key]["mean"] for key in ["label_hit_before", "label_hit_after", "phase_l2_before", "phase_l2_after"]]))
        lines.append("")
        lines.extend(["Fresh-generation test cohort, raw K=1 (held out from current fits and unavailable during the earlier architecture selection)",
                      "Model | Graphs | Direct cut ratio | After-polish reference hit | Function queries"])
        for model in ["random", "paper_weighted", "ours30", "diffusion", "core_final"]:
            key = "|".join(map(str, ["full", model, "raw", 1, "generation_cohort=newly_generated"]))
            row = summary["groups"][key]["metrics"]
            lines.append("%s | %d | %.6f | %.4f | %.2f" % (model, row["cut_ratio_before"]["graphs"],
                row["cut_ratio_before"]["mean"], row["reference_hit"]["mean"], row["total_function_queries"]["mean"]))
        lines.append("")
    else:
        lines.extend(["5. Matched comparison pending", "", "No paper/diffusion results are asserted before the recorded training and evaluation finish. See status.json for the live stage.", ""])
    if fits:
        lines.extend(["6. Measured model resources", "", "Fit | Parameters | Feature seconds | Training seconds | Own training plus prediction seconds | Additional full-recipe inference seconds | Best epoch | GPU"])
        for name, value in fits.items():
            lines.append("%s | %d | %.2f | %.2f | %.2f | %.2f | %d | %s" % (name, value["parameters"], value["feature_seconds"], value["training_seconds"], value["training_and_prediction_seconds"], value.get("additional_core_prediction_seconds", 0.), value["best_epoch"], value["gpu"]))
    if final:
        lines.extend(["", "Completed expanded-core full-recipe results", "",
                      "Core usable population: %s. Selected architecture: %s. Selected post-processing: %s." %
                      (population.get("usable", "see manifest"), final.get("chosen", "see bakeoff.json"), final["strategy"]),
                      "Set | Sum of estimated work / random | Geometric mean work ratio | Reference hit"])
        for tag in ["test_split", "large"]:
            values = final["sets"][tag]
            selected = values[final["strategy"]]
            metrics = selected["summary"]["metrics"]
            ratio = math.fsum(r["expected_cost"] for r in selected["records"]) / math.fsum(r["expected_cost"] for r in values["random"]["records"])
            hit = metrics["hit_rate"]["value"] * (len(selected["records"]) / 600 if tag == "large" else 1.)
            lines.append("%s | %.6f | %.6f | %.4f" % (tag, ratio, metrics["geometric_mean_cost_ratio"]["value"], hit))
        lines.extend(["For the larger core set, work ratios remain conditional on valid references; the hit rate above counts the 16 unusable planned references as failures over all 600 IDs.", ""])
    if curve:
        lines.extend(["", "Expanded-core learning curve (common epoch cap; more graphs imply more updates)",
                      "This curve uses the core validation-selected %s policy with %d proposal heads; the matched comparison's primary table above uses one raw proposal." %
                      (curve["strategy"], bakeoff["configuration"].get("hypotheses", 1)),
                      "Fraction | Training graphs | Geometric mean work/random | Reference hit"])
        for percent in [25, 50, 100]:
            value = curve["fractions"][str(percent)]
            metrics = value["summary"]["metrics"]
            lines.append("%d%% | %d | %.6f | %.4f" % (percent, value["graphs"], metrics["geometric_mean_cost_ratio"]["value"], metrics["hit_rate"]["value"]))
    if billing:
        lines.extend(["", "Cumulative billing as of %s: posted $%.8f; upper reservation $%.8f; stop $280." %
                      (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(billing["checked_epoch"])),
                       billing["reported_usd"], billing["upper_reservation_usd"])])
    lines.extend(["", "7. Earlier completed experiment, kept separate", "",
                 "Original winner: four-proposal classifier D. On the original held-out size range, summed estimated warm-start/fallback work divided by summed random work was .581166; for n=13-14 it was .855401. These are empirical-target work estimates, not runtime speedups. Original nested 25/50/100% aggregate ratios were .659523/.635649/.581166, with hits 70.83/73.06/76.34%. Larger subsets also had more optimizer updates under a common epoch cap. The full continuation repeats the study on the completed population; do not mix the two populations' test IDs.", "",
                 "8. Execution history and engineering findings", "",
                 "Data generation uses the pinned NumPy/SciPy statevector and L-BFGS-B implementation on CPUs. GPU training does not accelerate that unchanged filter. An early pilot oversubscribed numerical threads after Modal replaced image environment settings. Its timings were excluded, its records retained, and its charges included. Parent and child process thread limits and loaded numerical libraries were subsequently checked. Corrected-pilot timeouts were resumed with unchanged graph parameters; a failed audited gradient check remained rejected.",
                 "The original interrupted attempt returned zero of 80 expected batch completions while saving 63672 graph records. A TLS handshake timeout appeared in the client output task. The suspected interaction with detached live-log streaming is an inference from that log and SDK behavior; the full causal chain was not established. The repaired runner removes live streaming, persists calls, reattaches recorded calls, and requires every assigned graph and batch completion. A paid 20-graph smoke check preceded completing the larger test set. Full-data continuation uses an isolated namespace and retains all earlier task costs, including failed transfers and retries.",
                 "Incomplete call manifests are reconciled before resubmission. The final unsaved generation tail was recovered with bounded one-CPU jobs. Every new paid allocation is profile-locked to tejas-16842 and checks cumulative billing. Training uses at most one allowed GPU worker at a time; its idle time and allocation overhead are included in reservations. The late explicit request to repeat the seven-model bake-off was applied at a completed-job boundary, preserving existing D work and the exact scientific data gate. expanded_bakeoff_user_request.json and expanded_winner_reuse_check.json record that chronology when available.", "",
                 "9. Artifacts and provenance", "",
                 "protocol.json fixes scientific choices. training/*.json records histories, checkpoint selection, software/GPU and source hashes. predictions/*.json contains all candidate angles and synchronized batch inference timings. evaluations/*/*.jsonl contains every candidate polish, its initial/final angles, objective trajectory and optimizer status. per_graph_metrics.csv and per_polish_metrics.csv are convenient flat tables. summary_metrics.csv, summary.json and figures report comparisons. Classical certificates and earlier raw exports are independent local analyses, explicitly labeled.",
             "generation_statistics.csv and generation_statistics.json report saved/usable/audited counts, filter-reported nfev/restarts, and generation-time distributions by node count and family. Summed saved-record elapsed times exclude interrupted work and are not billed core-hours; consult the ledger for actual task charges.",
                 "The core results/report.md, gnn/final_results.json, gnn/learning_curve.json, preflight gates and cost_ledger.md remain separate authoritative evidence for the full experiment. The original 63,288-usable experiment and its archive are preserved. Budget stops or interrupted comparisons are reported as incomplete, never as null/negative model findings.", "",
                 "Sources: user-supplied NeuralNetworkMAQAOA.pdf; https://link.springer.com/article/10.1140/epjqt/s40507-025-00397-4; pinned local filter/checker/builder; protocol and source hashes.", ""])
    hash_fix = read(EXPERIMENT / "results/preflight/runner_hash_provenance_correction.json")
    if hash_fix:
        index = lines.index("9. Artifacts and provenance")
        lines[index:index] = ["A preparation-manifest field initially hashed the derived runner as normalized UTF-8/LF text before Windows wrote CRLF bytes. Actual upload/worker byte checks already used the correct SHA-256. The preserved correction distinguishes both definitions and checked %d recorded training uploads against the actual runner byte SHA-256 %s. Its normalized UTF-8/LF SHA-256 is %s. runner_hash_provenance_correction.json and the byte-exact prior preparation record preserve this correction; graph, label and model source bytes were unchanged." %
            (hash_fix["training_upload_manifests_checked"], hash_fix["actual_runner_byte_sha256"],
             hash_fix["normalized_utf8_lf_runner_sha256"]), ""]
    interpretation = OUTPUT / "interpretation_addendum.txt"
    if interpretation.exists():
        lines.extend(["10. Interpretation of the completed measurements", "",
                      interpretation.read_text(encoding="utf-8"), ""])
    (OUTPUT / "technical_report.txt").write_text("\n".join(lines), encoding="utf-8")
    draft = ["Subject: Weighted MA-QAOA dataset and parameter-prediction experiments", "", "Hi Ian,", "",
             "Following my earlier email, I kept all training graphs weighted and used the unchanged filter to select one reference parameter vector per graph. The study remains focused on predicting MA-QAOA parameters from a weighted graph, then checking both the direct prediction and the result of local refinement on unseen graphs.", "",
             "The first run saved 63,672 graphs and accepted 63,288 (99.4%). It was interrupted before the remaining IDs ran; it did not stop because of poor data quality or reaching the spend limit. I subsequently resumed only those missing IDs in a separate experiment, preserving all original records. The current continuation status is: " + core.get("stage", "unknown") + ".", "",
             "I also distinguished consistency of the filter's selected label from a proof of global optimality. Exact classical Max-Cut checks give a lower energy bound, but p=1 MA-QAOA need not attain it. The report includes certificates and a simple analytic example illustrating that distinction.", ""]
    if population:
        draft.extend(["The resumed n=5-12 dataset contains %d saved graphs and %d usable labels (%.2f%%), with no unrun IDs. I also completed all 600 planned n=13-14 test graphs. The new matched evaluation includes every larger graph; the 16 unusable reference labels count as failed reference hits." %
                      (population["saved"], population["usable"], 100 * population["usable"] / population["saved"]), ""])
    if classical.get("p1_counts_saved_usable"):
        p1 = classical["p1_counts_saved_usable"]
        usable = sum(value for key, value in p1.items() if key.startswith("full:"))
        certified = p1.get("full:matches_classical_bound", 0) + p1.get("full:matches_stronger_locality_bound", 0)
        bad = [row for row in classical.get("known_suboptimal_star_findings", []) if row["corpus"] == "full" and row["saved_usable"]]
        draft.extend(["The classical checks now certify p=1 optimal energy within tolerance for %d of %d usable n=5-12 labels. They also found %d usable labels with explicit lower-energy p=1 witnesses, showing that seed agreement alone does not guarantee a global minimum. I kept the frozen selected labels for the matched benchmark and flagged these cases in the raw data and report." %
                      (certified, usable, len(bad)), ""])
    if bakeoff and bakeoff.get("selection_provenance", {}).get("reused_prior_selection") is False:
        draft.extend(["I repeated the seven-model bake-off on the expanded training set, using the same fixed 2000 validation graphs and selection rule. It selected %s with the %s policy. The report gives all seven validation results and the paired comparisons." %
                      (bakeoff["selection"]["chosen"], bakeoff["strategy"]), ""])
        reuse = read(EXPERIMENT / "results/preflight/expanded_winner_reuse_check.json")
        if reuse and reuse.get("passed"):
            draft.extend(["Three full D fits had finished before the request to repeat selection arrived. The repeated bake-off chose exactly the same configuration and policy, so I reused those fits after checking their training-ID and source hashes. Their earlier computation remains in the ledger.", ""])
    if summary:
        draft.extend(["The matched comparison uses the same weighted graphs, physical labels and train/validation/test split for a documented reproduction of the paper's GINE model, our selected GNN architecture and a graph-conditioned diffusion model. The common comparison uses 30 epochs; the longer full-recipe GNN fits are reported separately. The diffusion model learns to denoise the filter's labels, and does not require a new minima dataset. The completed matched training seeds were " + str(summary["completed_seeds"]["used"]) + "; the budget policy and requested seeds are documented in the report.", "",
                      "For one raw proposal per graph, the held-out results were:"])
        for corpus in ["full", "large"]:
            for model in ["paper_weighted", "ours30", "diffusion", "core_final"]:
                values = summary["groups"]["|".join(map(str, [corpus, model, "raw", 1, "all"]))]["metrics"]
                draft.append("%s / %s: direct expected-cut ratio %.4f; reference-energy hit after local refinement %.2f%%; mean %.1f function queries and %.4f seconds in CPU local optimization." %
                             (corpus, model, values["cut_ratio_before"]["mean"], 100 * values["reference_hit"]["mean"],
                              values["total_function_queries"]["mean"], values["optimizer_wall_seconds"]["mean"]))
        draft.extend(["", "The report also includes equally budgeted four-proposal comparisons, raw versus snapped predictions, node/family breakdowns, confidence intervals, ideal sampling probabilities and the 25/50/100% training-data curves. I have retained the graph records, labels, predictions, optimization traces, checkpoints and source hashes so every result can be inspected."])
    else:
        draft.extend(["The matched paper-GNN and diffusion comparison is still pending; I am not treating planned experiments as completed results. I will insert measured results once the training and evaluation gates pass."])
    if curve:
        draft.extend(["", "For nested subsets of the same training split, evaluated on the same held-out graphs:",
                      "The curve uses the core validation-selected %s policy and %d proposal heads. The matched comparison above uses one raw proposal." %
                      (curve["strategy"], bakeoff["configuration"].get("hypotheses", 1)),
                      "Fraction / training graphs / geometric mean estimated work relative to random / reference hit"])
        for percent in [25, 50, 100]:
            value = curve["fractions"][str(percent)]
            metrics = value["summary"]["metrics"]
            draft.append("%d%% / %d / %.5f / %.2f%%" % (percent, value["graphs"],
                metrics["geometric_mean_cost_ratio"]["value"], 100 * metrics["hit_rate"]["value"]))
        draft.extend(["These work ratios include estimated random fallback after unsuccessful warm starts and exclude training/inference. A common epoch cap also gives larger subsets more optimizer updates, so the curve combines data and training-compute effects."])
    if billing:
        draft.extend(["", "The task ledger as of %s records $%.2f posted and a conservative $%.2f upper reservation, under the unchanged $280 stop. The upper reservation includes usage that may not yet be posted." %
                      (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(billing["checked_epoch"])),
                       billing["reported_usd"], billing["upper_reservation_usd"])])
    draft.extend(["", "I have kept the cumulative $280 stop and recorded the compute ledger. The technical report explains the original edge-ordering issue, all paper-based implementation choices, the limitations of the uniqueness evidence, and which measured benefits concern predictions versus subsequent optimization.", "", "Thanks,", "Ashay", "", "DRAFT ONLY — not sent."])
    (OUTPUT / "email_draft.txt").write_text("\n".join(draft), encoding="utf-8")
    write_git_guidance()
    return summary


# Deliver the raw comparison and exact source bytes separately from the large core archive.
def archive(m, fits):
    for value in fits.values():
        path = m.download_artifact(value["checkpoint"])
        m.require(m.sha256(path) == value["checkpoint_sha256"], "Downloaded checkpoint differs")
    m.write_billing_summary()
    files = {}
    for path in OUTPUT.rglob("*"):
        if not path.is_file() or any(part in ["delivery", "tmp"] for part in path.relative_to(OUTPUT).parts):
            continue
        if path.name.endswith("_resume.pt") or path.suffix in [".log", ".tmp"]:
            continue
        files["paper_comparison/" + path.relative_to(OUTPUT).as_posix()] = path
    for name in ["paper_model.py", "paper_comparison.py", "research_verification.py", "complete_full100k.py"]:
        files["src/" + name] = ROOT / "src" / name
    for name in [".gitattributes", ".gitignore", "README.md"]:
        files[name] = ROOT / name
    for name in ["gnn.py", "modal_app.py", "run_filter.py", "make_dataset.py", "check_rows.py", "graph_sweep.py"]:
        files["expanded_source/src/" + name] = m.ROOT / "src" / name
    files["data/reference_picks.json"] = m.ROOT / "data/reference_picks.json"
    for relative in ["report.md", "cost_ledger.md", "gnn/splits.json", "gnn/corpus_manifest.json", "gnn/bakeoff.json",
                     "gnn/billing_summary.json", "preflight/training_population.json", "preflight/reference_floor_review.json"]:
        path = m.RESULTS / relative
        if path.exists():
            files["core_evidence/" + relative] = path
    for name in ["saved_data_bias_check.json", "bias_independent_mean_check.json"]:
        path = ROOT / "results/preflight" / name
        if path.exists():
            files["historical_evidence/" + name] = path
    destination = OUTPUT / "delivery/paper_comparison.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    readme = destination.parent / "archive_readme.txt"
    readme.write_text(
        "Weighted MA-QAOA matched model comparison\n\n"
        "paper_comparison/ contains the technical report, unsent email draft, exact protocol, raw graph/case "
        "and per-polish evidence, predictions, model weights, metrics, certificates and figures. "
        "The separately delivered core archive contains the full generated graph corpus and core weights.\n\n"
        "All expected comparison cases and numerical gates finish before packaging. A live status snapshot "
        "can still describe evaluation or archiving; the adjacent paper_comparison.json is the final archive "
        "verification proof, written after CRC, every manifest SHA-256 and credential checks pass. "
        "The live status.json becomes complete after those checks. manifest.json identifies every entry.\n",
        encoding="utf-8")
    files["ARCHIVE_README.txt"] = readme
    manifest = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in files.items()}
    LOG.info("Packaging %d comparison artifacts", len(files))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as handle:
        for name, path in files.items():
            handle.write(path, name)
        handle.writestr("manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as handle:
        m.require(handle.testzip() is None, "Comparison archive CRC failed")
        for name, entry in manifest.items():
            with handle.open(name) as item:
                m.require(hashlib.file_digest(item, "sha256").hexdigest() == entry["sha256"], "Archive file hash differs")
    secret_check = m.verify_delivery_archive_secrets(destination)
    proof = {"archive": str(destination), "sha256": digest(destination), "bytes": destination.stat().st_size,
             "files": len(files), "manifest_verified": True, "credential_scan": secret_check,
             "core_archive": str(m.RESULTS / "delivery"), "completed_epoch": time.time()}
    save(destination.with_suffix(".json"), proof)
    return proof


# Keep generated datasets/checkpoints out of Git while listing the small reproducible source set.
def write_git_guidance():
    names = ["src/make_dataset.py", "src/check_rows.py", "src/gnn.py", "src/modal_app.py",
             "src/complete_full100k.py", "src/paper_model.py", "src/paper_comparison.py",
             "src/research_verification.py", "src/audit_polish_energies.py", "src/audit_evaluation_numerics.py",
             "src/run_filter.py", "src/graph_sweep.py", "src/report_comparison_additions.py", "src/report_floor_review.py",
             "data/reference_picks.json", "data/MaxCutMAQAOAData.csv", ".gitattributes", ".gitignore", "README.md"]
    value = {name: {"sha256": digest(ROOT / name), "bytes": (ROOT / name).stat().st_size} for name in names if (ROOT / name).is_file()}
    save(OUTPUT / "source_manifest.json", value)
    text = ["GitHub delivery guidance", "", "Commit the reusable code and small pinned reference input:",
            "  src/gnn.py, src/modal_app.py, src/complete_full100k.py, src/paper_model.py, src/paper_comparison.py,",
            "  src/research_verification.py, src/audit_polish_energies.py, src/audit_evaluation_numerics.py, data/reference_picks.json.",
            "  src/report_comparison_additions.py and src/report_floor_review.py are earlier report helpers copied by the coordinator.",
            "  Also include .gitattributes, README.md and .gitignore.", "",
            "Include the exact current src/make_dataset.py, src/check_rows.py, src/run_filter.py, src/graph_sweep.py and existing data/MaxCutMAQAOAData.csv copies when adding .gitattributes. Their working-file bytes are unchanged. The -text rules also preserve the added hash-checked Python modules and reference JSON. Git may report a change against previously normalized blobs. Do not reformat these files. source_manifest.json records the required current hashes. The isolated full_100k/src/modal_app.py is a generated copy derived by complete_full100k.py; do not add the whole full_100k tree.", "",
            "Do not commit: .venv, local agent configuration, analysis scratch files, graphify outputs, generated results, datasets, checkpoints, logs, copied source trees or multi-GB archives. Deliver those as a separate checked artifact. No credentials or Modal token configuration belong in Git. Existing tracked historical results remain tracked even when a new ignore rule matches them.", "",
            "No commit, push or email send is performed by this experiment. Review git status and stage the named files explicitly.", ""]
    text.extend(["Changes in the current continuation: .gitattributes, .gitignore, README.md, src/complete_full100k.py, src/paper_comparison.py, src/paper_model.py, src/research_verification.py.",
                 "Earlier experiment additions remain untracked until reviewed: the runner, GNN, full-data coordinator, audit/report helpers and pinned reference JSON listed above.",
                 "The generated technical report, email draft, protocol, raw evidence and figures are delivery artifacts, not extra repository files.", ""])
    (OUTPUT / "github_files.txt").write_text("\n".join(text), encoding="utf-8")


# Check physics, query accounting, inference shapes and diffusion padding without paid compute.
def selfcheck():
    global CALL_BUDGET
    m = runner()
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import gnn
    import paper_model as p
    torch.set_num_threads(1)
    rows = read(ROOT / "data/reference_picks.json")["graphs"][:2]
    for row in rows:
        weights = np.asarray(row["w_sorted"])
        target = np.r_[np.asarray(row["pick"][:len(weights)]) * weights, row["pick"][len(weights):]]
        value = evaluate_case(row, [target], "raw", 1, time.time() + 60, reference_valid=True)
        assert value["reference_hit"] and value["total_function_queries"] == value["joint_simulator_calls"] + 1
        assert value["first_reference_hit_query"] == 1
        assert value["parameter_shift_equivalent_calls"] == 1 + (2 * len(target) + 1) * value["joint_simulator_calls"]
        assert abs(value["energy_after"] + value["expected_cut"]) < 1e-9
        invalid = evaluate_case(row, [target], "raw", 1, time.time() + 60, reference_valid=False)
        assert not invalid["reference_hit"] and not invalid["direct_reference_hit"] and not invalid["label_hit_before"]
        assert invalid["first_reference_hit_query"] is None
    old_budget = CALL_BUDGET
    try:
        CALL_BUDGET = 2
        value = evaluate_case(rows[0], [np.full(len(rows[0]["pick"]), .123)], "raw", 1, time.time() + 60)
        assert value["total_function_queries"] <= 2 and value["polishes"][0]["budget_exhausted"]
    finally:
        CALL_BUDGET = old_budget
    features = [p.paper_features(row) for row in rows]
    data = gnn.feature_batch(features)
    for baseline in [True, False]:
        selected_features = [gnn.graph_features(row, baseline=baseline) for row in rows]
        old = gnn.feature_batch(selected_features, random_signs=np.random.default_rng(19), baseline=baseline)
        new = p.feature_batch(selected_features, random_signs=np.random.default_rng(19), baseline=baseline)
        assert all(torch.equal(old[key], new[key]) for key in old)
    paper = p.PaperModel(data["node"].shape[-1], data["edge"].shape[-1])
    prediction = paper(data)
    edge, node = p.targets(rows, data, True)
    import check_rows
    maximum_energy_error = 0.
    generator = np.random.default_rng(723)
    # Check the actual gate conventions independently, including extreme allowed weights.
    for row in rows + [dict(rows[0], w_sorted=[.5, 2.] * (len(rows[0]["edges"]) // 2))]:
        n, count = row["n"], len(row["edges"])
        gamma = generator.uniform(-np.pi, np.pi, count)
        beta = generator.uniform(-np.pi, np.pi, n)
        weights = np.asarray(row["w_sorted"])
        rotation, mixer = -2 * weights * gamma, -beta
        bits = np.arange(1 << n)
        z = 1 - 2 * ((bits[:, None] >> np.arange(n)) & 1)
        state = np.ones(1 << n, dtype=complex) / np.sqrt(1 << n)
        for (u, v), angle in zip(row["edges"], rotation):
            state *= np.exp(-.5j * angle * z[:, u] * z[:, v])
        for vertex, angle in enumerate(mixer):
            state = np.cos(angle) * state - 1j * np.sin(angle) * state[bits ^ (1 << vertex)]
        probability = abs(state) ** 2
        energy = sum(.5 * weight * (probability @ (z[:, u] * z[:, v]) - 1)
                     for (u, v), weight in zip(row["edges"], weights))
        maximum_energy_error = max(maximum_energy_error, abs(energy - check_rows.energy(n, row["edges"], weights, gamma, beta)))
        assert np.allclose(-.5 * rotation, weights * gamma, atol=1e-14)
        assert np.array_equal(-mixer, beta)
    assert maximum_energy_error < 1e-12
    for k, row in enumerate(rows):
        count = len(row["edges"])
        assert np.allclose(-.5 * edge[k, :count].numpy(), np.asarray(row["pick"][:count]) * row["w_sorted"], atol=1e-6)
        assert np.allclose(-node[k, :row["n"]].numpy(), row["pick"][count:], atol=1e-6)
    loss = p.paper_loss(prediction, edge, node, data)
    assert abs(float(loss.detach()) - float(p.paper_loss(prediction, edge + 2 * np.pi, node - 2 * np.pi, data).detach())) < 1e-5
    loss.backward()
    features = [gnn.graph_features(row) for row in rows]
    data = gnn.feature_batch(features)
    model = p.GraphDiffusion(data["node"].shape[-1], data["edge"].shape[-1])
    loss = p.diffusion_loss(model, rows, data)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    model.eval()
    single = gnn.feature_batch(features[:1])
    timestep = torch.tensor([500, 500])
    noisy_edge = torch.randn((*data["edge_mask"].shape, 2))
    noisy_node = torch.randn((*data["node_mask"].shape, 2))
    with torch.no_grad():
        together = model(data, noisy_edge, noisy_node, timestep)
        alone = model(single, noisy_edge[:1, :single["edge"].shape[1]], noisy_node[:1, :single["node"].shape[1]], timestep[:1])
    assert torch.allclose(together[0][:1, :alone[0].shape[1]], alone[0], atol=2e-6)
    assert torch.allclose(together[1][:1, :alone[1].shape[1]], alone[1], atol=2e-6)
    old = p.SAMPLING_STEPS
    try:
        p.SAMPLING_STEPS = 3
        with torch.no_grad():
            a, b = p.diffusion_sample(model, data, 42), p.diffusion_sample(model, data, 42)
        assert torch.equal(a["theta"], b["theta"]) and a["theta"].shape[:2] == (2, 4)
    finally:
        p.SAMPLING_STEPS = old
    proof = {"passed": True, "paid_compute": False, "source_sha256": digest(ROOT / "src/paper_model.py"),
             "checks": ["reference-label oracle", "independent energy versus ideal sampling expectation", "query accounting",
                        "hard shared query cap", "invalid references count as failures", "paper 2pi cosine loss", "mixed-size diffusion forward/backward",
                        "padding invariance", "exact optimized feature collation", "finite gradients", "seeded sampling"]}
    save(OUTPUT / "local_selfcheck.json", proof)
    convention_path = OUTPUT / "paper_angle_convention_review.json"
    if not convention_path.exists():
        assert not (OUTPUT / "training_session.json").exists()
        save(convention_path, {"roundtrip_passed": True, "paid_comparison_started": False,
                              "previous_protocol_sha256": digest(OUTPUT / "protocol.json"),
                              "paper_pdf_sha256": read(OUTPUT / "protocol.json").get("paper_pdf_sha256"),
                              "mapping": p.PAPER_ANGLE_UNITS, "maximum_independent_energy_error": maximum_energy_error,
                              "checks": "F64 independent RZZ/RX versus pinned filter circuit; float32 target decode; weights .5 and 2; no new labels"})
    print(json.dumps(proof))


# Run only authorized extensions, retaining a clear stop reason and ledger if a gate fails.
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT / "tmp"
    temporary.mkdir(exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary.resolve())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT / "comparison.log", encoding="utf-8")])
    for name in ["hpack", "h2", "grpclib", "modal", "asyncio"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    if mode == "prepare":
        prepare()
        return
    if mode == "selfcheck":
        return selfcheck()
    if mode == "report":
        return report()
    if mode != "run":
        raise ValueError("Use prepare, run, report or selfcheck")
    if not (OUTPUT / "protocol.json").exists():
        prepare()
    m = None
    try:
        save(OUTPUT / "status.json", {"stage": "waiting for core pipeline", "completed": False, "updated_epoch": time.time()})
        wait_for_core()
        m = runner()
        m.verify_workspace()
        result = subprocess.run([sys.executable, str(ROOT / "src/research_verification.py")], cwd=ROOT,
                                capture_output=True, text=True)
        m.require(result.returncode == 0, "Expanded exact classical verification failed: " + result.stderr[-2000:])
        LOG.info("Updated classical certificates: %s", result.stdout.strip())
        save(OUTPUT / "status.json", {"stage": "matched training", "completed": False, "updated_epoch": time.time()})
        fits = train_models(m)
        save(OUTPUT / "status.json", {"stage": "matched evaluation", "completed": False, "updated_epoch": time.time()})
        evaluate_models(m, fits)
        m.write_billing_summary()
        report()
        archive(m, fits)
        save(OUTPUT / "status.json", {"stage": "complete", "completed": True, "updated_epoch": time.time(),
                                     "completed_seeds": read(OUTPUT / "completed_seeds.json")})
    except BaseException as error:
        LOG.exception("Comparison stopped")
        save(OUTPUT / "status.json", {"stage": "stopped", "completed": False, "error": str(error), "updated_epoch": time.time()})
        if m:
            m.write_billing_summary()
        try:
            report()
        except Exception:
            LOG.exception("Could not refresh the incomplete report")
        raise


if __name__ == "__main__":
    main()
