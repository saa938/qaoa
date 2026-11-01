"""
Usage: python src/modal_app.py
The debugger default verifies saved inputs and regenerates the report without paid compute.
Set RUN_STAGE to an approved generation, training, evaluation or delivery stage to execute it.
Run approved Modal stages only after the local input and cost gates pass.
Shared gates use complete expected-id manifests, including missing and pickless rows.
Every worker checks the exact approved source bytes before scientific work.
"""

import ast
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PREFLIGHT = RESULTS / "preflight"
REFERENCE_SOURCE = Path("C:/Users/ashay/Downloads/reference_picks.json")
REFERENCE_TARGET = ROOT / "data" / "reference_picks.json"
MODAL_PROFILE = "tejas-16842"
MODAL_WORKSPACE = "tejas-16842"
EXPECTED_HASHES = {
    "data/reference_picks.json": "a6c2831c21f940fdd50ed9bc1306ca2f89cb4eb9b91f5a023ded450c9cdd5345",
    "src/make_dataset.py": "0cdb10df3c553dc24f12ac71de184a0fddd2f3d9728da2a78b58d0474cc8b197",
    "src/check_rows.py": "b5ab974f51775c3677a172c7678738fbbe86baee568bf903477101aa013d94a2",
    "src/run_filter.py": "32697252b1d377b940a7dd13808b354a60e070cd11f7b18d4fc390abac662efc",
}
INPUT_PINS = {"numpy": "2.5.3", "scipy": "1.18.1", "networkx": "3.7", "pandas": "3.0.5"}
REFERENCE_COUNT = 21
REFERENCE_SOLVE_COUNT = 12
MIN_USABLE = 0.95
TARGET_PILOT_USABLE = 0.97
MIN_AGREEMENT = 0.98
MIN_ORACLE = 0.99
ENERGY_TOL = 1e-6
PICK_TOL = 1e-3
IAN_ENERGY_TOL = 1e-12
DATASET_ENERGY_TOL = 1e-10
TORCH_ENERGY_TOL = 1e-10
TORCH_GRAD_TOL = 1e-8
TORCH_POINTS = 1000
GPU_CHECK_TYPES = ["L4", "A10", "T4"]
GPU_CHECK_MAX_HOURLY_USD = 1.1
TRAINING_GPU_TYPES = ["L4", "A10", "L40S"]
TRAINING_MAX_GPU_HOURLY_USD = 1.952
TRAINING_POOL = None
TRAINING_IDLE_SECONDS = 90
A100_CHECK_GPU_HOURLY_USD = 2.1
A100_CHECK_BODY_SECONDS = 240
A100_CHECK_STARTUP_SECONDS = 120
A100_CHECK_WAIT_SECONDS = 360
A100_CHECK_RESOURCE_SECONDS = 420
HARD_CAP_USD = 300.0
SPEND_STOP_USD = 280.0
TRAIN_EVAL_RESERVE_USD = 45.0
CPU_SECOND_USD = 0.04730 / 3600
GIB_SECOND_USD = 0.00800 / 3600
IAN_FLOORS = {10: -11.0, 11: -15.0, 12: -12.5, 13: -9.5, 14: -12.0,
              15: -12.0, 16: -10.0, 17: -10.0, 18: -12.0, 19: -13.5}
RESEARCH_PROVENANCE = {
    "learnability": {"weighted_maxcut": 105, "field_free_spin_glass": 69},
    "relabel": {"agreed": 19, "total": 20, "maxcut_graphs": 10, "spin_glass_graphs": 10},
    "two_percent_weights": {"agreed": 16, "total": 20, "maxcut_graphs": 10, "spin_glass_graphs": 10},
    "reference_graphs": {"weighted_maxcut": 21},
}
LOG = logging.getLogger("qaoa.modal")
RUN_STAGE = "report"
APP_NAME = "weighted-maxcut-unique-labels"
VOLUME_NAME = "weighted-maxcut-unique-labels-20261028-threadfixed"
REMOTE_ROOT = "/root/qaoa"
REFERENCE_TIMEOUT = 3600
REFERENCE_RETRIES = 1
REFERENCE_MEMORY_MIB = 2048
IMAGE_BUILD_RESERVE_USD = 2.0
INITIAL_MEAN_SECONDS = {5: 25, 6: 42, 7: 78, 8: 58, 9: 85, 10: 186, 11: 242, 12: 241,
                        13: 965, 14: 570}
UPLOAD_FILES = ["src/modal_app.py", "src/run_filter.py", "src/graph_sweep.py", "src/make_dataset.py",
                "src/check_rows.py", "src/gnn.py", "data/reference_picks.json", "data/MaxCutMAQAOAData.csv"]
DELIVERY_ONLY_FILES = ["src/report_comparison_additions.py", "src/report_floor_review.py",
                       "src/audit_evaluation_numerics.py", "src/audit_polish_energies.py"]
DELIVERY_OPERATIONAL_EXCLUSIONS = {"active_operator_state.json", "workspace_allocation_diagnostic.json",
                                 "gpu_fallback_scheduling_diagnostic.json", "GPU_occupancy_diagnostic.json",
                                 "environment_allocation_diagnostic.json", "logging_incident.json"}
BATCH_CORES = 8
BATCH_MEMORY_MIB = 8192
BATCH_CONTAINERS = 40
BATCH_TIMEOUT = 3600
COMMIT_SECONDS = 120
THREAD_VARIABLES = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ORT_INTRA_OP_NUM_THREADS"]
THREADPOOL_PIN = "3.6.0"
THREAD_LIMITER = None
REPAIRED_RUN_RESERVE_USD = 14.1
FULL_SHARDS = 4000
FULL_CONTAINERS = 80
FULL_TIMEOUT = 24 * 3600
RESOURCE_TOKEN_TARGET = 4000
RESOURCE_RPC_TIMEOUT = 15
RESOURCE_LEAD_SECONDS = 45
VOLUME_COMMIT_TIMEOUT = 30
FULL_RESOURCE_HEADROOM = 1.15
TEST_RESOURCE_HEADROOM = 1.5
VALIDATION_INTERVAL = 5
EARLY_STOP_CHECKS = 4
TRAINING_TIMEOUT = 3600
EVALUATION_TIMEOUT = 900
LARGE_EVALUATION_BATCH_SIZE = 10
TEST_TIMING_TIMEOUT = 7200
PILOT_RECOVERY_TIMEOUT = 7200
DELIVERY_NAME = "weighted_maxcut_ma_qaoa"
REPORT_DECIMALS = 4
COLLECTION_RETRIES = 3
COLLECTION_RETRY_SLEEP_SECONDS = 5


# Stop the current stage on any unmet scientific or operational gate.
class GateFailure(RuntimeError):
    pass


# Persist gate evidence without permitting NaN or truncated JSON output.
def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


# Convert every failed gate into an exception rather than a diagnostic print.
def require(condition, message):
    if not condition:
        LOG.error("STOP: %s", message)
        raise GateFailure(message)


# Hash the exact file bytes without newline or encoding normalization.
def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# Install only the actual reference attachment after verifying its identity.
def install_reference():
    expected = EXPECTED_HASHES["data/reference_picks.json"]
    if REFERENCE_TARGET.exists():
        require(sha256(REFERENCE_TARGET) == expected, "Existing reference target differs; refusing to overwrite it")
        return
    require(REFERENCE_SOURCE.is_file(), "Missing actual reference attachment: %s" % REFERENCE_SOURCE)
    require(sha256(REFERENCE_SOURCE) == expected, "The reference attachment SHA-256 does not match")
    REFERENCE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REFERENCE_SOURCE, REFERENCE_TARGET)
    require(sha256(REFERENCE_TARGET) == expected, "Copied reference attachment failed its SHA-256 check")


# Refuse all later imports and launches if any approved input differs.
def check_input_hashes():
    records = []
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        record = {"path": relative, "expected": expected, "actual": actual, "passed": actual == expected}
        records.append(record)
        LOG.info("Input hash %s: %s", relative, "PASS" if record["passed"] else "FAIL")
    result = {"passed": all(row["passed"] for row in records), "files": records}
    save_json(PREFLIGHT / "input_hashes.json", result)
    require(result["passed"], "Approved input hashes differ; Stage 0 and every Modal launch are blocked")
    return result


# Reject another profile or credential override instead of silently falling back.
def lock_profile():
    require(os.environ.get("MODAL_PROFILE", MODAL_PROFILE) == MODAL_PROFILE, "Another Modal profile is selected")
    overrides = [key for key in os.environ if key.startswith("MODAL_TOKEN_") or key.startswith("MODAL_OAUTH_")]
    require(not overrides, "Environment credentials override the approved Modal profile")
    os.environ["MODAL_PROFILE"] = MODAL_PROFILE
    configure_threads()


# Override Modal's resource-derived runtime settings before NumPy or SciPy is imported.
def configure_threads():
    global THREAD_LIMITER
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    from threadpoolctl import threadpool_limits
    THREAD_LIMITER = threadpool_limits(limits=1)
    require(all(os.environ[key] == "1" for key in THREAD_VARIABLES), "Worker threading configuration failed")


# Check loaded BLAS and OpenMP pools rather than relying only on environment variables.
def numeric_thread_proof():
    from threadpoolctl import threadpool_info
    configure_threads()
    pools = threadpool_info()
    require(bool(pools) and all(pool["num_threads"] == 1 for pool in pools),
            "An active numerical library uses more than one thread")
    return {"settings": {key: os.environ[key] for key in THREAD_VARIABLES}, "pools": pools}


# Verify inherited child settings without running a graph search.
def remote_thread_inheritance_probe(volume_name, upload_hashes):
    import sys
    import subprocess
    from pathlib import Path
    import modal
    root = Path("/root/qaoa")
    sys.path.insert(0, str(root / "src"))
    import modal_app as runner
    before = {key: runner.os.environ.get(key) for key in runner.THREAD_VARIABLES}
    runner.configure_threads()
    for relative, expected in upload_hashes.items():
        runner.require(runner.sha256(root / relative) == expected, "Thread probe source bytes differ")
    import numpy
    import scipy.linalg
    parent = runner.numeric_thread_proof()
    code = "import os,json,numpy,scipy.linalg;from threadpoolctl import threadpool_info;print(json.dumps({'settings':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ORT_INTRA_OP_NUM_THREADS']},'pools':threadpool_info()}))"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    child = runner.json.loads(output.stdout)
    runner.require(all(value == "1" for value in child["settings"].values()), "Plain child did not inherit one-thread settings")
    runner.require(child["pools"] and all(pool["num_threads"] == 1 for pool in child["pools"]),
                   "A plain inherited child starts more than one numerical thread")
    result = {"runtime_settings_before": before, "parent": parent, "plain_child": child, "passed": True}
    runner.save_json(runner.RESULTS / "preflight" / "thread_inheritance_probe.json", result)
    modal.Volume.from_name(volume_name).commit()
    return result


# Import scientific code only after the input identity and profile gates pass.
def scientific_modules():
    import make_dataset as dataset
    import run_filter as run_filter
    return dataset, run_filter


# Represent the complete expected graph list without silently dropping duplicate ids.
def expected_manifest(pairs):
    result = {}
    for graph_id, n in pairs:
        require(type(graph_id) is int and type(n) is int, "Expected ids and node counts must be integers")
        require(graph_id not in result, "Duplicate expected id %d" % graph_id)
        result[graph_id] = n
    require(bool(result), "The expected-id manifest is empty")
    return result


# Read all shard rows and treat duplicate or corrupt completed records as failures.
def load_records(folder, expected):
    records = {}
    for path in sorted(Path(folder).glob("shard_*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            require(bool(line.strip()), "Blank shard record at %s:%d" % (path, line_number))
            try:
                row = json.loads(line)
            except ValueError as error:
                raise GateFailure("Invalid shard JSON at %s:%d" % (path, line_number)) from error
            require(isinstance(row, dict), "A shard row is not an object")
            graph_id = row.get("id")
            require(type(graph_id) is int and graph_id in expected, "Unexpected graph id in %s" % path)
            require(graph_id not in records, "Duplicate shard record for id %d" % graph_id)
            require(row.get("n") == expected[graph_id], "Wrong node count for id %d" % graph_id)
            records[graph_id] = row
    return records


# Require a real finite angle vector before counting any record as successful.
def has_pick(row, n):
    if not isinstance(row, dict) or row.get("n") != n:
        return False
    edges, pick = row.get("edges"), row.get("pick")
    if not isinstance(edges, list) or not isinstance(pick, list) or len(pick) != len(edges) + n:
        return False
    return all(type(value) in (float, int) and math.isfinite(value) for value in pick)


# Check evidence fields as well as the dataset builder's usable flag.
def usable(row, n):
    if not has_pick(row, n):
        return False
    floor, energy = row.get("floor"), row.get("energy")
    finite = all(type(value) in (int, float) and math.isfinite(value) for value in [floor, energy, row.get("grad_norm")])
    return bool(finite and row.get("usable") is True and row.get("confirmed") is True
                and row.get("filter") == 1 and row.get("groups") == 1
                and len(set(row.get("seeds", []))) >= 2
                and abs(energy - floor) <= ENERGY_TOL and row["grad_norm"] <= 1e-6
                and (row.get("audit") is not True or row.get("audit_same") is True))


# Always divide by the complete expected-id list, including absent and pickless rows.
def fraction(expected, records, predicate):
    require(bool(expected), "A percentage cannot have an empty expected-id denominator")
    failures = []
    for graph_id, n in expected.items():
        row = records.get(graph_id)
        if not has_pick(row, n) or not predicate(row, n):
            failures.append(graph_id)
    return {"passed": len(expected) - len(failures), "expected": len(expected),
            "fraction": (len(expected) - len(failures)) / len(expected), "failed_ids": failures}


# Verify every reference graph and weight exactly without performing a search.
def reference_build_gate():
    dataset, run_filter = scientific_modules()
    import numpy as np
    graphs = json.loads(REFERENCE_TARGET.read_text(encoding="utf-8"))["graphs"]
    require(len(graphs) == REFERENCE_COUNT, "The reference corpus must contain all 21 graphs")
    require(len({row["id"] for row in graphs}) == REFERENCE_COUNT, "Reference ids are duplicated")
    dataset.setup("full")
    results = []
    for row in graphs:
        kind, edges, weights, audit = dataset.build(row["id"], row["n"])
        same = edges == [tuple(edge) for edge in row["edges"]] and np.array_equal(weights, row["w_sorted"])
        results.append({"id": row["id"], "n": row["n"], "passed": bool(same)})
    save_json(PREFLIGHT / "reference_build.json", {"expected": REFERENCE_COUNT, "results": results})
    require(all(row["passed"] for row in results), "Reference graph or weight rebuilding failed")
    return graphs


# Enforce independent CSV checks rather than trusting the checker's exit status alone.
def csv_gate(path, expected_ids=None, ian=False):
    import check_rows
    import networkx as nx
    import numpy as np
    import pandas as pd
    path = Path(path)
    require(path.is_file(), "Missing CSV %s" % path)
    process = subprocess.run([sys.executable, str(ROOT / "src" / "check_rows.py"), str(path), "14"],
                             capture_output=True, text=True, check=False)
    require(process.returncode == 0, "Independent CSV checker crashed: %s" % process.stderr)
    LOG.info("%s", process.stdout.strip())
    frame = pd.read_csv(path)
    if ian:
        require(len(frame) == 20, "Ian's CSV must have all 20 rows")
    else:
        require(expected_ids is not None, "Dataset CSV verification requires expected pick-bearing ids")
        ids = frame["Id"].tolist()
        require(len(ids) == len(set(ids)) and set(ids) == set(expected_ids), "Merged CSV id coverage differs")
    maximum = 0.0
    for index, row in frame.iterrows():
        n = int(row["Number of Nodes"])
        edges = [tuple(edge) for edge in ast.literal_eval(row["Edges"])]
        order = [tuple(sorted(edge)) for edge in nx.Graph(edges).edges()]
        gamma_edges = order if ian else [tuple(edge) for edge in ast.literal_eval(row["Gamma Edges"])]
        weights = np.ones(len(edges)) if ian else np.asarray(ast.literal_eval(row["Weights"]), float)
        gammas = np.ravel(ast.literal_eval(row["Predicted Gammas" if ian else "Gammas"]))
        betas = np.ravel(ast.literal_eval(row["Predicted Betas" if ian else "Betas"]))
        require(0 < n <= 14 and gamma_edges == order, "Invalid node count or gamma order at CSV row %d" % index)
        require(len(weights) == len(edges) == len(gammas) and len(betas) == n, "CSV angle or weight count differs")
        require(np.isfinite(np.r_[weights, gammas, betas, float(row["Energy"])]).all(), "Nonfinite CSV value")
        error = abs(check_rows.energy(n, gamma_edges, weights, gammas, betas) - float(row["Energy"]))
        maximum = max(maximum, error)
    threshold = IAN_ENERGY_TOL if ian else DATASET_ENERGY_TOL
    require(maximum <= threshold, "CSV energy error %.17g exceeds %.17g" % (maximum, threshold))
    return {"rows": len(frame), "max_energy_error": maximum, "threshold": threshold}


# Require a reproduced floor and pick for every expected reference solve.
def reference_solve_gate(reference_rows, records):
    dataset, run_filter = scientific_modules()
    require(len(reference_rows) == REFERENCE_SOLVE_COUNT, "Reference solve gate requires all 12 expected graphs")
    expected = expected_manifest([(row["id"], row["n"]) for row in reference_rows])
    results = []
    for reference in reference_rows:
        row = records.get(reference["id"])
        valid = has_pick(row, reference["n"])
        if valid:
            valid = usable(row, reference["n"]) and row.get("edges") == reference["edges"] and row.get("w_sorted") == reference["w_sorted"]
        if valid:
            distance = dataset.pick_dist(row["n"], row["edges"], row["w_sorted"], reference["pick"], row["pick"])
            valid = abs(row["floor"] - reference["floor"]) <= ENERGY_TOL and distance <= PICK_TOL
        results.append({"id": reference["id"], "passed": bool(valid)})
    require(all(row["passed"] for row in results), "A reference solve is missing or fails its floor/pick gate")
    return {"expected": len(expected), "results": results}


# Apply pilot gates using the complete manifest for usable, audit and comparison rates.
def pilot_gates(expected, pilot, pilot400, allow_budget_fallback=False):
    dataset, run_filter = scientific_modules()
    metrics = {"usable": fraction(expected, pilot, usable)}
    metrics["audit"] = fraction(expected, pilot, lambda row, n: row.get("audit") is True and row.get("audit_same") is True)
    same, comparisons = {}, []
    for graph_id, n in expected.items():
        left, right = pilot.get(graph_id), pilot400.get(graph_id)
        valid = has_pick(left, n) and has_pick(right, n)
        if valid:
            valid = left["edges"] == right["edges"] and left["w_sorted"] == right["w_sorted"]
        distance = None
        if valid:
            distance = dataset.pick_dist(n, left["edges"], left["w_sorted"], left["pick"], right["pick"])
            valid = abs(left["floor"] - right["floor"]) <= ENERGY_TOL and distance <= PICK_TOL
        same[graph_id] = bool(valid)
        comparisons.append({"id": graph_id, "n": n, "type": None if left is None else left.get("type"),
                            "same": bool(valid), "distance": distance,
                            "floor": None if left is None else left.get("floor"),
                            "floor400": None if right is None else right.get("floor")})
    metrics["compare"] = fraction(expected, pilot, lambda row, n: same[row["id"]])
    metrics["investigate_usable"] = metrics["usable"]["fraction"] < TARGET_PILOT_USABLE
    save_json(RESULTS / "dataset_pilot" / "runner_gates.json", metrics)
    save_json(RESULTS / "dataset_pilot" / "runner_compare.json", comparisons)
    require(metrics["usable"]["fraction"] >= MIN_USABLE, "Pilot usable fraction is below 95% of all expected ids")
    require(metrics["audit"]["fraction"] >= MIN_AGREEMENT, "Pilot audit agreement is below 98% of all expected ids")
    if not allow_budget_fallback:
        require(metrics["compare"]["fraction"] >= MIN_AGREEMENT, "Pilot comparison is below 98%; a budgeted STALL=400 decision is required")
    metrics["requires_stall400"] = metrics["compare"]["fraction"] < MIN_AGREEMENT
    return metrics


# Keep coverage separate from quality when the user explicitly selects saved records.
def dataset_gate(expected, records, saved_population=False):
    require(not set(records) - set(expected), "Dataset contains unexpected ids")
    population = {i: expected[i] for i in records} if saved_population else expected
    overall = fraction(population, records, usable)
    by_n = {str(n): fraction({i: size for i, size in population.items() if size == n}, records, usable)
            for n in sorted(set(population.values()))}
    denominator = "saved records" if saved_population else "all expected ids"
    require(overall["fraction"] >= MIN_USABLE, "Dataset usable fraction is below 95% of " + denominator)
    return {"overall": overall, "by_n": by_n, "quality_denominator": denominator,
            "planned": len(expected), "saved": len(records), "not_run": len(expected) - len(records)}


# Diagnose interruption bias from existing records only, without launching Modal compute.
def saved_data_bias_check():
    from collections import Counter
    import numpy as np
    config = load_frozen_plan()
    dataset = configured_dataset("full", config)
    expected = expected_manifest(dataset.selected("full"))
    records = load_records(RESULTS / "dataset", expected)
    pilot = [json.loads(line) for path in (RESULTS / "dataset_pilot").glob("shard*.jsonl")
             for line in path.read_text().splitlines() if line.strip()]
    rng = np.random.default_rng(20261029)
    sizes = {}
    for n in sorted(set(expected.values())):
        rows = [row for row in records.values() if row["n"] == n]
        controls = [row for row in pilot if row["n"] == n]
        require(len(controls) == 40, "Bias check requires the complete random pilot, 40 per n")
        a = np.asarray([row["secs_main"] for row in rows], float)
        b = np.asarray([row["secs_main"] for row in controls], float)
        require(np.isfinite(a).all() and np.isfinite(b).all() and (a >= 0).all() and (b > 0).all(),
                "Invalid saved or pilot timing")
        # Pilot sampling uncertainty dominates; both independent samples are resampled.
        numerator = np.concatenate([rng.choice(a, (100, len(a))).mean(axis=1) for unused in range(50)])
        denominator = rng.choice(b, (5000, len(b))).mean(axis=1)
        interval = np.quantile(numerator / denominator, [.025, .975]).tolist()
        family = {}
        for kind in sorted({row["type"] for row in rows} | {row["type"] for row in controls}):
            left = [row["secs_main"] for row in rows if row["type"] == kind]
            right = [row["secs_main"] for row in controls if row["type"] == kind]
            family[kind] = {"saved": len(left), "pilot": len(right),
                            "saved_mean": float(np.mean(left)) if left else None,
                            "pilot_mean": float(np.mean(right)) if right else None}
        sizes[str(n)] = {"saved": len(a), "pilot": len(b), "saved_mean_secs_main": float(a.mean()),
                         "pilot_mean_secs_main": float(b.mean()), "mean_ratio": float(a.mean() / b.mean()),
                         "bootstrap_mean_ratio_95ci": interval,
                         "saved_median": float(np.median(a)), "pilot_median": float(np.median(b)),
                         "saved_p90": float(np.quantile(a, .9)), "pilot_p90": float(np.quantile(b, .9)),
                         "by_type": family,
                         "clearly_easy_signal": interval[1] < .8}
    shards = Counter(i % FULL_SHARDS for i in records)
    missing_whole, missing_tail, partial, nonprefix = [], [], [], []
    shard_ids = [[] for unused in range(FULL_SHARDS)]
    for graph_id in sorted(expected):
        shard_ids[graph_id % FULL_SHARDS].append(graph_id)
    for shard, planned in enumerate(shard_ids):
        saved = [i for i in planned if i in records]
        if not saved:
            missing_whole.extend(planned)
        elif len(saved) < len(planned):
            partial.append(shard)
            if saved != planned[:len(saved)]:
                nonprefix.append(shard)
            missing_tail.extend(i for i in planned if i not in records)
    result = {"no_modal_compute": True, "saved": len(records), "pilot": len(pilot), "by_n": sizes,
              "bootstrap_resamples": 5000, "bootstrap_seed": 20261029,
              "practical_signal_rule": "95% mean-ratio interval entirely below 0.8 in any n; examine family cells too",
              "clearly_easy": any(item["clearly_easy_signal"] for item in sizes.values()),
              "shards": {"expected": FULL_SHARDS, "complete": sum(count == 25 for count in shards.values()),
                         "partial": len(partial), "unstarted": FULL_SHARDS - len(shards),
                         "nonprefix_partial_shards": nonprefix,
                         "missing_in_whole_shards": len(missing_whole), "missing_in_started_shard_tails": len(missing_tail),
                         "whole_shard_missing_by_n": dict(Counter(expected[i] for i in missing_whole)),
                         "tail_missing_by_n": dict(Counter(expected[i] for i in missing_tail))},
              "interpretation": "Whole unstarted shards and interrupted ascending-id shard tails both contribute. Size mix is biased toward small n; within-size timing must be assessed separately. Timing is a difficulty proxy, not proof of label or topology representativeness.",
              "captured_epoch": time.time()}
    save_json(PREFLIGHT / "saved_data_bias_check.json", result)
    require(not result["clearly_easy"], "Saved graphs are clearly easier than the random pilot; stop before further compute")
    return result


# Freeze the user-approved saved population without changing the original graph identity recipe.
def freeze_saved_training_dataset():
    bias = saved_data_bias_check()
    config = load_frozen_plan()
    dataset = configured_dataset("full", config)
    expected = expected_manifest(dataset.selected("full"))
    records = load_records(RESULTS / "dataset", expected)
    gate = dataset_gate(expected, records, saved_population=True)
    selected = sorted(i for i, row in records.items() if usable(row, expected[i]))
    require(len(records) == 63672 and len(selected) == 63288, "Saved population differs from the approved 63,288 usable ids")
    manifest = {"ids": selected, "not_run_ids": sorted(set(expected) - set(records)),
                "rejected_saved_ids": sorted(set(records) - set(selected)),
                "saved_ids": sorted(records), "expected": {i: expected[i] for i in selected},
                "saved": len(records), "usable": len(selected), "not_run": len(expected) - len(records),
                "quality": gate, "frozen_plan_sha256": sha256(PREFLIGHT / "frozen_plan.json"),
                "input_hashes": EXPECTED_HASHES, "bias_check_sha256": sha256(PREFLIGHT / "saved_data_bias_check.json"),
                "row_hashes": {i: hashlib.sha256(json.dumps(records[i], sort_keys=True).encode()).hexdigest() for i in selected},
                "learning_curve": {"fractions": [.25, .5, 1.0], "status": "scheduling clarification pending"},
                "created_epoch": time.time()}
    path = PREFLIGHT / "training_population.json"
    if path.exists():
        existing = load_training_population()
        require(existing["ids"] == selected and existing["row_hashes"] == manifest["row_hashes"], "Frozen training population changed")
        return existing
    save_json(path, manifest)
    save_json(PREFLIGHT / "training_population_lock.json", {"sha256": sha256(path)})
    save_json(PREFLIGHT / "full_corpus_coverage.json", gate)
    csv = csv_gate(RESULTS / "dataset" / "dataset.csv", [i for i, row in records.items() if has_pick(row, expected[i])])
    csv.update({"population": "all saved pick-bearing rows; training uses only frozen usable ids",
                "csv_sha256": sha256(RESULTS / "dataset" / "dataset.csv")})
    save_json(PREFLIGHT / "full_corpus_csv_gate.json", csv)
    return manifest


# Require the exact approved subset and immutable source plan whenever training data is read.
def load_training_population():
    path = PREFLIGHT / "training_population.json"
    lock = read_evidence(PREFLIGHT / "training_population_lock.json")
    require(lock is not None and path.exists() and sha256(path) == lock["sha256"], "Training population hash differs")
    value = json.loads(path.read_text())
    require(value["frozen_plan_sha256"] == sha256(PREFLIGHT / "frozen_plan.json")
            and value["input_hashes"] == EXPECTED_HASHES, "Training population uses different graph identities")
    require(len(value["ids"]) == len(set(value["ids"])) == 63288 and len(value["not_run_ids"]) == 36328,
            "Frozen training population count differs")
    return value


# Validate expected Ian floors while treating absent or pickless solves as failures.
def ian_floor_gate(records):
    for graph_id, expected in IAN_FLOORS.items():
        row = records.get(graph_id)
        require(has_pick(row, 8), "Missing or pickless Ian floor row %d" % graph_id)
        actual = row.get("floor")
        require(type(actual) in (int, float) and math.isfinite(actual), "Nonfinite Ian floor")
        require(abs(actual - expected) <= ENERGY_TOL, "Ian floor differs at row %d" % graph_id)


# Require every specified independent torch energy and gradient comparison to pass.
def torch_gate(checks):
    require(len(checks) == TORCH_POINTS, "Torch verification requires all 1000 random points")
    for row in checks:
        require(5 <= row.get("n", 0) <= 12, "Torch check has an invalid graph size")
        for key, threshold in [("energy_error", TORCH_ENERGY_TOL), ("gradient_error", TORCH_GRAD_TOL)]:
            value = row.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= threshold,
                    "Torch %s failed" % key)
    require(set(row["n"] for row in checks) == set(range(5, 13)), "Torch verification omits a graph size")


# Count missing and pickless oracle starts as failed starts in the full evaluation set.
def oracle_gate(expected, records):
    result = fraction(expected, records, lambda row, n: row.get("oracle_hit") is True)
    require(result["fraction"] >= MIN_ORACLE, "Oracle hit fraction is below 99% of expected ids")
    return result


# Refuse launches when any billed cost or conservative remaining projection is unknown.
def budget_gate(spent, committed, remaining):
    values = [spent, committed, remaining]
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values),
            "Budget values must be finite nonnegative dollar amounts")
    require(spent < SPEND_STOP_USD, "Actual spend has reached the $280 stop threshold")
    total = math.fsum(values)
    require(total <= HARD_CAP_USD, "Projected total %.2f exceeds the $300 cap" % total)
    return total


# Bound the cheapest legal remaining dataset using zeros for every unfinished pilot timing.
def minimum_budget_gate(expected, records):
    dataset, run_filter = scientific_modules()
    require(not set(records) - set(expected), "Budget timing records contain unexpected ids")
    factor = 1000 / dataset.PLAN_COUNTS[12]
    minimum = {n: int(round(count * factor)) for n, count in dataset.PLAN_COUNTS.items()}
    seconds, counts = {}, {}
    for graph_id, n in expected.items():
        counts[n] = counts.get(n, 0) + 1
        row = records.get(graph_id)
        value = 0 if row is None else row.get("secs_main")
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "Pilot timing must be finite and nonnegative")
        seconds[n] = seconds.get(n, 0.0) + value
    require(set(counts) == set(minimum) and all(count == 40 for count in counts.values()),
            "The minimum-budget proof requires the full 40-per-size pilot manifest")
    by_n = {n: minimum[n] * seconds[n] / counts[n] * 1.05 * (CPU_SECOND_USD + GIB_SECOND_USD)
            for n in minimum}
    lower = math.fsum(by_n.values())
    result = {"completed": len(records), "expected": len(expected), "minimum_uniform_factor": factor,
              "minimum_counts": minimum, "minimum_full_projection_lower_bound": lower,
              "by_n": by_n, "training_evaluation_reserve": TRAIN_EVAL_RESERVE_USD,
              "past_spend_assumed": 0, "test_cost_assumed": 0, "comparison_cost_assumed": 0,
              "missing_pilot_main_seconds_assumed": 0,
              "lower_bound_plus_reserve": lower + TRAIN_EVAL_RESERVE_USD}
    save_json(PREFLIGHT / "minimum_budget_gate.json", result)
    require(result["lower_bound_plus_reserve"] <= HARD_CAP_USD,
            "Even the minimum permitted uniform shrink projects above $300 with all other costs set to zero")
    return result


# Verify runtime pins, the original CSV and deterministic builds after the hash gate.
def stage_zero():
    import importlib.metadata
    require(all(os.environ.get(key) == "1" for key in THREAD_VARIABLES),
            "Stage 0 requires one library thread per graph worker")
    require(sys.version_info[:2] == (3, 13), "Data generation requires Python 3.13")
    versions = {name: importlib.metadata.version(name) for name in INPUT_PINS}
    require(versions == INPUT_PINS, "Scientific package pins differ")
    original = csv_gate(ROOT / "data" / "MaxCutMAQAOAData.csv", ian=True)
    references = reference_build_gate()
    result = {"passed": True, "versions": versions, "original_csv": original, "reference_graphs": len(references)}
    require(importlib.metadata.version("threadpoolctl") == THREADPOOL_PIN, "Thread inspection tool version differs")
    result["numerical_threads"] = numeric_thread_proof()
    save_json(PREFLIGHT / "stage_zero.json", result)
    return result


# Verify only the authorized credential with a bounded read-only workspace lookup.
def verify_workspace():
    import asyncio
    from modal.config import _lookup_workspace, config
    response = asyncio.run(_lookup_workspace(config.get("server_url"),
                                             config.get("token_id", profile=MODAL_PROFILE, use_env=False),
                                             config.get("token_secret", profile=MODAL_PROFILE, use_env=False)))
    result = {"profile": MODAL_PROFILE, "workspace": response.username,
              "verified": response.username == MODAL_WORKSPACE}
    save_json(PREFLIGHT / "workspace.json", result)
    require(result["verified"], "The selected credential does not belong to the approved workspace")
    return result


# Reserve the bounded reference launch and project the rest before building an image.
def reference_launch_budget(reference_rows):
    dataset, run_filter = scientific_modules()
    rate = CPU_SECOND_USD + GIB_SECOND_USD
    full = math.fsum(dataset.PLAN_COUNTS[n] * INITIAL_MEAN_SECONDS[n] * 1.05 * rate
                     for n in dataset.PLAN_COUNTS)
    test = math.fsum(dataset.TEST_COUNTS[n] * INITIAL_MEAN_SECONDS[n] * 1.05 * rate
                     for n in dataset.TEST_COUNTS)
    pilot = math.fsum(40 * INITIAL_MEAN_SECONDS[n] * (2 + 2.3) * rate for n in dataset.PLAN_COUNTS)
    reference_rate = CPU_SECOND_USD + REFERENCE_MEMORY_MIB / 1024 * GIB_SECOND_USD
    bounded = len(reference_rows) * REFERENCE_TIMEOUT * (REFERENCE_RETRIES + 1) * reference_rate
    expected = math.fsum(row["secs_stall100"] * reference_rate for row in reference_rows)
    reservation = bounded + IMAGE_BUILD_RESERVE_USD
    remaining = full + test + pilot + TRAIN_EVAL_RESERVE_USD
    projection = budget_gate(0.0, reservation, remaining)
    result = {"reference_expected_compute_usd": expected, "reference_timeout_compute_bound_usd": bounded,
              "image_build_reserve_usd": IMAGE_BUILD_RESERVE_USD, "reference_reservation_usd": reservation,
              "preliminary_remaining_usd": remaining, "projected_total_with_reference_reservation_usd": projection,
              "basis": "Live list prices and task historical timings; replace with pilot measurements before full generation",
              "cpu_per_reference": 1, "memory_limit_mib": REFERENCE_MEMORY_MIB,
              "reference_timeout_seconds": REFERENCE_TIMEOUT, "max_retries": REFERENCE_RETRIES}
    save_json(PREFLIGHT / "reference_launch_budget.json", result)
    return result


# Validate the binary upload inside each container before performing its reference search.
def remote_reference(reference, volume_name, upload_hashes):
    import sys
    import time
    from pathlib import Path
    import modal
    started = time.monotonic()
    root = Path("/root/qaoa")
    sys.path.insert(0, str(root / "src"))
    import modal_app as runner
    threads_before = {key: runner.os.environ.get(key) for key in runner.THREAD_VARIABLES}
    runner.configure_threads()
    runner.PREFLIGHT = runner.RESULTS / "reference_stage" / ("proof_%d" % reference["id"])
    volume = modal.Volume.from_name(volume_name)
    for relative, expected in upload_hashes.items():
        runner.require(runner.sha256(root / relative) == expected, "Uploaded file bytes differ: %s" % relative)
    runner.check_input_hashes()
    environment = runner.stage_zero()
    dataset, run_filter = runner.scientific_modules()
    dataset.setup("full")
    record = dataset.solve(reference["id"], reference["n"], False)
    runner.save_json(runner.RESULTS / "reference_stage" / ("id_%d.json" % reference["id"]), record)
    runner.commit_volume(volume)
    distance = None
    if runner.has_pick(record, reference["n"]):
        distance = dataset.pick_dist(reference["n"], record["edges"], record["w_sorted"], reference["pick"], record["pick"])
    passed = bool(distance is not None and distance <= runner.PICK_TOL
                  and abs(record["floor"] - reference["floor"]) <= runner.ENERGY_TOL
                  and record.get("confirmed") is True)
    proof = {"id": reference["id"], "n": reference["n"], "passed": passed,
             "pick_distance": distance, "record": record, "environment": environment,
             "thread_settings_before": threads_before,
             "thread_settings": {key: runner.os.environ.get(key) for key in runner.THREAD_VARIABLES},
             "upload_hashes": upload_hashes, "elapsed_seconds": time.monotonic() - started}
    runner.save_json(runner.PREFLIGHT / "result.json", proof)
    runner.commit_volume(volume)
    return proof


# Launch only the gated references and cancel the app immediately on any failure.
def launch_references():
    import modal
    require(not (PREFLIGHT / "reference_launch.json").exists(), "A reference launch already exists; reconcile it before submitting again")
    verify_workspace()
    graphs = json.loads(REFERENCE_TARGET.read_text(encoding="utf-8"))["graphs"]
    candidates = [row for row in graphs if 6 <= row["n"] <= 9]
    selected = sorted(candidates, key=lambda row: (row["secs_stall100"], row["id"]))[:REFERENCE_SOLVE_COUNT]
    require(len(selected) == REFERENCE_SOLVE_COUNT, "The reference launch does not have its 12 expected graphs")
    budget = reference_launch_budget(selected)
    upload_hashes = {relative: sha256(ROOT / relative) for relative in UPLOAD_FILES}
    save_json(PREFLIGHT / "upload_manifest.json", upload_hashes)
    image = modal.Image.debian_slim(python_version="3.13").pip_install(
        *[name + "==" + version for name, version in INPUT_PINS.items()])
    image = image.env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "PYTHONPATH": REMOTE_ROOT + "/src", "PYTHONUNBUFFERED": "1"})
    for relative in UPLOAD_FILES:
        image = image.add_local_file(ROOT / relative, REMOTE_ROOT + "/" + relative, copy=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
    app = modal.App(APP_NAME)
    function = app.function(image=image, cpu=(1.0, 1.0), memory=(1024, REFERENCE_MEMORY_MIB),
                            timeout=REFERENCE_TIMEOUT, retries=REFERENCE_RETRIES, max_containers=12,
                            scaledown_window=2, volumes={REMOTE_ROOT + "/results": volume},
                            serialized=True, name="reference_check")(remote_reference)
    proofs = []
    started = time.monotonic()
    try:
        with app.run(detach=True):
            launch = {"app_id": app.app_id, "dashboard": app.get_dashboard_url(), "profile": MODAL_PROFILE,
                      "volume": VOLUME_NAME, "expected_ids": [row["id"] for row in selected], "budget": budget}
            save_json(PREFLIGHT / "reference_launch.json", launch)
            for proof in function.map(selected, kwargs={"volume_name": VOLUME_NAME, "upload_hashes": upload_hashes},
                                      order_outputs=False, return_exceptions=True):
                require(not isinstance(proof, BaseException), "Remote reference failed: %s" % proof)
                proofs.append(proof)
                save_json(PREFLIGHT / "reference_results.json", proofs)
                require(proof.get("passed") is True, "Reference floor or pick differs for id %d" % proof["id"])
                LOG.info("Reference %d passed (%d/%d), distance %.3g", proof["id"], len(proofs), len(selected), proof["pick_distance"])
            records = {proof["id"]: proof["record"] for proof in proofs}
            require(len(proofs) == len(selected), "Reference results are missing")
            result = reference_solve_gate(selected, records)
            save_json(PREFLIGHT / "reference_stage_gate.json", result)
            app.stop()
    except BaseException:
        if app.app_id is not None:
            app.stop()
        raise
    finally:
        save_json(PREFLIGHT / "reference_client_timing.json", {"elapsed_seconds": time.monotonic() - started,
                                                               "completed": len(proofs), "expected": len(selected)})
    return proofs


# Repair only an unfinished final JSON line before a shard resumes.
def repair_partial_shard(path):
    path = Path(path)
    if not path.exists():
        return 0
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    kept = 0
    for index, line in enumerate(lines):
        try:
            json.loads(line)
        except ValueError:
            require(index == len(lines) - 1 and not line.endswith(b"\n"), "Corrupt completed shard record: %s" % path)
            with path.open("r+b") as stream:
                stream.truncate(kept)
            return len(raw) - kept
        kept += len(line)
    return 0


# Run fresh shard subprocesses on eight cores and commit completed records regularly.
def remote_pilot_batch(job, mode, volume_name, upload_hashes):
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path
    import modal
    root = Path("/root/qaoa")
    sys.path.insert(0, str(root / "src"))
    import modal_app as runner
    threads_before = {key: runner.os.environ.get(key) for key in runner.THREAD_VARIABLES}
    runner.configure_threads()
    started = time.monotonic()
    runner.PREFLIGHT = runner.RESULTS / "worker_proofs" / mode / ("batch_%04d" % job["batch"])
    volume = modal.Volume.from_name(volume_name)
    for relative, expected in upload_hashes.items():
        runner.require(runner.sha256(root / relative) == expected, "Uploaded bytes differ: %s" % relative)
    runner.check_input_hashes()
    runner.stage_zero()
    folder = runner.RESULTS / ("dataset_" + mode)
    logs = folder / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    todo, active, records = list(job["graphs"]), {}, []
    last_commit = time.monotonic()
    try:
        while todo or active:
            if "deadline_epoch" in job:
                runner.require(time.time() <= job["deadline_epoch"], "Pilot wall-clock reservation expired")
            while todo and len(active) < runner.BATCH_CORES:
                graph_id, n = todo.pop(0)
                shard = folder / ("shard_%04d_of_%04d.jsonl" % (graph_id, job["n_shards"]))
                runner.repair_partial_shard(shard)
                log = (logs / ("graph_%d.log" % graph_id)).open("a", encoding="utf-8")
                process = subprocess.Popen([sys.executable, str(root / "src" / "modal_app.py"), "pilot_record", mode,
                                            str(graph_id), str(job["n_shards"])], cwd=root,
                                           stdout=log, stderr=subprocess.STDOUT)
                active[graph_id] = {"process": process, "log": log, "path": shard, "n": n}
            for graph_id, state in list(active.items()):
                code = state["process"].poll()
                if code is None:
                    continue
                state["log"].close()
                runner.require(code == 0, "Shard subprocess failed for id %d; see its saved log" % graph_id)
                rows = [json.loads(line) for line in state["path"].read_text().splitlines()]
                runner.require(len(rows) == 1 and rows[0]["id"] == graph_id and rows[0]["n"] == state["n"],
                               "One-graph pilot shard coverage differs for id %d" % graph_id)
                records.append(rows[0])
                del active[graph_id]
            if time.monotonic() - last_commit >= runner.COMMIT_SECONDS:
                runner.commit_volume(volume)
                last_commit = time.monotonic()
                print("%s batch %d: %d/%d records complete" % (mode, job["batch"], len(records), len(job["graphs"])), flush=True)
            if active:
                time.sleep(0.5)
    finally:
        for state in active.values():
            process = state["process"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            state["log"].close()
        runner.commit_volume(volume)
    result = {"batch": job["batch"], "mode": mode, "records": records,
              "thread_settings_before": threads_before,
              "thread_settings": {key: runner.os.environ.get(key) for key in runner.THREAD_VARIABLES},
              "elapsed_seconds": time.monotonic() - started, "upload_hashes": upload_hashes}
    runner.save_json(runner.PREFLIGHT / "result.json", result)
    runner.commit_volume(volume)
    return result


# Define the same binary source upload for each subsequent data stage.
def data_image():
    import modal
    image = modal.Image.debian_slim(python_version="3.13").pip_install(
        *[name + "==" + version for name, version in INPUT_PINS.items()], "threadpoolctl==" + THREADPOOL_PIN)
    image = image.env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "PYTHONPATH": REMOTE_ROOT + "/src", "PYTHONUNBUFFERED": "1"})
    for relative in UPLOAD_FILES:
        image = image.add_local_file(ROOT / relative, REMOTE_ROOT + "/" + relative, copy=True)
    return image


# Reserve both pilot modes including retries while leaving the full plan costed.
def pilot_launch_budget():
    dataset, run_filter = scientific_modules()
    graphs = json.loads(REFERENCE_TARGET.read_text())["graphs"]
    selected = sorted([row for row in graphs if 6 <= row["n"] <= 9], key=lambda row: row["secs_stall100"])[:12]
    initial = reference_launch_budget(selected)
    rate = BATCH_CORES * CPU_SECOND_USD + BATCH_MEMORY_MIB / 1024 * GIB_SECOND_USD
    bound = BATCH_CONTAINERS * BATCH_TIMEOUT * (REFERENCE_RETRIES + 1) * rate
    pilot_bound = bound
    previous = PREFLIGHT / "pilot_launch.json"
    if previous.exists():
        prior = json.loads(previous.read_text())["budget"]
        pilot_bound = prior.get("pilot_timeout_retry_bound_usd", prior["per_pilot_mode_timeout_retry_bound_usd"])
    both_bound = pilot_bound + bound
    pilot_expected = math.fsum(40 * INITIAL_MEAN_SECONDS[n] * (2 + 2.3) *
                              (CPU_SECOND_USD + GIB_SECOND_USD) for n in dataset.PLAN_COUNTS)
    remaining = initial["preliminary_remaining_usd"] - pilot_expected
    total = budget_gate(0.0, initial["reference_reservation_usd"] + both_bound + REPAIRED_RUN_RESERVE_USD, remaining)
    result = {"per_pilot_mode_timeout_retry_bound_usd": bound,
              "pilot_timeout_retry_bound_usd": pilot_bound, "pilot400_timeout_retry_bound_usd": bound,
              "both_modes_timeout_retry_bound_usd": both_bound,
              "both_modes_expected_compute_usd": pilot_expected,
              "projected_total_with_both_pilot_bounds_usd": total,
              "reserved_reference_usd": initial["reference_reservation_usd"],
              "reserved_oversubscribed_run_and_probe_usd": REPAIRED_RUN_RESERVE_USD,
              "remaining_full_test_training_eval_usd": remaining,
              "cpu_per_container": BATCH_CORES, "memory_mib_per_container": BATCH_MEMORY_MIB,
              "max_containers": BATCH_CONTAINERS, "timeout_seconds": BATCH_TIMEOUT, "retries": REFERENCE_RETRIES}
    save_json(PREFLIGHT / "pilot_launch_budget.json", result)
    return result


# Run and merge one complete pilot mode without changing the approved builder.
def launch_pilot_mode(mode):
    import modal
    require(mode in ["pilot", "pilot400"], "Unsupported pilot mode")
    require((PREFLIGHT / "reference_stage_gate.json").is_file(), "Reference gate has not passed")
    require(not (PREFLIGHT / (mode + "_launch.json")).exists(), "Pilot launch already exists; reconcile it before resubmission")
    verify_workspace()
    dataset, run_filter = scientific_modules()
    dataset.setup(mode)
    expected = expected_manifest(dataset.selected(mode))
    require(len(expected) == 320, "The pilot must include all 320 expected ids")
    n_shards = len(dataset.plan())
    deadline = time.time() + BATCH_TIMEOUT * (REFERENCE_RETRIES + 1)
    jobs = [{"batch": k, "graphs": list(expected.items())[k::BATCH_CONTAINERS], "n_shards": n_shards,
             "deadline_epoch": deadline}
            for k in range(BATCH_CONTAINERS)]
    assigned = [graph_id for job in jobs for graph_id, n in job["graphs"]]
    require(len(assigned) == len(set(assigned)) == len(expected), "Pilot assignments overlap or omit ids")
    budget = pilot_launch_budget()
    upload_hashes = {relative: sha256(ROOT / relative) for relative in UPLOAD_FILES}
    image = data_image()
    volume = modal.Volume.from_name(VOLUME_NAME)
    app = modal.App(APP_NAME + "-" + mode)
    function = app.function(image=image, cpu=(BATCH_CORES, BATCH_CORES),
                            memory=(BATCH_MEMORY_MIB, BATCH_MEMORY_MIB), timeout=BATCH_TIMEOUT,
                            retries=REFERENCE_RETRIES, max_containers=BATCH_CONTAINERS, scaledown_window=2,
                            volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                            name="pilot_batch")(remote_pilot_batch)
    records, batches = {}, []
    folder = RESULTS / ("dataset_" + mode)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with app.run(detach=True):
            save_json(PREFLIGHT / (mode + "_launch.json"), {"app_id": app.app_id, "dashboard": app.get_dashboard_url(),
                      "profile": MODAL_PROFILE, "expected": expected, "jobs": jobs,
                      "upload_hashes": upload_hashes, "budget": budget, "created_epoch": time.time()})
            for batch in function.map(jobs, kwargs={"mode": mode, "volume_name": VOLUME_NAME,
                                                    "upload_hashes": upload_hashes},
                                      order_outputs=False, return_exceptions=True):
                require(not isinstance(batch, BaseException), "Pilot batch failed: %s" % batch)
                for row in batch["records"]:
                    graph_id = row["id"]
                    require(graph_id in expected and graph_id not in records, "Unexpected or repeated pilot id")
                    records[graph_id] = row
                    path = folder / ("shard_%04d_of_%04d.jsonl" % (graph_id, n_shards))
                    path.write_text(json.dumps(row, allow_nan=False) + "\n", encoding="utf-8")
                batches.append({key: value for key, value in batch.items() if key != "records"})
                save_json(PREFLIGHT / (mode + "_batches.json"), batches)
                LOG.info("%s: %d/%d expected graph records received", mode, len(records), len(expected))
                if mode == "pilot":
                    minimum_budget_gate(expected, records)
            app.stop()
    except BaseException:
        if app.app_id is not None:
            app.stop()
        raise
    observed = load_records(folder, expected)
    summary = dataset_gate(expected, observed)
    save_json(PREFLIGHT / (mode + "_coverage.json"), summary)
    if mode == "pilot":
        audit = fraction(expected, observed, lambda row, n: row.get("audit") is True and row.get("audit_same") is True)
        save_json(PREFLIGHT / "pilot_audit.json", audit)
        require(audit["fraction"] >= MIN_AGREEMENT, "Pilot audit agreement is below 98% of all 320 expected ids")
    process = subprocess.run([sys.executable, str(ROOT / "src" / "make_dataset.py"), "merge", mode],
                             capture_output=True, text=True, check=False)
    require(process.returncode == 0, "Pilot CSV merge failed: %s" % process.stderr)
    LOG.info("%s", process.stdout)
    pick_ids = [graph_id for graph_id, row in observed.items() if has_pick(row, expected[graph_id])]
    csv = csv_gate(folder / "dataset.csv", pick_ids)
    save_json(PREFLIGHT / (mode + "_csv_gate.json"), csv)
    return expected, observed


# Run the audited pilot before paying for the confirmation-length comparison.
def launch_pilots():
    expected, pilot = launch_pilot_mode("pilot")
    minimum_budget_gate(expected, pilot)
    expected400, pilot400 = launch_pilot_mode("pilot400")
    require(expected == expected400, "Pilot modes use different expected-id manifests")
    result = pilot_gates(expected, pilot, pilot400, allow_budget_fallback=True)
    minimum = minimum_budget_gate(expected, pilot)
    if result["requires_stall400"]:
        budget_gate(REPAIRED_RUN_RESERVE_USD, pilot_launch_budget()["both_modes_timeout_retry_bound_usd"],
                    2.3 * minimum["minimum_full_projection_lower_bound"] + TRAIN_EVAL_RESERVE_USD)
    save_json(PREFLIGHT / "stopping_rule.json", {"stall": 400 if result["requires_stall400"] else 100,
              "reason": "Pilot comparison below 98%; final budget still requires test timings" if result["requires_stall400"]
                        else "At least 98% agreement over all 320 expected pilot ids"})
    dataset, run_filter = scientific_modules()
    dataset.compare()
    return result


# Recheck existing pilot outputs without resubmitting paid graph searches.
def finalize_pilots():
    dataset, run_filter = scientific_modules()
    dataset.setup("pilot")
    expected = expected_manifest(dataset.selected("pilot"))
    pilot = load_records(RESULTS / "dataset_pilot", expected)
    pilot400 = load_records(RESULTS / "dataset_pilot400", expected)
    result = pilot_gates(expected, pilot, pilot400, allow_budget_fallback=True)
    minimum = minimum_budget_gate(expected, pilot)
    if result["requires_stall400"]:
        spend = reconciled_spend_reservation()["spend_reservation_usd"]
        budget_gate(spend, 0.0, 2.3 * minimum["minimum_full_projection_lower_bound"] + TRAIN_EVAL_RESERVE_USD)
    save_json(PREFLIGHT / "stopping_rule.json", {"stall": 400 if result["requires_stall400"] else 100,
              "reason": "Below 98% comparison; the frozen-plan budget must include a 2.3 multiplier" if result["requires_stall400"]
                        else "At least 98% comparison agreement over all 320 expected ids"})
    dataset.compare()
    return result


# Apply a frozen runtime plan without changing a byte of the approved builder.
def configured_dataset(mode, config):
    dataset, run_filter = scientific_modules()
    require(config["input_hashes"] == EXPECTED_HASHES, "Frozen plan names different approved input bytes")
    dataset.PLAN_COUNTS = {int(n): count for n, count in config["plan_counts"].items()}
    dataset.TEST_COUNTS = {int(n): count for n, count in config["test_counts"].items()}
    dataset.STALL = config["stall"]
    dataset.setup(mode)
    dataset.OUT_DIR = RESULTS / config["folders"][mode]
    dataset.MERGED = dataset.OUT_DIR / "dataset.csv"
    return dataset


# Run one fresh shard process with the exact frozen plan and single-threaded libraries.
def child_shard(mode, shard, n_shards, config_path, expected_config_hash=None):
    configure_threads()
    for relative, expected in EXPECTED_HASHES.items():
        require(sha256(ROOT / relative) == expected, "Child input bytes differ: %s" % relative)
    if expected_config_hash is not None:
        require(sha256(Path(config_path)) == expected_config_hash, "Frozen plan changed before a fresh shard started")
    config = json.loads(Path(config_path).read_text())
    dataset = configured_dataset(mode, config)
    proof = numeric_thread_proof()
    save_json(RESULTS / "worker_proofs" / "child_threads" / mode / ("shard_%04d.json" % shard), proof)
    path = dataset.OUT_DIR / ("shard_%04d_of_%04d.jsonl" % (shard, n_shards))
    repair_partial_shard(path)
    dataset.run_shard(shard, n_shards, mode)


# Initialize each pilot or timing child explicitly before importing the approved builder.
def child_pilot_record(mode, shard, n_shards):
    configure_threads()
    for relative, expected in EXPECTED_HASHES.items():
        require(sha256(ROOT / relative) == expected, "Pilot child source bytes differ")
    dataset, run_filter = scientific_modules()
    dataset.setup(mode)
    proof = numeric_thread_proof()
    save_json(RESULTS / "worker_proofs" / "child_threads" / mode / ("shard_%04d.json" % shard), proof)
    dataset.run_shard(shard, n_shards, mode)


# Atomically reserve bounded resource time before allowing graph workers to continue.
def take_resource_token(queue_name):
    import asyncio
    import queue
    import modal

    # Bound metadata delays so children cannot run indefinitely on an expired slice.
    async def receive():
        resource_queue = modal.Queue.from_name(queue_name)
        return await asyncio.wait_for(resource_queue.get.aio(block=False), timeout=RESOURCE_RPC_TIMEOUT)

    try:
        seconds = asyncio.run(receive())
    except (queue.Empty, TimeoutError) as error:
        raise GateFailure("The stage resource reservation is exhausted or unavailable") from error
    require(seconds is not None, "The stage resource reservation is exhausted")
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 60,
            "Invalid resource reservation token")
    return seconds


# Bound commit delays so a failed storage call cannot leave paid workers idle indefinitely.
def commit_volume(volume):
    import asyncio

    # Keep each remote storage wait shorter than the reserved resource-time lead.
    async def commit():
        await asyncio.wait_for(volume.commit.aio(), timeout=VOLUME_COMMIT_TIMEOUT)

    return asyncio.run(commit())


# Pull small shards through eight fresh processes while consuming a shared time budget.
def remote_shard_batch(job, mode, config_hash, volume_name, upload_hashes):
    import sys
    import time
    import subprocess
    from pathlib import Path
    import modal
    root = Path("/root/qaoa")
    sys.path.insert(0, str(root / "src"))
    import modal_app as runner
    resource_deadline = time.monotonic() + runner.take_resource_token(job["budget_queue"])
    tokens_taken = 1
    threads_before = {key: runner.os.environ.get(key) for key in runner.THREAD_VARIABLES}
    runner.configure_threads()
    runner.PREFLIGHT = runner.RESULTS / "worker_proofs" / mode / ("batch_%04d" % job["batch"])
    for relative, expected in upload_hashes.items():
        runner.require(runner.sha256(root / relative) == expected, "Uploaded shard-worker source bytes differ")
    runner.check_input_hashes()
    runner.stage_zero()
    config_path = runner.RESULTS / "preflight" / "frozen_plan.json"
    runner.require(runner.sha256(config_path) == config_hash, "Uploaded frozen plan differs")
    config = runner.json.loads(config_path.read_text())
    dataset = runner.configured_dataset(mode, config)
    expected = runner.expected_manifest(dataset.selected(mode))
    folder = dataset.OUT_DIR
    logs = folder / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(volume_name)
    started, last_commit = time.monotonic(), time.monotonic()
    todo, active, finished = list(job["shards"]), {}, []
    try:
        while todo or active:
            runner.require(time.time() <= job["deadline_epoch"], "Stage resource reservation expired; stop spending")
            if time.monotonic() >= resource_deadline - runner.RESOURCE_LEAD_SECONDS:
                resource_deadline += runner.take_resource_token(job["budget_queue"])
                tokens_taken += 1
            while todo and len(active) < runner.BATCH_CORES:
                shard = todo.pop(0)
                path = folder / ("shard_%04d_of_%04d.jsonl" % (shard, job["n_shards"]))
                runner.repair_partial_shard(path)
                handle = (logs / ("shard_%04d.log" % shard)).open("a", encoding="utf-8")
                process = subprocess.Popen([sys.executable, str(root / "src" / "modal_app.py"), "shard",
                                            mode, str(shard), str(job["n_shards"]), str(config_path), config_hash],
                                           cwd=root, stdout=handle, stderr=subprocess.STDOUT)
                active[shard] = {"process": process, "handle": handle, "path": path}
            for shard, state in list(active.items()):
                code = state["process"].poll()
                if code is None:
                    continue
                state["handle"].close()
                runner.require(code == 0, "Shard subprocess failed: %s/%d" % (mode, shard))
                records = [runner.json.loads(line) for line in state["path"].read_text().splitlines() if line.strip()]
                assigned = {i for i in expected if i % job["n_shards"] == shard}
                runner.require(len(records) == len(assigned) and {row["id"] for row in records} == assigned,
                               "Completed shard omits or duplicates an expected id")
                finished.append({"shard": shard, "records": len(records),
                                 "usable": sum(runner.usable(row, expected[row["id"]]) for row in records)})
                del active[shard]
            if time.monotonic() - last_commit >= runner.COMMIT_SECONDS:
                active_records, active_usable = 0, 0
                for state in active.values():
                    if state["path"].exists():
                        data = state["path"].read_bytes()
                        lines = data.split(b"\n")[:-1]
                        rows = [runner.json.loads(line) for line in lines if line.strip()]
                        active_records += len(rows)
                        active_usable += sum(runner.usable(row, expected[row["id"]]) for row in rows)
                runner.save_json(runner.PREFLIGHT / "progress.json", {"batch": job["batch"],
                    "completed_shards": len(finished), "expected_shards": len(job["shards"]),
                    "completed_records": sum(row["records"] for row in finished) + active_records,
                    "usable_records": sum(row["usable"] for row in finished) + active_usable,
                    "tokens_taken": tokens_taken, "captured_epoch": time.time()})
                runner.commit_volume(volume)
                print("%s batch %d: %d/%d shards done" % (mode, job["batch"], len(finished), len(job["shards"])), flush=True)
                last_commit = time.monotonic()
            if active:
                time.sleep(0.5)
    finally:
        for state in active.values():
            process = state["process"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            state["handle"].close()
        runner.commit_volume(volume)
    result = {"mode": mode, "batch": job["batch"], "finished": finished,
              "elapsed_seconds": time.monotonic() - started, "config_hash": config_hash,
              "tokens_taken": tokens_taken,
              "thread_settings_before": threads_before,
              "thread_settings": {key: runner.os.environ.get(key) for key in runner.THREAD_VARIABLES}}
    runner.save_json(runner.PREFLIGHT / "result.json", result)
    runner.commit_volume(volume)
    return result


# Download only the chosen evidence tree and reject paths outside the local results folder.
def download_volume(prefix, destination=None):
    import modal
    from concurrent.futures import ThreadPoolExecutor
    volume = modal.Volume.from_name(VOLUME_NAME)
    destination = RESULTS / prefix if destination is None else Path(destination)
    entries = [entry for entry in volume.iterdir(prefix, recursive=True)
               if entry.type == modal.volume.FileEntryType.FILE]

    # Preserve every byte while ensuring remote names cannot escape the selected destination.
    def copy_entry(entry):
        relative = Path(entry.path).relative_to(prefix)
        target = destination / relative
        require(target.resolve().is_relative_to(destination.resolve()), "Remote path escapes results destination")
        data = b"".join(volume.read_file(entry.path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return len(data)

    with ThreadPoolExecutor(max_workers=16) as pool:
        sizes = list(pool.map(copy_entry, entries))
    return {"files": len(sizes), "bytes": sum(sizes), "volume": VOLUME_NAME, "prefix": prefix}


# Measure the ten prescribed held-out graphs before freezing any training-size counts.
def launch_test_timings():
    import modal
    require((RESULTS / "dataset_pilot" / "runner_gates.json").exists(), "Pilot comparison has not completed")
    require(not (PREFLIGHT / "test_timing_launch.json").exists(), "Test timing launch already exists")
    verify_workspace()
    ids = list(range(5)) + list(range(300, 305))
    expected = {graph_id: 13 if graph_id < 300 else 14 for graph_id in ids}
    timeout = TEST_TIMING_TIMEOUT
    bound = 10 * timeout * 2 * (CPU_SECOND_USD + 2 * GIB_SECOND_USD)
    prior = pilot_launch_budget()
    total = budget_gate(0.0, prior["reserved_reference_usd"] + REPAIRED_RUN_RESERVE_USD
                        + prior["both_modes_timeout_retry_bound_usd"] + bound,
                        prior["remaining_full_test_training_eval_usd"])
    volume = modal.Volume.from_name(VOLUME_NAME)
    app = modal.App(APP_NAME + "-test-timing")
    upload = {name: sha256(ROOT / name) for name in UPLOAD_FILES}
    function = app.function(image=data_image(), cpu=(1, 1), memory=(1024, 2048), timeout=timeout,
                            retries=1, max_containers=10, scaledown_window=2,
                            volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                            name="test_timing")(remote_pilot_batch)
    jobs = [{"batch": k, "graphs": [[graph_id, expected[graph_id]]], "n_shards": 600,
             "deadline_epoch": time.time() + 2 * timeout} for k, graph_id in enumerate(ids)]
    records = {}
    folder = RESULTS / "dataset_test"
    folder.mkdir(exist_ok=True)
    try:
        with app.run(detach=True):
            save_json(PREFLIGHT / "test_timing_launch.json", {"app_id": app.app_id, "expected": expected,
                      "jobs": jobs, "reserve_usd": bound, "projected_total_usd": total,
                      "upload_hashes": upload, "created_epoch": time.time()})
            for batch in function.map(jobs, kwargs={"mode": "test", "volume_name": VOLUME_NAME,
                                                    "upload_hashes": upload}, order_outputs=False,
                                      return_exceptions=True):
                require(not isinstance(batch, BaseException), "Test timing worker failed: %s" % batch)
                for row in batch["records"]:
                    require(row["id"] in expected and row["id"] not in records, "Repeated test timing id")
                    records[row["id"]] = row
                    path = folder / ("shard_%04d_of_0600.jsonl" % row["id"])
                    path.write_text(json.dumps(row, allow_nan=False) + "\n", encoding="utf-8")
                LOG.info("Test timing: %d/10 graph records received", len(records))
    finally:
        if app.app_id:
            app.stop()
    save_json(PREFLIGHT / "test_timing_completed.json", {"app_id": app.app_id, "stopped_epoch": time.time()})
    require(set(records) == set(expected), "Test timing omits an expected id")
    require(all(type(row.get("secs_main")) in (int, float) and math.isfinite(row["secs_main"])
                and row["secs_main"] > 0 for row in records.values()), "Test timing contains an invalid runtime")
    result = {"expected": expected, "records": list(records.values()),
              "mean_seconds": {n: math.fsum(row["secs_main"] for row in records.values() if row["n"] == n) / 5
                               for n in [13, 14]}}
    save_json(PREFLIGHT / "test_timings.json", result)
    return result


# Combine reported app billing with resource bounds for every interval still unreported.
def reconciled_spend_reservation():
    import datetime
    import decimal
    import modal
    specifications = [
        ("reference_launch.json", "reference_stage_gate.json", 12, 1, 2),
        ("oversubscribed_pilot/pilot_launch.json", "oversubscribed_pilot/repair.json", 42, 8, 8),
        ("thread_probe_launch.json", "thread_probe_result.json", 1, 8, 8),
        ("pilot_launch.json", "pilot_coverage.json", 40, 8, 8),
        ("pilot400_launch.json", "pilot400_coverage.json", 40, 8, 8),
        ("test_timing_launch.json", "test_timings.json", 10, 1, 2),
        ("full_corpus_launch.json", "full_corpus_coverage.json", FULL_CONTAINERS, 8, 8),
        ("test_corpus_launch.json", "test_corpus_coverage.json", 10, 8, 8),
    ]
    apps = {}
    now = time.time()
    for launch_name, completed_name, containers, cpu, memory in specifications:
        path = PREFLIGHT / launch_name
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if not record.get("app_id"):
            continue
        completed = PREFLIGHT / completed_name
        end = min(now, completed.stat().st_mtime + 10) if completed.exists() else now
        start = record.get("created_epoch", path.stat().st_mtime)
        apps[record["app_id"]] = {"launch": launch_name, "start": start, "end": end,
                                  "containers": containers, "cpu": cpu, "memory_gib": memory,
                                  "maximum_hourly_usd": containers * (cpu * CPU_SECOND_USD + memory * GIB_SECOND_USD) * 3600}
    for path in (PREFLIGHT / "paid_apps").glob("*.json"):
        record = json.loads(path.read_text())
        apps[record["app_id"]] = {"launch": record["name"], "start": record["created_epoch"],
                                  "end": min(now, record.get("stopped_epoch", now) + 10),
                                  "maximum_hourly_usd": record["maximum_hourly_usd"],
                                  "budget_cap_usd": record.get("budget_cap_usd")}
    start = datetime.datetime.fromtimestamp(min(value["start"] for value in apps.values()) - 3600,
                                            datetime.timezone.utc)
    report = modal.Workspace.from_context().billing.report(start=start, resolution="h")
    volumes = {}
    for name in [VOLUME_NAME, "weighted-maxcut-unique-labels-20261028"]:
        volume = modal.Volume.from_name(name)
        volume.hydrate()
        volumes[volume.object_id] = name
    billing = [item for item in report if item.object_id in apps or item.object_id in volumes]
    actual_decimal = sum((decimal.Decimal(str(item.cost)) for item in billing), decimal.Decimal(0))
    actual = math.fsum(float(item.cost) for item in billing)
    require(abs(float(actual_decimal) - actual) <= 1e-8, "Independent billing sums differ")
    for item in billing:
        breakdown = math.fsum(float(value) for value in item.cost_by_resource.values())
        require(abs(breakdown - float(item.cost)) <= 1e-6, "Billing resource breakdown differs from its total")
    reported = {(item.object_id, int(item.interval_start.timestamp())) for item in billing}
    missing = []
    for app_id, spec in apps.items():
        interval = int(spec["start"] // 3600) * 3600
        while interval < spec["end"]:
            if (app_id, interval) not in reported:
                seconds = max(0.0, min(spec["end"], interval + 3600) - max(spec["start"], interval))
                rate = spec["maximum_hourly_usd"] / 3600
                missing.append({"app_id": app_id, "interval_epoch": interval, "seconds": seconds,
                                "resource_upper_bound_usd": seconds * rate})
            interval += 3600
    for app_id, spec in apps.items():
        if spec.get("budget_cap_usd") is not None:
            billed = math.fsum(float(item.cost) for item in billing if item.object_id == app_id)
            rows = [row for row in missing if row["app_id"] == app_id]
            bound = math.fsum(row["resource_upper_bound_usd"] for row in rows)
            cap = max(0.0, spec["budget_cap_usd"] - billed)
            if bound > cap:
                for row in rows:
                    row["resource_upper_bound_usd"] *= cap / bound
    unreported = math.fsum(row["resource_upper_bound_usd"] for row in missing)
    result = {"checked_epoch": now, "reported_app_cost_usd": actual,
              "reported_app_cost_decimal_usd": str(actual_decimal),
              "unreported_resource_reservation_usd": unreported,
              "image_and_startup_reserve_usd": IMAGE_BUILD_RESERVE_USD,
              "spend_reservation_usd": actual + unreported + IMAGE_BUILD_RESERVE_USD,
              "apps": apps, "unreported_intervals": missing,
              "task_volume_ids": volumes,
              "billing_rows": [{"app_id": item.object_id, "interval": item.interval_start.isoformat(),
                                "cost_usd": str(item.cost), "description": item.description,
                                "cost_by_resource": {key: str(value) for key, value in item.cost_by_resource.items()}}
                               for item in billing]}
    save_json(PREFLIGHT / "billing_reconciliation.json", result)
    require(actual < SPEND_STOP_USD, "Reported actual spend reached $280")
    return result


# Track each later paid app independently so completed hours and open intervals both count.
def register_paid_app(app_id, name, maximum_hourly_usd, stopped=False, created_epoch=None, budget_cap_usd=None):
    path = PREFLIGHT / "paid_apps" / (app_id + ".json")
    record = json.loads(path.read_text()) if path.exists() else {
        "app_id": app_id, "name": name, "maximum_hourly_usd": maximum_hourly_usd,
        "created_epoch": time.time() if created_epoch is None else created_epoch, "profile": MODAL_PROFILE}
    if stopped:
        record["stopped_epoch"] = time.time()
    if budget_cap_usd is not None:
        record["budget_cap_usd"] = budget_cap_usd
    save_json(path, record)


# Stop only task-owned apps on the approved profile whenever a required gate fails.
def stop_task_apps():
    specifications = [("reference_launch.json", "reference_stage_gate.json"),
                      ("oversubscribed_pilot/pilot_launch.json", "oversubscribed_pilot/repair.json"),
                      ("thread_probe_launch.json", "thread_probe_result.json"),
                      ("pilot_launch.json", "pilot_coverage.json"), ("pilot400_launch.json", "pilot400_coverage.json"),
                      ("test_timing_launch.json", "test_timing_completed.json"),
                      ("full_corpus_launch.json", "full_corpus_coverage.json"),
                      ("test_corpus_launch.json", "test_corpus_coverage.json"), ("ian_launch.json", "ian_floor_gate.json")]
    ids = set()
    for name, completed in specifications:
        path = PREFLIGHT / name
        if path.exists() and not (PREFLIGHT / completed).exists():
            record = json.loads(path.read_text())
            app_id = record.get("app_id")
            if app_id and app_id.startswith("ap-") and record.get("profile", MODAL_PROFILE) == MODAL_PROFILE:
                ids.add(app_id)
    for path in (PREFLIGHT / "paid_apps").glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("stopped_epoch") is not None:
            ids.discard(record["app_id"])
        else:
            ids.add(record["app_id"])
    child_env = {key: value for key, value in os.environ.items()
                 if not key.startswith("MODAL_TOKEN_") and not key.startswith("MODAL_OAUTH_")}
    child_env["MODAL_PROFILE"] = MODAL_PROFILE
    child_env["MODAL_LOGLEVEL"] = "WARNING"
    from concurrent.futures import ThreadPoolExecutor

    # Issue independent stops together so one unavailable endpoint cannot delay another app.
    def stop_one(app_id):
        try:
            result = subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id,
                                     "--profile", MODAL_PROFILE, "--yes"], capture_output=True, text=True,
                                    env=child_env, timeout=60)
            return {"app_id": app_id, "returncode": result.returncode,
                    "detail": (result.stdout + result.stderr)[-1000:]}
        except subprocess.TimeoutExpired:
            return {"app_id": app_id, "returncode": -1, "detail": "Stop request timed out; stop is unconfirmed"}

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(ids)))) as pool:
        stopped = list(pool.map(stop_one, sorted(ids)))
    save_json(PREFLIGHT / "task_stop_actions.json", stopped)
    return stopped


# Record a halted run using measured gate evidence without inventing later results.
def write_stop_report(reason):
    lines = ["# Run halted", "", str(reason), "",
             "Further task-owned compute has been stopped or a stop was requested. See preflight/task_stop_actions.json.", ""]
    for name, label in [("pilot_coverage.json", "Audited pilot"), ("full_corpus_coverage.json", "Full corpus"),
                         ("test_corpus_coverage.json", "Larger held-out corpus")]:
        path = PREFLIGHT / name
        if path.exists():
            value = json.loads(path.read_text())["overall"]
            lines.append("%s usable: %d/%d (%.6f%%). Missing and pickless ids count as failures."
                         % (label, value["passed"], value["expected"], 100 * value["fraction"]))
    audit = PREFLIGHT / "pilot_audit.json"
    if audit.exists():
        value = json.loads(audit.read_text())
        lines.append("Audited-pilot agreement: %d/%d (%.6f%%)." % (value["passed"], value["expected"], 100 * value["fraction"]))
    billing = PREFLIGHT / "billing_reconciliation.json"
    if billing.exists():
        value = json.loads(billing.read_text())
        lines.extend(["", "Last reported task resource cost: $%.8f. Unreported usage reservation: $%.8f. "
                      "This is the last saved reconciliation, not a claim that an open billing interval is final."
                      % (value["reported_app_cost_usd"], value["unreported_resource_reservation_usd"])])
    lines.extend(["", "Completed records and worker evidence remain on the task Volumes. Unfinished comparisons do not support a model-performance claim.",
                  "", "No email was sent and no GitHub push was made. The approved scientific source files remain unchanged.",
                  "", "The earlier 174-instance result comprised 105 weighted Max-Cut and 69 spin-glass cases. "
                  "The relabel 19/20 and weight-perturbation 16/20 results pooled ten of each. All 21 supplied references are weighted Max-Cut."])
    if (PREFLIGHT / "logging_incident.json").exists():
        lines.extend(["", "An early debug log exposed the approved Modal token in tool output. Saved transport log lines were removed. "
                      "The token should be rotated after this run."])
    (RESULTS / "gnn").mkdir(exist_ok=True)
    (RESULTS / "gnn" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Resume only pilot ids absent after a stopped allocation, with one graph per core.
def resume_pilot_records(mode="pilot"):
    import modal
    require(mode in ["pilot", "pilot400"], "Unsupported pilot recovery mode")
    expected = {int(key): n for key, n in json.loads((PREFLIGHT / "pilot_launch.json").read_text())["expected"].items()}
    folder = RESULTS / ("dataset_" + mode)
    records = load_records(folder, expected)
    missing = [(i, n) for i, n in expected.items() if i not in records]
    if missing:
        verify_workspace()
        timeout = PILOT_RECOVERY_TIMEOUT
        bound = len(missing) * timeout * 2 * (CPU_SECOND_USD + 2 * GIB_SECOND_USD)
        spending = reconciled_spend_reservation()
        dataset, run_filter = scientific_modules()
        means = {n: (math.fsum(row["secs_main"] for row in records.values() if row["n"] == n)
                     + sum(size == n for i, size in missing) * timeout) / 40 for n in dataset.PLAN_COUNTS}
        minimum_full = math.fsum(count / 4 * means[n] for n, count in dataset.PLAN_COUNTS.items())
        minimum_full *= 1.05 * (CPU_SECOND_USD + GIB_SECOND_USD)
        comparison = pilot_launch_budget()["pilot400_timeout_retry_bound_usd"] if mode == "pilot" else 0
        remaining = minimum_full + comparison + TRAIN_EVAL_RESERVE_USD + 15
        total = budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"]
                            - spending["reported_app_cost_usd"] + bound, remaining)
        volume = modal.Volume.from_name(VOLUME_NAME)
        upload = {name: sha256(ROOT / name) for name in UPLOAD_FILES}
        app = modal.App(APP_NAME + "-" + mode + "-recovery")
        function = app.function(image=data_image(), cpu=(1, 1), memory=(1024, 2048), timeout=timeout, retries=1,
                                max_containers=len(missing), scaledown_window=2,
                                volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                                name="pilot_recovery")(remote_pilot_batch)
        jobs = [{"batch": 1000 + k, "graphs": [[graph_id, n]], "n_shards": 100000,
                 "deadline_epoch": time.time() + 2 * timeout} for k, (graph_id, n) in enumerate(missing)]
        hourly = len(missing) * (.0473 + 2 * .008)
        try:
            with app.run(detach=True):
                register_paid_app(app.app_id, mode + " missing records", hourly)
                save_json(PREFLIGHT / (mode + "_recovery_launch.json"), {"app_id": app.app_id, "jobs": jobs,
                          "reserve_usd": bound, "projected_total_usd": total, "upload_hashes": upload,
                          "basis": "Minimum allowed conditional full allocation with missing main times bounded by 7200 seconds; no counts frozen",
                          "created_epoch": time.time()})
                for batch in function.map(jobs, kwargs={"mode": mode, "volume_name": VOLUME_NAME,
                                                        "upload_hashes": upload}, order_outputs=False,
                                          return_exceptions=True):
                    require(not isinstance(batch, BaseException), "Pilot recovery failed: %s" % batch)
                    for row in batch["records"]:
                        require(row["id"] in expected and row["id"] not in records, "Pilot recovery repeated a completed id")
                        records[row["id"]] = row
                        (folder / ("shard_%04d_of_100000.jsonl" % row["id"])).write_text(
                            json.dumps(row, allow_nan=False) + "\n", encoding="utf-8")
                    LOG.info("%s recovery: %d/%d complete records", mode, len(records), len(expected))
        finally:
            if app.app_id:
                app.stop()
                register_paid_app(app.app_id, mode + " missing records", hourly, stopped=True)
    require(set(records) == set(expected), "Pilot recovery is missing expected timings")
    coverage = dataset_gate(expected, records)
    save_json(PREFLIGHT / (mode + "_coverage.json"), coverage)
    if mode == "pilot":
        audit = fraction(expected, records, lambda row, n: row.get("audit") is True and row.get("audit_same") is True)
        save_json(PREFLIGHT / "pilot_audit.json", audit)
        require(audit["fraction"] >= MIN_AGREEMENT, "Pilot audit agreement is below 98% of all 320 expected ids")
    dataset, run_filter = scientific_modules()
    dataset.setup(mode)
    dataset.merge()
    csv = csv_gate(folder / "dataset.csv", [i for i, row in records.items() if has_pick(row, expected[i])])
    save_json(PREFLIGHT / (mode + "_csv_gate.json"), csv)
    return coverage


# Compute held-out unweighted floors and verify the old paper prediction in sorted order.
def remote_ian_floor(row_number, volume_name, upload_hashes):
    import sys
    import time
    from pathlib import Path
    import modal
    root = Path("/root/qaoa")
    sys.path.insert(0, str(root / "src"))
    import modal_app as runner
    runner.configure_threads()
    runner.PREFLIGHT = runner.RESULTS / "ian_floors" / ("proof_%d" % row_number)
    for relative, expected in upload_hashes.items():
        runner.require(runner.sha256(root / relative) == expected, "Ian worker uploaded bytes differ")
    runner.check_input_hashes()
    runner.stage_zero()
    dataset, run_filter = runner.scientific_modules()
    dataset.setup("full")
    import numpy as np
    frame = dataset.pd.read_csv(root / "data" / "MaxCutMAQAOAData.csv")
    source = frame.iloc[row_number]
    original = runner.ast.literal_eval(source["Edges"])
    edges = run_filter.canonical_edges(original)
    weights = np.ones(len(edges))
    record = run_filter.run(row_number, edges, 8, weights, 1, 1000 * row_number + 11, False, None)
    order = [tuple(sorted(edge)) for edge in dataset.nx.Graph(original).edges()]
    where = {edge: k for k, edge in enumerate(order)}
    old_gamma = np.asarray(runner.ast.literal_eval(source["Predicted Gammas"])).ravel()
    old_beta = np.asarray(runner.ast.literal_eval(source["Predicted Betas"])).ravel()
    prediction = np.r_[old_gamma[[where[tuple(edge)] for edge in edges]], old_beta]
    energy, unused_batch, gradient, dimension = run_filter.make_energy(8, edges, weights, 1)
    old_energy = float(energy(prediction))
    runner.require(abs(old_energy - float(source["Energy"])) <= runner.IAN_ENERGY_TOL,
                   "Ian paper prediction does not reproduce its recorded energy")
    record.update({"id": row_number, "n": 8, "m": len(edges), "edges": [list(edge) for edge in edges],
                   "w_sorted": weights.tolist(), "type": str(source["Type"]), "weighted": False,
                   "max_cut": dataset.max_cut(8, edges, weights), "paper_prediction_raw": prediction.tolist(),
                   "paper_energy_before": old_energy, "paper_csv_energy": float(source["Energy"])})
    if row_number in runner.IAN_FLOORS:
        runner.require(runner.has_pick(record, 8) and abs(record["floor"] - runner.IAN_FLOORS[row_number]) <= runner.ENERGY_TOL,
                       "Ian floor differs at row %d" % row_number)
    volume = modal.Volume.from_name(volume_name)
    runner.save_json(runner.RESULTS / "ian_floors" / ("row_%02d.json" % row_number), record)
    runner.commit_volume(volume)
    return record


# Run the twenty held-out floor searches under a separately recorded small reservation.
def launch_ian_floors(parallel_with_test=False):
    import modal
    require(not (PREFLIGHT / "ian_launch.json").exists(), "Ian floor launch already exists")
    require((PREFLIGHT / "full_corpus_coverage.json").exists(), "Verify the saved training population first")
    if parallel_with_test:
        load_training_population()
        active_test = read_evidence(PREFLIGHT / "test_completion_launch.json")
        require(active_test is not None and active_test["config_hash"] == sha256(PREFLIGHT / "frozen_plan.json"),
                "Parallel Ian checks require the approved fixed test run")
        require((PREFLIGHT / "full_corpus_csv_gate.json").exists()
                and read_evidence(PREFLIGHT / "test_smoke_gate.json")["passed"], "Required earlier data checks have not passed")
    else:
        require((PREFLIGHT / "test_corpus_coverage.json").exists(), "Complete the test corpus before this sequential stage")
    verify_workspace()
    spending = reconciled_spend_reservation()
    reserve = 20 * 3600 * 2 * (CPU_SECOND_USD + 2 * GIB_SECOND_USD)
    budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"],
                TRAIN_EVAL_RESERVE_USD)
    volume = modal.Volume.from_name(VOLUME_NAME)
    upload = {name: sha256(ROOT / name) for name in UPLOAD_FILES}
    app = modal.App(APP_NAME + "-ian-floors")
    function = app.function(image=data_image(), cpu=(1, 1), memory=(1024, 2048), timeout=3600, retries=1,
                            max_containers=20, scaledown_window=2, volumes={REMOTE_ROOT + "/results": volume},
                            serialized=True, name="ian_floor")(remote_ian_floor)
    records = {}
    try:
        with app.run(detach=True):
            register_paid_app(app.app_id, "Ian floors", 20 * (.0473 + 2 * .008))
            save_json(PREFLIGHT / "ian_launch.json", {"app_id": app.app_id, "reserve_usd": reserve,
                      "upload_hashes": upload, "expected_ids": list(range(20)), "created_epoch": time.time(),
                      "parallel_with_independent_test_generation": parallel_with_test})
            for record in function.map(range(20), kwargs={"volume_name": VOLUME_NAME, "upload_hashes": upload},
                                        order_outputs=False, return_exceptions=True):
                require(not isinstance(record, BaseException), "Ian floor worker failed: %s" % record)
                require(record["id"] not in records, "Duplicate Ian floor id")
                records[record["id"]] = record
                save_json(RESULTS / "ian_floors" / ("row_%02d.json" % record["id"]), record)
    finally:
        if app.app_id:
            app.stop()
            register_paid_app(app.app_id, "Ian floors", 20 * (.0473 + 2 * .008), stopped=True)
    require(set(records) == set(range(20)), "Ian floor result manifest is incomplete")
    ian_floor_gate(records)
    save_json(PREFLIGHT / "ian_floor_gate.json", {"expected": 20, "known_floor_rows": IAN_FLOORS,
              "passed": True, "max_paper_energy_error": max(abs(row["paper_energy_before"] - row["paper_csv_energy"])
                                                             for row in records.values())})
    return records


# Reject edits to the frozen graph identity map before any later stage uses it.
def load_frozen_plan():
    path = PREFLIGHT / "frozen_plan.json"
    lock = PREFLIGHT / "frozen_plan_lock.json"
    require(path.exists() and lock.exists(), "The frozen plan or its hash lock is missing")
    expected = json.loads(lock.read_text())["sha256"]
    require(sha256(path) == expected, "Frozen dataset counts or budget changed after the plan was locked")
    for mode in ["full", "test"]:
        launch = PREFLIGHT / (mode + "_corpus_launch.json")
        if launch.exists():
            require(json.loads(launch.read_text())["config_hash"] == expected, "A corpus launch uses a different frozen plan")
    config = json.loads(path.read_text())
    require(config["input_hashes"] == EXPECTED_HASHES, "Frozen plan names different scientific source bytes")
    return config


# Shrink once using complete Modal timings and preserve the resulting graph identity map.
def freeze_plan():
    import modal
    path = PREFLIGHT / "frozen_plan.json"
    if path.exists():
        return load_frozen_plan()
    require(not list((RESULTS / "dataset").glob("shard_*.jsonl")), "Cannot shrink after a full shard exists")
    dataset, run_filter = scientific_modules()
    dataset.setup("pilot")
    expected = expected_manifest(dataset.selected("pilot"))
    pilot = load_records(RESULTS / "dataset_pilot", expected)
    require(set(pilot) == set(expected), "Plan freezing needs every pilot timing")
    means = {n: math.fsum(pilot[i]["secs_main"] for i, size in expected.items() if size == n) / 40
             for n in sorted(set(expected.values()))}
    timing = json.loads((PREFLIGHT / "test_timings.json").read_text())
    test_means = {int(n): value for n, value in timing["mean_seconds"].items()}
    stopping = json.loads((PREFLIGHT / "stopping_rule.json").read_text())
    multiplier = 2.3 if stopping["stall"] == 400 else 1.0
    projection_means = dict(means)
    measured400 = None
    if stopping["stall"] == 400:
        comparison = load_records(RESULTS / "dataset_pilot400", expected)
        require(set(comparison) == set(expected), "STALL=400 budgeting needs every comparison timing")
        measured400 = {n: math.fsum(comparison[i]["secs_main"] for i, size in expected.items() if size == n) / 40
                       for n in means}
        projection_means = {n: max(2.3 * means[n], measured400[n]) for n in means}
    spending = reconciled_spend_reservation()
    spent_bound = spending["spend_reservation_usd"]
    rate = CPU_SECOND_USD + GIB_SECOND_USD
    original = dict(dataset.PLAN_COUNTS)
    test_counts = dict(dataset.TEST_COUNTS)
    raw_full = math.fsum(original[n] * projection_means[n] for n in original) * 1.05 * rate
    raw_test = math.fsum(test_counts[n] * test_means[n] for n in test_counts) * 1.05 * rate * multiplier
    factor = 1.0
    shrinking = spent_bound + raw_full + raw_test + TRAIN_EVAL_RESERVE_USD > SPEND_STOP_USD
    shrinking = shrinking or spent_bound + FULL_RESOURCE_HEADROOM * raw_full + TEST_RESOURCE_HEADROOM * raw_test + TRAIN_EVAL_RESERVE_USD > HARD_CAP_USD
    if shrinking:
        if raw_test > 30:
            test_counts = {13: 150, 14: 150}
        raw_test = math.fsum(test_counts[n] * test_means[n] for n in test_counts) * 1.05 * rate * multiplier
        available = SPEND_STOP_USD - spent_bound - TEST_RESOURCE_HEADROOM * raw_test - TRAIN_EVAL_RESERVE_USD
        factor = max(1000 / original[12], min(1.0, available / (FULL_RESOURCE_HEADROOM * raw_full)))
    counts = {n: max(1, int(math.floor(count * factor + 1e-9))) for n, count in original.items()}
    require(counts[12] >= 1000, "Uniform shrink violates the n=12 minimum")
    full_cost = math.fsum(counts[n] * projection_means[n] for n in counts) * 1.05 * rate
    full_reserve, test_reserve = full_cost * FULL_RESOURCE_HEADROOM, raw_test * TEST_RESOURCE_HEADROOM
    total = budget_gate(spent_bound, full_reserve + test_reserve, TRAIN_EVAL_RESERVE_USD)
    config = {"input_hashes": EXPECTED_HASHES, "plan_counts": counts, "test_counts": test_counts,
              "stall": stopping["stall"], "uniform_factor": factor, "shrunk": shrinking,
              "folders": {"full": "dataset", "test": "dataset_test" if test_counts[13] == 300 else "dataset_test_reduced"},
              "pilot_mean_seconds": means, "test_mean_seconds": test_means,
              "pilot400_mean_seconds": measured400, "projected_main_seconds": projection_means,
              "cost": {"prior_spend_reservation_usd": spent_bound, "full_projection_usd": full_cost,
                       "test_projection_usd": raw_test, "full_reservation_usd": full_reserve,
                       "test_reservation_usd": test_reserve, "training_evaluation_reserve_usd": TRAIN_EVAL_RESERVE_USD,
                       "projected_total_usd": total, "stall400_multiplier": multiplier,
                       "full_idle_headroom": FULL_RESOURCE_HEADROOM, "test_idle_headroom": TEST_RESOURCE_HEADROOM},
              "created_epoch": time.time()}
    volume = modal.Volume.from_name(VOLUME_NAME)
    try:
        existing = list(volume.iterdir("dataset", recursive=False))
    except (FileNotFoundError, modal.exception.NotFoundError):
        existing = []
    require(not any(entry.path.endswith(".jsonl") for entry in existing), "Remote full shards already exist; cannot shrink")
    save_json(path, config)
    lock_path = PREFLIGHT / "frozen_plan_lock.json"
    save_json(lock_path, {"sha256": sha256(path), "profile": MODAL_PROFILE,
                         "created_epoch": time.time(), "plan_counts": counts, "test_counts": test_counts})
    with volume.batch_upload() as batch:
        batch.put_file(path, "preflight/frozen_plan.json")
        batch.put_file(lock_path, "preflight/frozen_plan_lock.json")
    with (RESULTS / "cost_ledger.md").open("a", encoding="utf-8") as handle:
        handle.write("\nFrozen dataset factor %.10f: %d full graphs; test counts %s. Raw full/test projections $%.6f/$%.6f. "
                     "With idle headroom, prior reservations and $45 training/evaluation, total $%.6f. Counts will not change.\n"
                     % (factor, sum(counts.values()), test_counts, full_cost, raw_test, total))
    return config


# Persist one call per shard owner and retry transport reads without submitting duplicate work.
def collect_corpus_calls(function, jobs, kwargs, calls_path, results_path, existing_calls=None):
    calls, results = [], []
    if existing_calls is not None:
        import modal
        require(len(existing_calls) == len(jobs)
                and {item["batch"] for item in existing_calls} == {job["batch"] for job in jobs}
                and len({item["call_id"] for item in existing_calls}) == len(jobs),
                "Reattachment requires every distinct original corpus call")
        by_batch = {item["batch"]: item for item in existing_calls}
        calls = [dict(by_batch[job["batch"]], handle=modal.FunctionCall.from_id(by_batch[job["batch"]]["call_id"]))
                 for job in jobs]
    else:
        for job in jobs:
            call = function.spawn(job, **kwargs)
            calls.append({"batch": job["batch"], "call_id": call.object_id, "handle": call})
            save_json(calls_path, [{key: value for key, value in item.items() if key != "handle"} for item in calls])
    for job, call in zip(jobs, calls):
        for attempt in range(COLLECTION_RETRIES):
            try:
                result = call["handle"].get()
                break
            except (ConnectionError, TimeoutError):
                if attempt + 1 == COLLECTION_RETRIES:
                    raise
                LOG.warning("Retrying transport read for existing call %s", call["call_id"])
                time.sleep(COLLECTION_RETRY_SLEEP_SECONDS)
        require(isinstance(result, dict) and result.get("batch") == job["batch"]
                and result.get("config_hash") == kwargs["config_hash"], "Corpus completion belongs to a different job")
        completed = [item["shard"] for item in result.get("finished", [])]
        require(len(completed) == len(set(completed)) and set(completed) == set(job["shards"]),
                "Corpus job did not return every assigned shard completion")
        results.append(result)
        save_json(results_path, results)
        LOG.info("%s: %d/%d shard batches done", kwargs["mode"], len(results), len(jobs))
    require(len(results) == len(jobs) and {item["batch"] for item in results} == {job["batch"] for job in jobs},
            "The corpus returned incomplete job coverage; generation must not advance")
    return results


# Launch a frozen full or held-out corpus with small shards and an absolute spending limit.
def launch_sharded_mode(mode, resume=False):
    import modal
    require(mode in ["full", "test"], "Unsupported corpus mode")
    launch_path = PREFLIGHT / (mode + "_corpus_launch.json")
    previous = read_evidence(launch_path)
    require(previous is None or resume, "Corpus launch already exists; reconcile before resubmission")
    require(previous is not None or not resume, "No previous corpus launch exists to resume")
    verify_workspace()
    config = load_frozen_plan()
    dataset = configured_dataset(mode, config)
    expected = expected_manifest(dataset.selected(mode))
    n_shards = FULL_SHARDS if mode == "full" else len(expected)
    containers = FULL_CONTAINERS if mode == "full" else 10
    reserve = config["cost"]["full_reservation_usd" if mode == "full" else "test_reservation_usd"]
    queue_name, resource_queue, quantum, tokens = None, None, None, None
    overhead = min(2.0, reserve * .1)
    rate = BATCH_CORES * CPU_SECOND_USD + BATCH_MEMORY_MIB / 1024 * GIB_SECOND_USD
    if previous is not None:
        registered = read_evidence(PREFLIGHT / "paid_apps" / (previous["app_id"] + ".json"))
        require(registered is not None and registered.get("stopped_epoch") is not None,
                "Stop and reconcile the previous app before resuming a corpus")
        require(not (PREFLIGHT / (mode + "_corpus_csv_gate.json")).exists(), "A completed corpus must not be relaunched")
        require({int(key): value for key, value in previous["expected"].items()} == expected,
                "Resume would change the frozen graph identities")
        queue_name, quantum = previous["budget_queue"], previous["token_seconds"]
        resource_queue = modal.Queue.from_name(queue_name)
        tokens = resource_queue.len()
        require(tokens > containers, "The remaining corpus resource reservation cannot fund a resume")
        old_bound = (previous["initial_tokens"] - tokens) * quantum * rate + previous["startup_shutdown_reserve_usd"]
        register_paid_app(previous["app_id"], "corpus " + mode, containers * rate * 3600,
                          budget_cap_usd=min(previous["reserve_usd"], old_bound))
        reserve = tokens * quantum * rate + overhead
    spending = reconciled_spend_reservation()
    pending = 0.0
    for other_mode in ["full", "test"]:
        name = "full_reservation_usd" if other_mode == "full" else "test_reservation_usd"
        launched = PREFLIGHT / (other_mode + "_corpus_launch.json")
        completed = PREFLIGHT / (other_mode + "_corpus_coverage.json")
        if other_mode == mode:
            pending += reserve
        elif not launched.exists():
            pending += config["cost"][name]
        elif not completed.exists():
            info = json.loads(launched.read_text())
            app_id = info["app_id"]
            observed = math.fsum(float(row["cost_usd"]) for row in spending["billing_rows"] if row["app_id"] == app_id)
            observed += math.fsum(row["resource_upper_bound_usd"] for row in spending["unreported_intervals"]
                                  if row["app_id"] == app_id)
            pending += max(0.0, info["reserve_usd"] - observed)
    budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"]
                - spending["reported_app_cost_usd"] + pending, TRAIN_EVAL_RESERVE_USD)
    if resource_queue is None:
        quantum = max(60, 60 * math.ceil((reserve - overhead) / (RESOURCE_TOKEN_TARGET * rate * 60)))
        tokens = int(math.floor((reserve - overhead) / (quantum * rate)))
        require(containers < tokens <= 5000, "Resource token allocation is too small or exceeds queue capacity")
        queue_name = "qaoa-budget-%s-%d" % (mode, int(time.time()))
        resource_queue = modal.Queue.from_name(queue_name, create_if_missing=True)
        resource_queue.put_many([quantum] * tokens, block=False)
    if previous is not None:
        archive = PREFLIGHT / "runner_repairs" / (mode + "_attempt_" + previous["app_id"])
        require(archive.resolve().is_relative_to(ROOT.resolve()), "Attempt archive escapes the workspace")
        archive.mkdir(parents=True, exist_ok=True)
        launch_path.rename(archive / "launch.json")
        for name in [mode + "_corpus_batches.json", mode + "_corpus_calls.json", "stop.json", "execution_error.json", "continuation_stop.json"]:
            path = PREFLIGHT / name
            if path.exists():
                path.rename(archive / path.name)
        save_json(archive / "resume_budget.json", {"retained_queue": queue_name, "remaining_tokens": tokens,
                  "new_app_reserve_usd": reserve, "prior_app_used_reservation_usd": old_bound,
                  "frozen_config_sha256": sha256(PREFLIGHT / "frozen_plan.json"), "counts_unchanged": True})
        with (RESULTS / "cost_ledger.md").open("a", encoding="utf-8") as handle:
            handle.write("\nCorpus %s resume retains %d unused resource tokens from %s; quantum %d seconds. New app reservation $%.8f, including $%.2f startup/shutdown overhead. The prior attempt retains its used-token reservation $%.8f. Frozen counts and input hashes are unchanged.\n" % (mode, tokens, previous["app_id"], quantum, reserve, overhead, old_bound))
    deadline = time.time() + FULL_TIMEOUT
    jobs = [{"batch": k, "shards": list(range(k, n_shards, containers)), "n_shards": n_shards,
             "deadline_epoch": deadline, "budget_queue": queue_name} for k in range(containers)]
    assigned = [shard for job in jobs for shard in job["shards"]]
    require(len(assigned) == len(set(assigned)) == n_shards, "Corpus shard ownership overlaps or omits a shard")
    upload = {name: sha256(ROOT / name) for name in UPLOAD_FILES}
    volume = modal.Volume.from_name(VOLUME_NAME)
    config_hash = sha256(PREFLIGHT / "frozen_plan.json")
    app = modal.App(APP_NAME + "-" + mode)
    function = app.function(image=data_image(), cpu=(BATCH_CORES, BATCH_CORES),
                            memory=(BATCH_MEMORY_MIB, BATCH_MEMORY_MIB), timeout=FULL_TIMEOUT, retries=1,
                            max_containers=containers, scaledown_window=2,
                            volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                            name="corpus_batch")(remote_shard_batch)
    batches = []
    try:
        with app.run(detach=True):
            register_paid_app(app.app_id, "corpus " + mode, containers * rate * 3600, budget_cap_usd=reserve)
            save_json(launch_path, {"app_id": app.app_id, "expected": expected,
                      "jobs": jobs, "reserve_usd": reserve, "config_hash": config_hash,
                      "upload_hashes": upload, "created_epoch": time.time(), "deadline_epoch": deadline,
                      "budget_queue": queue_name, "token_seconds": quantum, "initial_tokens": tokens,
                      "token_rate_usd_per_second": rate, "startup_shutdown_reserve_usd": overhead,
                      "resumed_from_app_id": previous["app_id"] if previous else None})
            batches = collect_corpus_calls(function, jobs,
                {"mode": mode, "config_hash": config_hash, "volume_name": VOLUME_NAME, "upload_hashes": upload},
                PREFLIGHT / (mode + "_corpus_calls.json"), PREFLIGHT / (mode + "_corpus_batches.json"))
    finally:
        if app.app_id:
            app.stop()
            register_paid_app(app.app_id, "corpus " + mode, containers * rate * 3600, stopped=True, budget_cap_usd=reserve)
    remaining_tokens = resource_queue.len()
    used_bound = (tokens - remaining_tokens) * quantum * rate + overhead
    register_paid_app(app.app_id, "corpus " + mode, containers * rate * 3600, stopped=True,
                      budget_cap_usd=min(reserve, used_bound))
    save_json(PREFLIGHT / (mode + "_resource_usage_bound.json"), {"initial_tokens": tokens,
              "remaining_tokens": remaining_tokens, "seconds_per_token": quantum,
              "compute_and_overhead_upper_bound_usd": used_bound})
    if mode == "test" and config["folders"][mode] != "dataset_test":
        original, archive = RESULTS / "dataset_test", RESULTS / "dataset_test_timing"
        require(original.resolve().is_relative_to(ROOT.resolve()) and archive.resolve().is_relative_to(ROOT.resolve()),
                "Test timing archive path escapes the workspace")
        if original.exists() and not archive.exists():
            original.rename(archive)
    download_volume(config["folders"][mode], RESULTS / ("dataset" if mode == "full" else "dataset_test"))
    folder = RESULTS / ("dataset" if mode == "full" else "dataset_test")
    records = load_records(folder, expected)
    coverage = dataset_gate(expected, records)
    save_json(PREFLIGHT / (mode + "_corpus_coverage.json"), coverage)
    dataset.OUT_DIR, dataset.MERGED = folder, folder / "dataset.csv"
    dataset.merge()
    csv = csv_gate(dataset.MERGED, [i for i, row in records.items() if has_pick(row, expected[i])])
    save_json(PREFLIGHT / (mode + "_corpus_csv_gate.json"), csv)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(dataset.MERGED, config["folders"][mode] + "/dataset.csv")
    return coverage


# Complete the user-approved test corpus, first proving returned batch completions on twenty new graphs.
def launch_test_completion(smoke=False):
    import modal
    import numpy as np
    load_training_population()
    verify_workspace()
    config = load_frozen_plan()
    dataset = configured_dataset("test", config)
    expected = expected_manifest(dataset.selected("test"))
    folder = RESULTS / "dataset_test"
    records = load_records(folder, expected)
    name = "test_smoke" if smoke else "test_completion"
    gate_path = PREFLIGHT / (name + "_gate.json")
    if gate_path.exists():
        return read_evidence(gate_path)
    launch_path = PREFLIGHT / (name + "_launch.json")
    previous = read_evidence(launch_path)
    if previous:
        registration = read_evidence(PREFLIGHT / "paid_apps" / (previous["app_id"] + ".json"))
        if not smoke and registration is not None and registration.get("stopped_epoch") is None:
            return reattach_test_completion(previous, config, dataset, expected)
        require(smoke and registration is not None and registration.get("stopped_epoch") is not None,
                "Reconcile existing test calls before resubmission")
        statistics = read_evidence(RESULTS / "gnn" / "label_statistics_preview.json")
        require(statistics is not None and statistics["box_violations"] == 0
                and (PREFLIGHT / "label_box_units_diagnosis.json").exists(), "Correct and verify the interrupted label gate first")
        archive = PREFLIGHT / "runner_repairs" / (name + "_" + previous["app_id"])
        archive.mkdir(parents=True, exist_ok=True)
        for suffix in ["launch", "calls", "batches"]:
            source = PREFLIGHT / (name + "_" + suffix + ".json")
            if source.exists():
                source.rename(archive / source.name)
    missing = sorted(set(expected) - set(records))
    if smoke:
        if previous:
            chosen = previous["expected_ids"]
            require(len(chosen) == 20 and set(records) <= set(chosen) | set(range(5)) | set(range(300, 305)),
                    "Interrupted smoke test contains unexpected ids")
        else:
            require(len(missing) == 590, "The smoke test must start from the ten retained timing graphs")
            rng = np.random.default_rng(42)
            chosen = sorted(int(i) for n in [13, 14] for i in rng.choice(
                [i for i in missing if expected[i] == n], 10, replace=False))
        containers, reserve = 3, 2.0
    else:
        proof = read_evidence(PREFLIGHT / "test_smoke_gate.json")
        require(proof is not None and proof["passed"] and proof["returned_batches"] == proof["expected_batches"],
                "The twenty-graph completion test has not passed")
        chosen = missing
        require(len(chosen) == 570, "Expected exactly 570 graphs after the twenty-graph smoke test")
        containers, reserve = 10, config["cost"]["test_reservation_usd"]
    spending = reconciled_spend_reservation()
    budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"],
                reserve + TRAIN_EVAL_RESERVE_USD)
    rate = BATCH_CORES * CPU_SECOND_USD + BATCH_MEMORY_MIB / 1024 * GIB_SECOND_USD
    overhead = min(.2, reserve * .1)
    quantum = 120
    tokens = int((reserve - overhead) / (quantum * rate))
    require(containers < tokens <= 5000, "Invalid test resource-token count")
    queue_name = "qaoa-budget-%s-%d" % (name, int(time.time()))
    queue = modal.Queue.from_name(queue_name, create_if_missing=True)
    queue.put_many([quantum] * tokens, block=False)
    deadline = time.time() + FULL_TIMEOUT
    jobs = [{"batch": k, "shards": chosen[k::containers], "n_shards": len(expected),
             "deadline_epoch": deadline, "budget_queue": queue_name} for k in range(containers)]
    require(sorted(shard for job in jobs for shard in job["shards"]) == chosen, "Test shard assignment differs")
    upload = {relative: sha256(ROOT / relative) for relative in UPLOAD_FILES}
    volume = modal.Volume.from_name(VOLUME_NAME)
    config_hash = sha256(PREFLIGHT / "frozen_plan.json")
    app = modal.App(APP_NAME + "-" + name)
    function = app.function(image=data_image(), cpu=(8, 8), memory=(8192, 8192), timeout=FULL_TIMEOUT,
                            retries=1, max_containers=containers, scaledown_window=2,
                            volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                            name="corpus_batch")(remote_shard_batch)
    try:
        with app.run(detach=True):
            register_paid_app(app.app_id, name, containers * rate * 3600, budget_cap_usd=reserve)
            save_json(launch_path, {"app_id": app.app_id, "expected_ids": chosen, "jobs": jobs,
                      "reserve_usd": reserve, "upload_hashes": upload, "config_hash": config_hash,
                      "created_epoch": time.time(), "budget_queue": queue_name, "initial_tokens": tokens,
                      "token_seconds": quantum, "startup_shutdown_reserve_usd": overhead})
            batches = collect_corpus_calls(function, jobs,
                {"mode": "test", "config_hash": config_hash, "volume_name": VOLUME_NAME, "upload_hashes": upload},
                PREFLIGHT / (name + "_calls.json"), PREFLIGHT / (name + "_batches.json"))
    finally:
        if app.app_id:
            app.stop()
            register_paid_app(app.app_id, name, containers * rate * 3600, stopped=True, budget_cap_usd=reserve)
    remaining = queue.len()
    used_bound = (tokens - remaining) * quantum * rate + overhead
    register_paid_app(app.app_id, name, containers * rate * 3600, stopped=True, budget_cap_usd=min(reserve, used_bound))
    download_volume(config["folders"]["test"], folder)
    records = load_records(folder, expected)
    require(set(chosen) <= set(records), "Returned test batches are missing saved records")
    proof = {"passed": True, "expected_ids": chosen, "expected_batches": len(jobs), "returned_batches": len(batches),
             "new_graphs": len(chosen), "saved_total": len(records), "compute_upper_bound_usd": used_bound,
             "source_sha256": upload["src/modal_app.py"],
             "completion_host_source_sha256": sha256(ROOT / "src" / "modal_app.py"), "app_id": app.app_id}
    if not smoke:
        require(set(records) == set(expected), "The larger-test corpus must contain all 600 expected ids")
        save_json(PREFLIGHT / "test_corpus_coverage.json", dataset_gate(expected, records))
        dataset.OUT_DIR, dataset.MERGED = folder, folder / "dataset.csv"
        dataset.merge()
        csv = csv_gate(dataset.MERGED, [i for i, row in records.items() if has_pick(row, expected[i])])
        save_json(PREFLIGHT / "test_corpus_csv_gate.json", csv)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(dataset.MERGED, config["folders"]["test"] + "/dataset.csv")
    save_json(gate_path, proof)
    write_billing_summary()
    return proof


# Refresh the local driver by reading the same ten paid calls, without submitting any new graphs.
def reattach_test_completion(launch, config, dataset, expected):
    import modal
    require(launch["config_hash"] == sha256(PREFLIGHT / "frozen_plan.json"), "Reattached test identities differ")
    jobs = launch["jobs"]
    chosen = launch["expected_ids"]
    require(len(chosen) == 570 and sorted(shard for job in jobs for shard in job["shards"]) == chosen,
            "Reattached test ownership differs")
    for relative, digest in EXPECTED_HASHES.items():
        require(launch["upload_hashes"][relative] == digest, "Reattached workers use unapproved input bytes")
    calls = read_evidence(PREFLIGHT / "test_completion_calls.json")
    require(calls is not None, "The original test call manifest is missing")
    rate = BATCH_CORES * CPU_SECOND_USD + BATCH_MEMORY_MIB / 1024 * GIB_SECOND_USD
    try:
        batches = collect_corpus_calls(None, jobs, {"mode": "test", "config_hash": launch["config_hash"]},
            PREFLIGHT / "test_completion_calls.json", PREFLIGHT / "test_completion_batches.json", existing_calls=calls)
    finally:
        stopped = subprocess.run([sys.executable, "-m", "modal", "app", "stop", launch["app_id"],
                                  "--profile", MODAL_PROFILE, "--yes"], capture_output=True, text=True)
        require(stopped.returncode == 0, "Could not stop the reattached test app")
        register_paid_app(launch["app_id"], "test_completion", len(jobs) * rate * 3600,
                          stopped=True, budget_cap_usd=launch["reserve_usd"])
    remaining = modal.Queue.from_name(launch["budget_queue"]).len()
    used_bound = ((launch["initial_tokens"] - remaining) * launch["token_seconds"] * rate
                  + launch["startup_shutdown_reserve_usd"])
    require(used_bound <= launch["reserve_usd"] + 1e-8, "Reattached test exceeded its original resource cap")
    register_paid_app(launch["app_id"], "test_completion", len(jobs) * rate * 3600,
                      stopped=True, budget_cap_usd=used_bound)
    folder = RESULTS / "dataset_test"
    download_volume(config["folders"]["test"], folder)
    records = load_records(folder, expected)
    require(set(records) == set(expected), "The larger-test corpus must contain all 600 expected ids")
    save_json(PREFLIGHT / "test_corpus_coverage.json", dataset_gate(expected, records))
    dataset.OUT_DIR, dataset.MERGED = folder, folder / "dataset.csv"
    dataset.merge()
    save_json(PREFLIGHT / "test_corpus_csv_gate.json",
              csv_gate(dataset.MERGED, [i for i, row in records.items() if has_pick(row, expected[i])]))
    with modal.Volume.from_name(VOLUME_NAME).batch_upload(force=True) as batch:
        batch.put_file(dataset.MERGED, config["folders"]["test"] + "/dataset.csv")
    proof = {"passed": True, "expected_ids": chosen, "expected_batches": len(jobs),
             "returned_batches": len(batches), "new_graphs": len(chosen), "saved_total": len(records),
             "compute_upper_bound_usd": used_bound, "source_sha256": launch["upload_hashes"]["src/modal_app.py"],
             "completion_host_source_sha256": sha256(ROOT / "src" / "modal_app.py"),
             "app_id": launch["app_id"], "reattached_without_new_submissions": True}
    save_json(PREFLIGHT / "test_completion_gate.json", proof)
    write_billing_summary()
    return proof


# Read bounded snapshots without confusing earlier test-worker files with the active launch.
def test_completion_progress():
    import modal
    from concurrent.futures import ThreadPoolExecutor
    launch = read_evidence(PREFLIGHT / "test_completion_launch.json")
    require(launch is not None, "The remaining-test launch is missing")
    volume = modal.Volume.from_name(VOLUME_NAME)

    # Prefer a matching complete result and otherwise accept only this attempt's progress.
    def read_job(job):
        prefix = "worker_proofs/test/batch_%04d/" % job["batch"]
        for name in ["result.json", "progress.json"]:
            try:
                value = json.loads(b"".join(volume.read_file(prefix + name)).decode("utf-8"))
            except (FileNotFoundError, modal.exception.NotFoundError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if name == "result.json":
                finished = value.get("finished", [])
                if value.get("config_hash") != launch["config_hash"] or {item["shard"] for item in finished} != set(job["shards"]):
                    continue
                return {"batch": job["batch"], "complete": True,
                        "records": sum(item["records"] for item in finished),
                        "usable": sum(item["usable"] for item in finished)}
            if value.get("captured_epoch", 0) >= launch["created_epoch"] and value.get("expected_shards") == len(job["shards"]):
                return {"batch": job["batch"], "complete": False, "records": value["completed_records"],
                        "usable": value["usable_records"]}
        return {"batch": job["batch"], "complete": False, "records": 0, "usable": 0, "unknown": True}

    with ThreadPoolExecutor(max_workers=10) as pool:
        batches = list(pool.map(read_job, launch["jobs"]))
    count = sum(item["records"] for item in batches)
    require(0 <= count <= len(launch["expected_ids"]), "Test progress exceeds its assigned population")
    result = {"new_saved": count, "new_expected": len(launch["expected_ids"]),
              "retained_before_launch": 30, "saved_total_lower_bound": 30 + count, "expected_total": 600,
              "batches": batches, "app_id": launch["app_id"], "captured_epoch": time.time()}
    save_json(PREFLIGHT / "test_completion_progress.json", result)
    return result


# Verify labels and exact cut denominators before building splits or training features.
def prepare_training_artifacts():
    import numpy as np
    import gnn
    import modal
    require((PREFLIGHT / "ian_floor_gate.json").exists(), "Ian floor gate has not passed")
    for mode in ["full", "test"]:
        require((PREFLIGHT / (mode + "_corpus_csv_gate.json")).exists(), "A dataset CSV gate has not passed")
    config = load_frozen_plan()
    corpora, expectations = {}, {}
    for mode, tag in [("full", "full"), ("test", "large")]:
        dataset = configured_dataset(mode, config)
        expected = expected_manifest(dataset.selected(mode))
        folder = RESULTS / ("dataset" if mode == "full" else "dataset_test")
        records = load_records(folder, expected)
        dataset_gate(expected, records, saved_population=mode == "full")
        metadata = expected_graph_metadata(mode, config)
        require(all(row["type"] == metadata[key]["type"] and
                    ("audit" not in row or row["audit"] == metadata[key]["audit"])
                    for key, row in records.items()), "Dataset families or audit flags differ from the expected manifest")
        if mode == "full":
            population = load_training_population()
            require(set(population["ids"]) == {i for i, row in records.items() if usable(row, expected[i])},
                    "Usable ids differ from the approved training population")
            for i in population["ids"]:
                require(hashlib.sha256(json.dumps(records[i], sort_keys=True).encode()).hexdigest()
                        == population["row_hashes"][str(i)], "A frozen training record changed")
            corpora[tag] = [records[i] for i in population["ids"]]
        else:
            require(set(records) == set(expected), "Finish all 600 larger-test graphs before training")
            corpora[tag] = [records[i] for i in sorted(records) if usable(records[i], expected[i])]
        expectations[tag] = expected
    corpora["ian"] = [json.loads((RESULTS / "ian_floors" / ("row_%02d.json" % i)).read_text()) for i in range(20)]
    ian_floor_gate({row["id"]: row for row in corpora["ian"]})
    energy_error, cut_error = 0.0, 0.0
    dataset, run_filter = scientific_modules()
    checked = 0
    for tag, rows in corpora.items():
        for row in rows:
            energy, unused_batch, gradient, dimension = run_filter.make_energy(row["n"], row["edges"], row["w_sorted"], 1)
            error = abs(float(energy(np.asarray(row["pick"]))) - row["floor"])
            energy_error = max(energy_error, error)
            require(error <= ENERGY_TOL, "A usable label does not reproduce its floor: %s/%d" % (tag, row["id"]))
            error = abs(gnn.independent_max_cut(row) - row["max_cut"])
            cut_error = max(cut_error, error)
            require(error <= DATASET_ENERGY_TOL, "Independent exact Max-Cut denominator differs")
            checked += 1
            if checked % 5000 == 0:
                LOG.info("Stage 3: %d label energies and exact cut values independently verified", checked)
    statistics = gnn.label_statistics(corpora["full"])
    require(statistics["box_violations"] == 0, "A training-size label lies outside the specified box")
    large_statistics = gnn.label_statistics(corpora["large"])
    require(large_statistics["box_violations"] == 0, "A larger-test label lies outside the specified box")
    save_json(RESULTS / "gnn" / "large_label_statistics.json", large_statistics)
    holdout_hashes = set()
    for row in corpora["ian"]:
        graph = dataset.nx.Graph(row["edges"])
        graph.add_nodes_from(range(row["n"]))
        holdout_hashes.add(dataset.nx.weisfeiler_lehman_graph_hash(graph))
    split = gnn.grouped_split(corpora["full"], holdout_hashes=holdout_hashes)
    train_ids = set(split["ids"]["train"])
    theta_classes = gnn.classification_values([row for row in corpora["full"] if row["id"] in train_ids], statistics)
    save_json(RESULTS / "gnn" / "label_statistics.json", statistics)
    save_json(RESULTS / "gnn" / "splits.json", split)
    save_json(RESULTS / "gnn" / "classification_values.json", {"theta": theta_classes,
              "beta": (np.arange(-2, 3) * np.pi / 8).tolist(), "centroid_source": "training labels only"})
    manifest = {"corpora": {}, "feature_implementation_hash": gnn.feature_implementation_hash(),
                "training_population_sha256": sha256(PREFLIGHT / "training_population.json"),
                "splits_sha256": sha256(RESULTS / "gnn" / "splits.json"),
                "classification_values_sha256": sha256(RESULTS / "gnn" / "classification_values.json"),
                "verified_graphs": checked, "max_label_floor_error": energy_error,
                "max_independent_cut_error": cut_error, "created_epoch": time.time()}
    uploads = [RESULTS / "gnn" / name for name in ["label_statistics.json", "large_label_statistics.json",
                                                   "splits.json", "classification_values.json"]]
    for tag, rows in corpora.items():
        folder = RESULTS / "gnn" / "corpus"
        folder.mkdir(parents=True, exist_ok=True)
        path, index_path = folder / (tag + ".jsonl"), folder / (tag + "_index.json")
        index = {}
        with path.open("wb") as handle:
            for row in rows:
                data = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
                index[row["id"]] = [handle.tell(), len(data), hashlib.sha256(data).hexdigest()]
                handle.write(data)
        save_json(index_path, index)
        entry = {"rows": len(rows), "data": str(path.relative_to(RESULTS)).replace("\\", "/"),
                 "index": str(index_path.relative_to(RESULTS)).replace("\\", "/"),
                 "data_sha256": sha256(path), "index_sha256": sha256(index_path), "features": {}}
        uploads.extend([path, index_path])
        for baseline, name in [(True, "A"), (False, "B")]:
            feature_path = folder / (tag + "_features_" + name + ".npz")
            gnn.save_feature_cache(rows, baseline, feature_path)
            entry["features"][name] = {"path": str(feature_path.relative_to(RESULTS)).replace("\\", "/"),
                                      "sha256": sha256(feature_path)}
            uploads.append(feature_path)
        manifest["corpora"][tag] = entry
    manifest_path = RESULTS / "gnn" / "corpus_manifest.json"
    save_json(manifest_path, manifest)
    uploads.append(manifest_path)
    volume = modal.Volume.from_name(VOLUME_NAME)
    with volume.batch_upload(force=True) as batch:
        for path in uploads:
            batch.put_file(path, str(path.relative_to(RESULTS)).replace("\\", "/"))
    save_json(PREFLIGHT / "stage3_gate.json", {"passed": True, "verified_graphs": checked,
              "max_label_floor_error": energy_error, "max_independent_cut_error": cut_error,
              "split_counts": split["counts"], "box_violations": 0,
              "large_box_violations": 0,
              "corpus_manifest_sha256": sha256(manifest_path)})
    return manifest


# Read only assigned graph records from a hash-verified numerical corpus index.
def read_corpus_rows(tag, ids=None, manifest=None):
    manifest = json.loads((RESULTS / "gnn" / "corpus_manifest.json").read_text()) if manifest is None else manifest
    entry = manifest["corpora"][tag]
    path, index_path = RESULTS / entry["data"], RESULTS / entry["index"]
    require(sha256(index_path) == entry["index_sha256"], "Corpus index bytes differ")
    index = json.loads(index_path.read_text())
    ids = sorted(int(key) for key in index) if ids is None else list(ids)
    require(len(ids) == len(set(ids)), "Corpus selection contains duplicate ids")
    rows = []
    with path.open("rb") as handle:
        for graph_id in ids:
            require(str(graph_id) in index, "Expected graph is missing from corpus index")
            offset, length, expected_hash = index[str(graph_id)]
            handle.seek(offset)
            data = handle.read(length)
            require(hashlib.sha256(data).hexdigest() == expected_hash, "Corpus graph bytes differ")
            row = json.loads(data)
            require(row["id"] == graph_id, "Corpus offset points at a different graph")
            rows.append(row)
    return rows


# Build the independent GPU environment before copying scientific sources byte for byte.
def gpu_image():
    import modal
    image = modal.Image.debian_slim(python_version="3.13").pip_install(
        *[name + "==" + version for name, version in INPUT_PINS.items()], "torch==2.15.0",
        "threadpoolctl==" + THREADPOOL_PIN)
    image = image.env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
                       "PYTHONPATH": REMOTE_ROOT + "/src"})
    for relative in UPLOAD_FILES:
        image = image.add_local_file(ROOT / relative, REMOTE_ROOT + "/" + relative, copy=True)
    return image


# Run the numerical GPU gate independently, before any model training is allowed.
def remote_gpu_energy_check(volume_name, upload_hashes):
    import sys
    from pathlib import Path
    import modal
    root = Path("/root/qaoa")
    sys.path.insert(0, str(root / "src"))
    import modal_app as runner
    runner.configure_threads()
    runner.PREFLIGHT = runner.RESULTS / "worker_proofs" / "gpu_energy"
    for relative, expected in upload_hashes.items():
        runner.require(runner.sha256(root / relative) == expected, "GPU verification uploaded bytes differ")
    runner.check_input_hashes()
    runner.stage_zero()
    import torch
    import gnn
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner.require(torch.cuda.is_available(), "GPU verification has no CUDA device")
    proof = gnn.check_energy(device="cuda")
    runner.torch_gate(proof["checks"])
    proof.update({"implementation_sha256": runner.sha256(root / "src" / "gnn.py"),
                  "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "passed": True})
    runner.save_json(runner.RESULTS / "gnn" / "torch_energy_check_cuda.json", proof)
    runner.commit_volume(modal.Volume.from_name(volume_name))
    return proof


# Fund only the independent numerical check and persist its call before waiting.
def launch_gpu_energy_check():
    import modal
    verify_workspace()
    path = RESULTS / "gnn" / "torch_energy_check_cuda.json"
    existing = read_evidence(path)
    if existing and existing.get("implementation_sha256") == sha256(ROOT / "src" / "gnn.py"):
        torch_gate(existing["checks"])
        return existing
    launch_path = PREFLIGHT / "gpu_energy_launch.json"
    require(not launch_path.exists(), "Reconcile the existing numerical GPU call before resubmission")
    spending = reconciled_spend_reservation()
    rates = modal.Workspace.from_context().billing.rates()
    require(max(float(rates[key]) for key in ["gpu_hour_cost_l4", "gpu_hour_cost_a10g", "gpu_hour_cost_t4"])
            <= GPU_CHECK_MAX_HOURLY_USD, "Numerical-check GPU prices exceed the costed fallback maximum")
    hourly = GPU_CHECK_MAX_HOURLY_USD + .0473 + .016
    budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"],
                hourly / 4 + TRAIN_EVAL_RESERVE_USD)
    volume = modal.Volume.from_name(VOLUME_NAME)
    upload = {relative: sha256(ROOT / relative) for relative in UPLOAD_FILES}
    app = modal.App(APP_NAME + "-gpu-energy-check")
    function = app.function(image=gpu_image(), gpu=GPU_CHECK_TYPES, cpu=(1, 1), memory=(2048, 2048),
                            timeout=900, retries=0, max_containers=1, scaledown_window=2,
                            volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                            name="gpu_energy_check")(remote_gpu_energy_check)
    try:
        with app.run(detach=True):
            register_paid_app(app.app_id, "GPU energy check", hourly, budget_cap_usd=hourly / 4)
            call = function.spawn(VOLUME_NAME, upload)
            save_json(launch_path, {"app_id": app.app_id, "call_id": call.object_id, "created_epoch": time.time(),
                      "reserve_usd": hourly / 4, "gpu_preferences": GPU_CHECK_TYPES, "upload_hashes": upload})
            proof = call.get()
    finally:
        if app.app_id:
            app.stop()
            register_paid_app(app.app_id, "GPU energy check", hourly, stopped=True, budget_cap_usd=hourly / 4)
    torch_gate(proof["checks"])
    save_json(path, proof)
    write_billing_summary()
    return proof


# Bound an optional A100 numerical check within the existing low-cost GPU allowance.
def launch_bounded_a100_energy_check():
    import modal
    verify_workspace()
    check_input_hashes()
    existing = read_evidence(RESULTS / "gnn" / "torch_energy_check_cuda.json")
    if existing and existing.get("implementation_sha256") == sha256(ROOT / "src" / "gnn.py"):
        torch_gate(existing["checks"])
        return existing
    previous = read_evidence(PREFLIGHT / "gpu_energy_launch.json")
    if previous:
        registered = read_evidence(PREFLIGHT / "paid_apps" / (previous["app_id"] + ".json"))
        require(registered is not None and registered.get("stopped_epoch") is not None,
                "Stop the existing numerical check before an A100 replacement")
    rates = modal.Workspace.from_context().billing.rates()
    require(float(rates["gpu_hour_cost_a100_40gb"]) <= A100_CHECK_GPU_HOURLY_USD, "A100 price exceeds the costed maximum")
    hourly = A100_CHECK_GPU_HOURLY_USD + .0473 + .016
    reserve = hourly * A100_CHECK_RESOURCE_SECONDS / 3600
    require(reserve <= .290825, "A100 numerical-check allowance exceeds the existing low-cost allowance")
    spending = reconciled_spend_reservation()
    budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"],
                reserve + TRAIN_EVAL_RESERVE_USD)
    launch_path = PREFLIGHT / "gpu_a100_energy_launch.json"
    require(not launch_path.exists(), "Reconcile the existing bounded A100 check before resubmission")
    volume = modal.Volume.from_name(VOLUME_NAME)
    upload = {relative: sha256(ROOT / relative) for relative in UPLOAD_FILES}
    app = modal.App(APP_NAME + "-bounded-a100-energy-check")
    function = app.function(image=gpu_image(), gpu="A100-40GB", cpu=(1, 1), memory=(2048, 2048),
                            timeout=A100_CHECK_BODY_SECONDS, startup_timeout=A100_CHECK_STARTUP_SECONDS,
                            retries=0, max_containers=1, scaledown_window=2,
                            volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                            name="gpu_energy_check")(remote_gpu_energy_check)
    try:
        with app.run(detach=True):
            register_paid_app(app.app_id, "bounded A100 numerical check", hourly, budget_cap_usd=reserve)
            call = function.spawn(VOLUME_NAME, upload)
            save_json(launch_path, {"app_id": app.app_id, "call_id": call.object_id, "created_epoch": time.time(),
                      "reserve_usd": reserve, "gpu": "A100-40GB", "upload_hashes": upload,
                      "startup_timeout_seconds": A100_CHECK_STARTUP_SECONDS, "body_timeout_seconds": A100_CHECK_BODY_SECONDS,
                      "call_wait_timeout_seconds": A100_CHECK_WAIT_SECONDS, "resource_allowance_seconds": A100_CHECK_RESOURCE_SECONDS,
                      "model_training": False})
            proof = call.get(timeout=A100_CHECK_WAIT_SECONDS)
    finally:
        if app.app_id:
            app.stop()
            register_paid_app(app.app_id, "bounded A100 numerical check", hourly, stopped=True, budget_cap_usd=reserve)
    torch_gate(proof["checks"])
    save_json(RESULTS / "gnn" / "torch_energy_check_cuda.json", proof)
    write_billing_summary()
    return proof


# Train or resume one bounded GPU block and persist predictions without CPU optimizer work.
def remote_train_block(job, volume_name, upload_hashes, manifest_hash):
    import sys
    import time
    from pathlib import Path
    import modal
    root = Path("/root/qaoa")
    sys.path.insert(0, str(root / "src"))
    import modal_app as runner
    runner.configure_threads()
    runner.PREFLIGHT = runner.RESULTS / "worker_proofs" / "training" / job["name"]
    for relative, expected in upload_hashes.items():
        runner.require(runner.sha256(root / relative) == expected, "GPU worker uploaded bytes differ")
    runner.check_input_hashes()
    runner.stage_zero()
    runner.require(runner.sha256(runner.RESULTS / "gnn" / "corpus_manifest.json") == manifest_hash,
                   "GPU corpus manifest differs")
    import torch
    import gnn
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner.require(torch.cuda.is_available(), "The training worker has no CUDA device")
    volume = modal.Volume.from_name(volume_name)
    proof_path = runner.RESULTS / "gnn" / "torch_energy_check_cuda.json"
    source_hash = runner.sha256(root / "src" / "gnn.py")
    proof = runner.json.loads(proof_path.read_text()) if proof_path.exists() else {}
    if proof.get("implementation_sha256") != source_hash:
        proof = gnn.check_energy(device="cuda")
        proof.update({"implementation_sha256": source_hash, "gpu": torch.cuda.get_device_name(),
                      "cuda": torch.version.cuda})
        runner.save_json(proof_path, proof)
        runner.commit_volume(volume)
    runner.torch_gate(proof["checks"])
    manifest = runner.json.loads((runner.RESULTS / "gnn" / "corpus_manifest.json").read_text())
    runner.require(runner.sha256(runner.RESULTS / "gnn" / "splits.json") == manifest["splits_sha256"],
                   "Training split ids differ from the fixed manifest")
    runner.require(runner.sha256(runner.RESULTS / "gnn" / "classification_values.json") == manifest["classification_values_sha256"],
                   "Classification values differ from the training-only definitions")
    splits = runner.json.loads((runner.RESULTS / "gnn" / "splits.json").read_text())
    config = job["config"]
    name = "A" if config.get("baseline", False) else "B"
    cache_info = manifest["corpora"]["full"]["features"][name]
    cache_path = runner.RESULTS / cache_info["path"]
    runner.require(runner.sha256(cache_path) == cache_info["sha256"], "Training feature cache bytes differ")
    cache = gnn.load_feature_cache(cache_path)
    train_ids = job.get("train_ids", splits["ids"]["train"])
    runner.require(bool(train_ids) and len(train_ids) == len(set(train_ids))
                   and set(train_ids) <= set(splits["ids"]["train"]), "Training subset is outside the frozen training split")
    train_ids_hash = runner.hashlib.sha256(runner.json.dumps(train_ids).encode()).hexdigest()
    runner.require(job.get("training_ids_sha256", train_ids_hash) == train_ids_hash, "Training subset hash differs")
    train_rows = runner.read_corpus_rows("full", train_ids, manifest)
    checkpoint_path = runner.RESULTS / "gnn" / "checkpoints" / (job["name"] + ".pt")
    working = checkpoint_path.with_suffix(".working.pt")
    identity = runner.hashlib.sha256(runner.json.dumps({"config": config, "target": job["target_epoch"],
                 "corpus_manifest": manifest_hash, "source": source_hash, "training_ids": train_ids_hash}, sort_keys=True).encode()).hexdigest()
    checkpoint = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        runner.require(checkpoint.get("job_identity") == identity, "Completed checkpoint uses a different job protocol")
    elif working.exists():
        checkpoint = torch.load(working, map_location="cpu", weights_only=False)
        runner.require(checkpoint.get("job_identity") == identity, "Partial checkpoint uses a different job protocol")
    elif job.get("input_checkpoint"):
        checkpoint = torch.load(runner.RESULTS / job["input_checkpoint"], map_location="cpu", weights_only=False)
    completed = 0 if checkpoint is None else int(checkpoint["epoch"])
    target = int(job["target_epoch"])
    runner.require(completed <= target, "Training checkpoint lies beyond the requested epoch")
    started = time.monotonic()
    model, state = None, checkpoint
    while completed < target:
        runner.require(time.time() <= job["deadline_epoch"], "Training resource reservation expired")
        count = min(runner.VALIDATION_INTERVAL, target - completed)
        model, state = gnn.train_block(config, train_rows, cache, checkpoint=state, epochs=count, device="cuda")
        completed = state["epoch"]
        state["job_name"] = job["name"]
        state["job_identity"] = identity
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = working.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(working)
        runner.commit_volume(volume)
        print("%s: epoch %d/%d loss %.7g" % (job["name"], completed, target, state["history"][-1]["loss"]), flush=True)
    if model is None:
        model, state = gnn.train_block(config, train_rows, cache, checkpoint=state, epochs=0, device="cuda")
    state.update({"job_name": job["name"], "job_identity": identity})
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint_path)
    predictions = {}
    for tag, selection in job.get("prediction_sets", {"val": {"corpus": "full", "ids": splits["ids"]["val_selection"]}}).items():
        source = selection["corpus"]
        rows = runner.read_corpus_rows(source, selection["ids"], manifest)
        if source == "full":
            feature_cache = cache
        else:
            info = manifest["corpora"][source]["features"][name]
            feature_path = runner.RESULTS / info["path"]
            runner.require(runner.sha256(feature_path) == info["sha256"], "Prediction features differ")
            feature_cache = gnn.load_feature_cache(feature_path)
        records = gnn.predict_rows(model, rows, feature_cache, device="cuda")
        path = runner.RESULTS / "gnn" / "predictions" / (job["name"] + "_" + tag + ".json")
        runner.save_json(path, records)
        predictions[tag] = {"path": str(path.relative_to(runner.RESULTS)), "sha256": runner.sha256(path),
                            "graphs": len(records)}
    result = {"name": job["name"], "config": config, "epoch": completed,
              "epochs_trained": target - int(job.get("start_epoch", 0)),
              "checkpoint": str(checkpoint_path.relative_to(runner.RESULTS)),
              "checkpoint_sha256": runner.sha256(checkpoint_path), "predictions": predictions,
              "elapsed_seconds": time.monotonic() - started, "history": state["history"],
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}
    result.update({"source_hashes": upload_hashes, "corpus_manifest_hash": manifest_hash,
                   "training_graphs": len(train_ids), "training_ids_sha256": train_ids_hash})
    result["prediction_sets"] = job.get("prediction_sets")
    runner.save_json(runner.PREFLIGHT / "result.json", result)
    runner.commit_volume(volume)
    return result


# Evaluate assigned graphs in the exact CPU image and reuse paired baseline evidence.
def remote_evaluation_batch(job, volume_name, upload_hashes, manifest_hash):
    import sys
    import time
    from pathlib import Path
    import modal
    root = Path("/root/qaoa")
    sys.path.insert(0, str(root / "src"))
    import modal_app as runner
    runner.configure_threads()
    runner.PREFLIGHT = runner.RESULTS / "worker_proofs" / "evaluation" / job["name"] / ("batch_%04d" % job["batch"])
    for relative, expected in upload_hashes.items():
        runner.require(runner.sha256(root / relative) == expected, "CPU evaluation source bytes differ")
    runner.check_input_hashes()
    runner.stage_zero()
    runner.require(runner.sha256(runner.RESULTS / "gnn" / "corpus_manifest.json") == manifest_hash,
                   "Evaluation corpus manifest differs")
    import gnn
    volume = modal.Volume.from_name(volume_name)
    manifest = runner.json.loads((runner.RESULTS / "gnn" / "corpus_manifest.json").read_text())
    rows = runner.read_corpus_rows(job["corpus"], job["ids"], manifest)
    baselines, oracle_hits = {}, {}
    started = time.monotonic()
    signature = gnn.hashlib.sha256("".join(gnn.inspect.getsource(function) for function in
                                         [gnn.measured_polish, gnn.random_cost, gnn.graph_baselines]).encode()).hexdigest()
    for row in rows:
        runner.require(time.time() <= job["deadline_epoch"], "Evaluation resource reservation expired")
        path = runner.RESULTS / "gnn" / "baselines" / job["set_name"] / ("%d.json" % row["id"])
        graph_hash = runner.hashlib.sha256(runner.json.dumps([row["n"], row["edges"], row["w_sorted"], row["floor"]]).encode()).hexdigest()
        if path.exists():
            cached = runner.json.loads(path.read_text())
            runner.require(cached["protocol_hash"] == signature and cached["graph_hash"] == graph_hash
                           and cached["restarts"] == job["restarts"] and cached["seed"] == gnn.MODEL_SEED,
                           "Cached baseline uses a different graph or evaluation protocol")
            baseline = cached["baseline"]
        else:
            baseline = gnn.graph_baselines(row, restarts=job["restarts"], seed=gnn.MODEL_SEED)
            runner.save_json(path, {"protocol_hash": signature, "graph_hash": graph_hash,
                                   "restarts": job["restarts"], "seed": gnn.MODEL_SEED, "baseline": baseline})
        baselines[row["id"]] = baseline
        oracle_hits[row["id"]] = baseline["oracle"]["hit"]
    if job.get("predictions"):
        path = runner.RESULTS / job["predictions"]["path"]
        runner.require(runner.sha256(path) == job["predictions"]["sha256"], "Evaluation prediction bytes differ")
        predictions = {row["id"]: row for row in runner.json.loads(path.read_text())}
        selected = {graph_id: predictions[graph_id] for graph_id in job["ids"] if graph_id in predictions}
        result = gnn.evaluate_predictions(rows, selected, baselines, run_oracle_gate=False, summarize=False,
                                          measure_labels=job["corpus"] != "ian")
    else:
        result = {"results": {name: [gnn.baseline_metric(row, baselines[row["id"]], name) for row in rows]
                              for name in ["random", "zero", "oracle"]}}
    result.update({"name": job["name"], "batch": job["batch"], "oracle_hits": oracle_hits,
                   "expected_ids": job["ids"], "elapsed_seconds": time.monotonic() - started})
    path = runner.RESULTS / "gnn" / "evaluations" / job["name"] / ("batch_%04d.json" % job["batch"])
    runner.save_json(path, result)
    runner.commit_volume(volume)
    return result


# Fetch a completed artifact without copying unrelated Volume content.
def download_artifact(relative):
    import modal
    path = RESULTS / relative
    require(path.resolve().is_relative_to(RESULTS.resolve()), "Artifact path escapes results")
    data = b"".join(modal.Volume.from_name(VOLUME_NAME).read_file(relative))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# Reuse one GPU across validation checks and stop it when the training phase ends.
@contextmanager
def training_session(name):
    import modal
    global TRAINING_POOL
    require(TRAINING_POOL is None, "A GPU training session is already active")
    verify_workspace()
    spending = reconciled_spend_reservation()
    budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"],
                TRAIN_EVAL_RESERVE_USD)
    upload = {relative: sha256(ROOT / relative) for relative in UPLOAD_FILES}
    volume = modal.Volume.from_name(VOLUME_NAME)
    hourly = TRAINING_MAX_GPU_HOURLY_USD + 4 * .0473 + 16 * .008
    cost = read_evidence(PREFLIGHT / "training_pool_budget.json")
    require(cost is not None and cost["model_source_sha256"] == sha256(ROOT / "src" / "gnn.py")
            and cost["maximum_worker_hourly_usd"] >= hourly and cost["total_projection_usd"] < SPEND_STOP_USD
            and cost["GPU_overhead_seconds_per_input"] >= TRAINING_IDLE_SECONDS + 30,
            "Cost the complete warm-worker plan before training")
    app = modal.App(APP_NAME + "-training-session-" + name)
    function = app.function(image=gpu_image(), gpu=TRAINING_GPU_TYPES, cpu=(4, 4), memory=(16384, 16384),
                            timeout=TRAINING_TIMEOUT, retries=1, max_containers=1, scaledown_window=TRAINING_IDLE_SECONDS,
                            volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                            name="training_block")(remote_train_block)
    try:
        with app.run(detach=True):
            register_paid_app(app.app_id, "training session " + name, hourly)
            TRAINING_POOL = {"app_id": app.app_id, "function": function, "upload_hashes": upload, "hourly": hourly}
            save_json(PREFLIGHT / ("training_session_" + name + ".json"), {
                "app_id": app.app_id, "upload_hashes": upload, "maximum_hourly_usd": hourly,
                "gpu_types": TRAINING_GPU_TYPES, "max_containers": 1, "scaledown_seconds": TRAINING_IDLE_SECONDS,
                "created_epoch": time.time()})
            yield
    finally:
        TRAINING_POOL = None
        if app.app_id:
            app.stop()
            register_paid_app(app.app_id, "training session " + name, hourly, stopped=True)


# Train a recorded block or reattach its existing call rather than duplicate GPU work.
def launch_training_job(job):
    import modal
    require((PREFLIGHT / "stage3_gate.json").exists(), "Stage 3 has not passed")
    proof = read_evidence(RESULTS / "gnn" / "torch_energy_check_cuda.json")
    require(proof is not None and proof.get("implementation_sha256") == sha256(ROOT / "src" / "gnn.py"),
            "Run the independent GPU numerical gate before training")
    torch_gate(proof["checks"])
    require(job["name"].replace("_", "").isalnum(), "Unsafe training artifact name")
    result_path = RESULTS / "gnn" / "training_jobs" / (job["name"] + ".json")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        require(result["config"] == job["config"] and result["epoch"] == job["target_epoch"],
                "Existing training result uses a different requested configuration")
        require(result["source_hashes"]["src/gnn.py"] == sha256(ROOT / "src" / "gnn.py"),
                "Existing training result uses a different model source")
        require(result.get("prediction_sets") == job.get("prediction_sets"), "Existing prediction job uses different test sets")
        if "train_ids" in job:
            require(result.get("training_ids_sha256") == hashlib.sha256(json.dumps(job["train_ids"]).encode()).hexdigest(),
                    "Existing training job uses different subset ids")
        return result
    verify_workspace()
    spending = reconciled_spend_reservation()
    budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"],
                TRAIN_EVAL_RESERVE_USD)
    launch_path = PREFLIGHT / "training_launches" / (job["name"] + ".json")
    hourly = TRAINING_MAX_GPU_HOURLY_USD + 4 * .0473 + 16 * .008
    if launch_path.exists():
        previous = json.loads(launch_path.read_text())
        hourly = previous.get("maximum_hourly_usd", previous["reserve_usd"] / 2)
        require(previous["job"]["config"] == job["config"] and previous["job"]["target_epoch"] == job["target_epoch"],
                "Cannot reattach a different training job")
        require(previous["job"].get("train_ids") == job.get("train_ids"), "Cannot reattach a different training subset")
        result = modal.FunctionCall.from_id(previous["call_id"]).get()
        if TRAINING_POOL is None or previous["app_id"] != TRAINING_POOL["app_id"]:
            stopped = subprocess.run([sys.executable, "-m", "modal", "app", "stop", previous["app_id"],
                                       "--profile", MODAL_PROFILE, "--yes"], capture_output=True, text=True)
            already_stopped = stopped.stderr.strip().startswith("App is already stopped.")
            require(stopped.returncode == 0 or already_stopped, "Could not stop the reattached training app")
            register_paid_app(previous["app_id"], "training " + job["name"], hourly, stopped=True)
    elif TRAINING_POOL is not None:
        upload = {relative: sha256(ROOT / relative) for relative in UPLOAD_FILES}
        require(upload == TRAINING_POOL["upload_hashes"], "Training-session source bytes changed; stop and reload the session")
        job = dict(job, deadline_epoch=time.time() + 2 * TRAINING_TIMEOUT)
        call = TRAINING_POOL["function"].spawn(job, VOLUME_NAME, upload, sha256(RESULTS / "gnn" / "corpus_manifest.json"))
        save_json(launch_path, {"app_id": TRAINING_POOL["app_id"], "call_id": call.object_id, "job": job,
                  "upload_hashes": upload, "reserve_usd": 2 * hourly, "maximum_hourly_usd": hourly,
                  "gpu_types": TRAINING_GPU_TYPES, "pooled_app": True, "created_epoch": time.time()})
        result = call.get()
    else:
        volume = modal.Volume.from_name(VOLUME_NAME)
        upload = {name: sha256(ROOT / name) for name in UPLOAD_FILES}
        app = modal.App(APP_NAME + "-train-" + job["name"])
        function = app.function(image=gpu_image(), gpu=TRAINING_GPU_TYPES, cpu=(4, 4), memory=(16384, 16384),
                                timeout=TRAINING_TIMEOUT, retries=1, max_containers=1, scaledown_window=2,
                                volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                                name="training_block")(remote_train_block)
        job = dict(job, deadline_epoch=time.time() + 2 * TRAINING_TIMEOUT)
        try:
            with app.run(detach=True):
                register_paid_app(app.app_id, "training " + job["name"], hourly)
                call = function.spawn(job, VOLUME_NAME, upload, sha256(RESULTS / "gnn" / "corpus_manifest.json"))
                save_json(launch_path, {"app_id": app.app_id, "call_id": call.object_id, "job": job,
                          "upload_hashes": upload, "reserve_usd": 2 * hourly, "maximum_hourly_usd": hourly,
                          "gpu_types": TRAINING_GPU_TYPES, "created_epoch": time.time()})
                result = call.get()
        finally:
            if app.app_id:
                app.stop()
                register_paid_app(app.app_id, "training " + job["name"], hourly, stopped=True)
    require(result["epoch"] == job["target_epoch"], "Training block did not reach its expected epoch")
    save_json(result_path, result)
    for info in result["predictions"].values():
        path = download_artifact(info["path"])
        require(sha256(path) == info["sha256"], "Downloaded predictions differ from the GPU output")
    return result


# Evaluate one fixed manifest and enforce the oracle gate only after all graph batches return.
def launch_evaluation(name, corpus, set_name, ids, predictions=None, restarts=20):
    import modal
    import gnn
    require(name.replace("_", "").isalnum(), "Unsafe evaluation artifact name")
    require(len(ids) == len(set(ids)) and bool(ids), "Evaluation ids must be nonempty and unique")
    path = RESULTS / "gnn" / "evaluations" / (name + ".json")
    if path.exists():
        previous = json.loads(path.read_text())
        require(previous["expected_ids"] == ids and previous["predictions"] == predictions
                and previous["restarts"] == restarts, "Existing evaluation uses a different protocol")
        return repair_cached_evaluation_numerics(previous, path)
    launch_path = PREFLIGHT / "evaluation_launches" / (name + ".json")
    require(not launch_path.exists(), "An unfinished evaluation must be reconciled before resubmission")
    verify_workspace()
    spending = reconciled_spend_reservation()
    budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"],
                TRAIN_EVAL_RESERVE_USD)
    batch_size = LARGE_EVALUATION_BATCH_SIZE if corpus == "large" else 25
    jobs = [{"name": name, "batch": k // batch_size, "corpus": corpus, "set_name": set_name, "ids": ids[k:k + batch_size],
             "predictions": predictions, "restarts": restarts, "deadline_epoch": time.time() + 2 * EVALUATION_TIMEOUT}
            for k in range(0, len(ids), batch_size)]
    containers = min(80, len(jobs))
    hourly = containers * (.0473 + 2 * .008)
    volume = modal.Volume.from_name(VOLUME_NAME)
    upload = {relative: sha256(ROOT / relative) for relative in UPLOAD_FILES}
    app = modal.App(APP_NAME + "-eval-" + name)
    function = app.function(image=data_image(), cpu=(1, 1), memory=(1024, 2048), timeout=EVALUATION_TIMEOUT, retries=1,
                            max_containers=containers, scaledown_window=2,
                            volumes={REMOTE_ROOT + "/results": volume}, serialized=True,
                            name="cpu_evaluation")(remote_evaluation_batch)
    records, oracle = {}, {}
    try:
        with app.run(detach=True):
            register_paid_app(app.app_id, "evaluation " + name, hourly)
            save_json(launch_path, {"app_id": app.app_id, "jobs": jobs, "upload_hashes": upload,
                      "reserve_usd": hourly / 2, "created_epoch": time.time()})
            for batch in function.map(jobs, kwargs={"volume_name": VOLUME_NAME, "upload_hashes": upload,
                                                    "manifest_hash": sha256(RESULTS / "gnn" / "corpus_manifest.json")},
                                      order_outputs=False, return_exceptions=True):
                require(not isinstance(batch, BaseException), "CPU evaluation batch failed: %s" % batch)
                for method, values in batch["results"].items():
                    records.setdefault(method, []).extend(values)
                for key, value in batch["oracle_hits"].items():
                    graph_id = int(key)
                    require(graph_id not in oracle, "Duplicate evaluation oracle id")
                    oracle[graph_id] = value
                save_json(RESULTS / "gnn" / "evaluations" / name / ("batch_%04d.json" % batch["batch"]), batch)
                LOG.info("%s: %d/%d graph evaluations received", name, len(oracle), len(ids))
    finally:
        if app.app_id:
            app.stop()
            register_paid_app(app.app_id, "evaluation " + name, hourly, stopped=True)
    return summarize_evaluation(name, corpus, set_name, ids, predictions, restarts, records, oracle)


# Summarize complete saved records with the same full-manifest numerical checks.
def summarize_evaluation(name, corpus, set_name, ids, predictions, restarts, records, oracle):
    import gnn
    path = RESULTS / "gnn" / "evaluations" / (name + ".json")
    rows = read_corpus_rows(corpus, ids)
    numerical_repairs = repair_polish_energy_roundoff(rows, records)
    expected = {row["id"]: row["n"] for row in rows}
    oracle_result = oracle_gate(expected, {row["id"]: dict(row, oracle_hit=oracle.get(row["id"], False)) for row in rows})
    summaries = {method: gnn.metric_summary(values, ids) for method, values in records.items()}
    for field in ["n", "type"]:
        for value in sorted({row[field] for row in rows}):
            selected = {row["id"] for row in rows if row[field] == value}
            summaries[field + "=" + str(value)] = {
                method: gnn.metric_summary([row for row in values if row["id"] in selected], sorted(selected))
                for method, values in records.items()}
    label_summaries = {method: gnn.label_summary(values, ids) for method, values in records.items()
                       if method in ["raw", "snap", "better"] and corpus != "ian"}
    result = {"name": name, "corpus": corpus, "set_name": set_name, "expected_ids": ids,
              "predictions": predictions, "restarts": restarts, "oracle_gate": oracle_result,
              "results": records, "summaries": summaries, "label_summaries": label_summaries}
    if numerical_repairs:
        result["posthoc_numerical_repairs"] = numerical_repairs
    save_json(path, result)
    return result


# Use physical periods before the independent matrix-exponential calculation.
def period_equivalent_angles(point, weights):
    import numpy as np
    point, weights = np.asarray(point, dtype=float), np.asarray(weights, dtype=float)
    require(point.ndim == 1 and len(point) >= len(weights) and np.isfinite(point).all(),
            "Invalid saved angles for periodic verification")
    scales = np.r_[weights, np.ones(len(point) - len(weights))]
    if np.max(np.abs(point)) <= 1000:
        return ((point * scales + np.pi / 2) % np.pi - np.pi / 2) / scales
    # Binary64 modulo can lose phase precision for the rare unbounded L-BFGS-B
    # points. Preserve the exact saved binary64 inputs while reducing by true pi.
    import mpmath as mp
    with mp.workdps(80):
        values = []
        for value, scale in zip(point, scales):
            phase = mp.mpf(float(value)) * mp.mpf(float(scale))
            reduced = (phase + mp.pi / 2) % mp.pi - mp.pi / 2
            values.append(float(reduced / mp.mpf(float(scale))))
    return np.asarray(values)


# Rebuild p=1 final-state energies without trusting optimizer result.fun metadata.
def independent_p1_final_energies(row, points):
    import numpy as np
    points = np.atleast_2d(points).astype(float)
    weights = np.asarray(row["w_sorted"])
    edges = np.asarray(row["edges"], dtype=int)
    m, n = len(weights), row["n"]
    for index in np.flatnonzero(np.max(np.abs(points), axis=1) > 1000):
        points[index] = period_equivalent_angles(points[index], weights)
    phases = points[:, :m] * weights
    theta = np.zeros((len(points), n, n))
    theta[:, edges[:, 0], edges[:, 1]] = phases
    theta[:, edges[:, 1], edges[:, 0]] = phases
    left, right = theta[:, edges[:, 0], :].copy(), theta[:, edges[:, 1], :].copy()
    for coordinate in [0, 1]:
        left[:, np.arange(m), edges[:, coordinate]] = 0
        right[:, np.arange(m), edges[:, coordinate]] = 0
    cosine, sine = np.cos(2 * points[:, m:]), np.sin(2 * points[:, m:])
    u, v = edges[:, 0], edges[:, 1]
    correlation = np.sin(2 * phases) * (
        cosine[:, u] * sine[:, v] * np.cos(2 * right).prod(axis=2)
        + sine[:, u] * cosine[:, v] * np.cos(2 * left).prod(axis=2))
    correlation += .5 * sine[:, u] * sine[:, v] * (
        np.cos(2 * (left - right)).prod(axis=2) - np.cos(2 * (left + right)).prod(axis=2))
    return .5 * ((correlation - 1) * weights).sum(axis=1)


# Correct independently demonstrated energy metadata, never optimizer work.
def repair_polish_energy_roundoff(rows, records):
    import numpy as np
    import check_rows
    indexed = {row["id"]: row for row in rows}
    repairs = []
    for method in ["raw", "snap", "better"]:
        for record in records.get(method, []):
            row = indexed[record["id"]]
            m, n = len(row["edges"]), row["n"]
            final_energies = independent_p1_final_energies(row, [point["angles_after"] for point in record["polishes"]])
            for index, (polish, physical_energy) in enumerate(zip(record["polishes"], final_energies)):
                original = polish["angles_after"]
                require(bool(physical_energy <= row["floor"] + ENERGY_TOL) == bool(polish["hit"]),
                        "Independent final-state energy changes a saved success decision")
                if abs(physical_energy - polish["energy_after"]) <= DATASET_ENERGY_TOL:
                    continue
                bounded = period_equivalent_angles(original, row["w_sorted"])
                independent = check_rows.energy(n, row["edges"], row["w_sorted"], bounded[:m], bounded[m:])
                require(abs(independent - physical_energy) <= DATASET_ENERGY_TOL,
                        "Independent p=1 formula differs from the pinned checker")
                before = polish["energy_after"]
                polish["optimizer_energy_after_before_numerical_repair"] = before
                polish["energy_after"] = independent
                polish["energy_verification"] = "Final-state energy independently rebuilt and checked by the unchanged pinned checker, using 80-digit period reduction for large angles; saved angles and optimizer work unchanged"
                repairs.append({"id": row["id"], "n": n, "method": method, "polish_index": index,
                                "optimizer_energy": before, "verified_energy": independent,
                                "difference": independent - before, "hit_unchanged": True,
                                "angles_sha256": hashlib.sha256(json.dumps(original).encode()).hexdigest()})
            # The same saved candidate attempts and the same stopping decisions
            # define costs. Only the displayed best final energy can change.
            best_energy = min(polish["energy_after"] for polish in record["polishes"])
            record["energy_after"] = best_energy
            record["approximation_ratio_after"] = -best_energy / row["max_cut"]
    return repairs


# Retain byte-exact original aggregates before correcting their energy metadata.
def repair_cached_evaluation_numerics(previous, path):
    rows = read_corpus_rows(previous["corpus"], previous["expected_ids"])
    repairs = repair_polish_energy_roundoff(rows, previous["results"])
    if not repairs:
        return previous
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    backup = PREFLIGHT / "runner_repairs" / "polish_energy_originals" / (digest + ".json")
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_bytes(original)
    require(sha256(backup) == digest, "Original evaluation backup differs")
    oracle = {record["id"]: bool(record["hit"]) for record in previous["results"]["oracle"]}
    repaired = summarize_evaluation(previous["name"], previous["corpus"], previous["set_name"],
                                    previous["expected_ids"], previous["predictions"], previous["restarts"],
                                    previous["results"], oracle)
    repaired["posthoc_numerical_repairs"] = previous.get("posthoc_numerical_repairs", []) + repairs
    save_json(path, repaired)
    proof_path = PREFLIGHT / "runner_repairs" / ("polish_energy_repair_" + previous["name"] + ".json")
    prior_repair = read_evidence(proof_path)
    save_json(proof_path, {
        "original_path": str(backup.relative_to(RESULTS)), "original_sha256": digest,
        "corrected_sha256": sha256(path), "repairs": repairs, "new_optimizations": 0,
        "circuit_costs_unchanged": True, "success_decisions_unchanged": True,
        "energy_tolerance_unchanged": DATASET_ENERGY_TOL, "previous_repair": prior_repair, "captured_epoch": time.time()})
    return repaired


# Recover a finished evaluation after a local write failure without repeating polishes.
def recover_evaluation(name):
    require(name.replace("_", "").isalnum(), "Unsafe evaluation recovery name")
    path = RESULTS / "gnn" / "evaluations" / (name + ".json")
    require(not path.exists(), "The evaluation already has its complete result")
    launch = read_evidence(PREFLIGHT / "evaluation_launches" / (name + ".json"))
    require(launch is not None and bool(launch["jobs"]), "The original evaluation launch is missing")
    paid = read_evidence(PREFLIGHT / "paid_apps" / (launch["app_id"] + ".json"))
    require(paid is not None and paid.get("stopped_epoch"), "Stop the original evaluation app before recovery")
    require(launch["upload_hashes"]["src/gnn.py"] == sha256(ROOT / "src" / "gnn.py"), "Evaluation model source differs")
    for relative, expected in EXPECTED_HASHES.items():
        require(launch["upload_hashes"][relative] == expected, "Evaluation used different pinned inputs")
    jobs = launch["jobs"]
    ids = [graph_id for job in jobs for graph_id in job["ids"]]
    require(len(ids) == len(set(ids)), "The original evaluation repeats graph ids")
    first = jobs[0]
    records, oracle, hashes = {}, {}, {}
    methods = {"raw", "snap", "better", "random", "zero", "oracle"} if first["predictions"] else {"random", "zero", "oracle"}
    for job in jobs:
        require(all(job[key] == first[key] for key in ["corpus", "set_name", "predictions", "restarts"])
                and job["name"] == name, "Evaluation batches use different protocols")
        batch_path = RESULTS / "gnn" / "evaluations" / name / ("batch_%04d.json" % job["batch"])
        batch = read_evidence(batch_path)
        require(batch is not None and batch["batch"] == job["batch"] and batch["name"] == name
                and batch["expected_ids"] == job["ids"], "An expected evaluation batch is missing or differs")
        require(set(batch["results"]) == methods, "An evaluation method is missing")
        assigned = set(job["ids"])
        require({int(key) for key in batch["oracle_hits"]} == assigned, "An oracle result is missing")
        for method, values in batch["results"].items():
            require(len(values) == len(assigned) and {row["id"] for row in values} == assigned,
                    "Evaluation recovery omits or duplicates a graph")
            records.setdefault(method, []).extend(values)
        for key, value in batch["oracle_hits"].items():
            require(int(key) not in oracle, "Evaluation recovery repeats an oracle id")
            oracle[int(key)] = value
        hashes[batch_path.name] = sha256(batch_path)
    result = summarize_evaluation(name, first["corpus"], first["set_name"], ids, first["predictions"],
                                  first["restarts"], records, oracle)
    save_json(PREFLIGHT / ("evaluation_recovery_" + name + ".json"), {
        "app_id": launch["app_id"], "batches": len(jobs), "expected_graphs": len(ids),
        "source_batch_hashes": hashes, "result_sha256": sha256(path), "new_paid_compute": False,
        "captured_epoch": time.time()})
    return result


# Choose post-processing using only paired validation graphs and a fixed simplicity order.
def validation_strategy(evaluation):
    import gnn
    records = {name: evaluation["results"][name] for name in ["raw", "snap", "better"]}
    return gnn.choose_variant(records, {"raw": 0, "snap": 1, "better": 2})


# Complete the matched 40-epoch ablations before selecting the final architecture.
def run_bakeoff():
    import gnn
    split = json.loads((RESULTS / "gnn" / "splits.json").read_text())
    classes = json.loads((RESULTS / "gnn" / "classification_values.json").read_text())["theta"]
    ids = split["ids"]["val_selection"]
    require(len(ids) == 2000, "Bake-off selection requires the fixed 2000-graph validation subset")
    launch_evaluation("val_baselines", "full", "val", ids, restarts=20)
    configurations = {
        "A": {"baseline": True, "classification": False, "hypotheses": 1},
        "B": {"baseline": False, "classification": False, "hypotheses": 1},
        "Bcls": {"baseline": False, "classification": True, "hypotheses": 1, "theta_classes": classes},
        "E": {"baseline": False, "classification": False, "hypotheses": 1, "label_weight": 0.0, "energy_weight": 1.0},
    }
    checkpoints, evaluations, strategies = {}, {}, {}
    for name, architecture in configurations.items():
        config = dict(architecture, seed=42, maximum_epochs=40)
        first = launch_training_job({"name": name + "_s42_e0020", "config": config,
                                     "target_epoch": 20, "start_epoch": 0})
        final = launch_training_job({"name": name + "_s42_e0040", "config": config,
                                     "target_epoch": 40, "start_epoch": 20, "input_checkpoint": first["checkpoint"]})
        checkpoints[name] = {"20": first, "40": final}
        evaluation = launch_evaluation(name + "_s42_val", "full", "val", ids, final["predictions"]["val"], 20)
        evaluations[name], strategies[name] = evaluation, validation_strategy(evaluation)
    base = gnn.choose_variant({name: evaluations[name]["results"][strategies[name]["chosen"]]
                               for name in ["B", "Bcls"]}, {"B": 0, "Bcls": 1})["chosen"]
    for alpha, name in [(.3, "C03"), (1.0, "C10")]:
        config = dict(configurations[base], seed=42, maximum_epochs=40, label_weight=1.0, energy_weight=alpha)
        final = launch_training_job({"name": name + "_s42_e0040", "config": config, "target_epoch": 40,
                                     "start_epoch": 20, "input_checkpoint": checkpoints[base]["20"]["checkpoint"]})
        checkpoints[name] = {"40": final}
        evaluations[name] = launch_evaluation(name + "_s42_val", "full", "val", ids, final["predictions"]["val"], 20)
        strategies[name] = validation_strategy(evaluations[name])
        configurations[name] = config
    config = dict(configurations[base], hypotheses=4, seed=42, maximum_epochs=40)
    final = launch_training_job({"name": "D_s42_e0040", "config": config, "target_epoch": 40, "start_epoch": 0})
    checkpoints["D"] = {"40": final}
    configurations["D"] = config
    evaluations["D"] = launch_evaluation("D_s42_val", "full", "val", ids, final["predictions"]["val"], 20)
    strategies["D"] = validation_strategy(evaluations["D"])
    selection = gnn.choose_variant({name: evaluation["results"][strategies[name]["chosen"]]
                                    for name, evaluation in evaluations.items()},
                                   {"A": 0, "B": 1, "E": 2, "Bcls": 3, "C03": 4, "C10": 4, "D": 5})
    chosen = selection["chosen"]
    result = {"selection": selection, "base_for_C_and_D": base,
              "configuration": dict(checkpoints[chosen]["40"]["config"]),
              "strategy": strategies[chosen]["chosen"], "strategies": strategies,
              "checkpoint_paths": {name: {epoch: item["checkpoint"] for epoch, item in values.items()}
                                    for name, values in checkpoints.items()},
              "C_recipe": "20 label epochs from the selected base checkpoint, then 20 energy fine-tuning epochs"}
    save_json(RESULTS / "gnn" / "bakeoff.json", result)
    return result


# Train three new seeds and stop only on the fixed validation warm-start metric.
def run_final_training():
    if (RESULTS / "gnn" / "final_training.json").exists():
        return read_evidence(RESULTS / "gnn" / "final_training.json")
    with training_session("final"):
        return train_final_seeds()


# Preserve each seed's checkpoint, validation history and early-stopping decision.
def train_final_seeds():
    import gnn
    import modal
    bakeoff = json.loads((RESULTS / "gnn" / "bakeoff.json").read_text())
    splits = json.loads((RESULTS / "gnn" / "splits.json").read_text())
    manifest = json.loads((RESULTS / "gnn" / "corpus_manifest.json").read_text())
    selection_ids = splits["ids"]["val_selection"]
    results = {}
    for seed in gnn.TRAIN_SEEDS:
        state_path = RESULTS / "gnn" / ("final_seed%d.json" % seed)
        if state_path.exists():
            results[seed] = json.loads(state_path.read_text())
            continue
        previous, best, stale = None, None, 0
        checks = []
        fine_tuning = bakeoff["selection"]["chosen"] in ["C03", "C10"]
        alpha = bakeoff["configuration"].get("energy_weight", 0.0)
        for epoch in range(VALIDATION_INTERVAL, gnn.FINAL_EPOCHS + 1, VALIDATION_INTERVAL):
            config = dict(bakeoff["configuration"], seed=seed, maximum_epochs=gnn.FINAL_EPOCHS)
            if fine_tuning and epoch <= 75:
                config["energy_weight"] = 0.0
            job = {"name": "final_s%d_e%04d" % (seed, epoch), "config": config,
                   "target_epoch": epoch, "start_epoch": epoch - VALIDATION_INTERVAL}
            if previous is not None:
                job["input_checkpoint"] = previous["checkpoint"]
            trained = launch_training_job(job)
            evaluation = launch_evaluation(job["name"] + "_val", "full", "val", selection_ids,
                                           trained["predictions"]["val"], 20)
            metrics = evaluation["summaries"][bakeoff["strategy"]]["metrics"]
            score, hit = metrics["geometric_mean_cost_ratio"]["value"], metrics["hit_rate"]["value"]
            eligible = not fine_tuning or epoch > 75
            improved = eligible and (best is None or score < best["score"] - 1e-6
                                      or abs(score - best["score"]) <= 1e-12 and hit > best["hit"])
            if improved:
                best = {"score": score, "hit": hit, "epoch": epoch, "training": trained, "evaluation": evaluation["name"]}
                stale = 0
            elif eligible:
                stale += 1
            checks.append({"epoch": epoch, "score": score, "hit": hit, "eligible": eligible, "improved": improved})
            save_json(RESULTS / "gnn" / ("final_seed%d_progress.json" % seed), {"checks": checks, "best": best, "stale": stale})
            previous = trained
            if stale >= EARLY_STOP_CHECKS:
                break
        require(best is not None, "Final training did not produce an eligible checkpoint")
        prediction_sets = {"test_split": {"corpus": "full", "ids": splits["ids"]["test"]},
                           "large": {"corpus": "large", "ids": sorted(int(key) for key in json.loads(
                               (RESULTS / manifest["corpora"]["large"]["index"]).read_text()))},
                           "ian": {"corpus": "ian", "ids": list(range(20))}}
        prediction = launch_training_job({"name": "final_s%d_predictions" % seed,
                         "config": best["training"]["config"], "target_epoch": best["epoch"], "start_epoch": best["epoch"],
                         "input_checkpoint": best["training"]["checkpoint"], "prediction_sets": prediction_sets})
        final_path = "gnn/checkpoints/final_seed%d.pt" % seed
        modal.Volume.from_name(VOLUME_NAME).copy_files([best["training"]["checkpoint"]], final_path)
        result = {"seed": seed, "best": best, "checks": checks, "prediction": prediction,
                  "final_checkpoint": final_path, "stopped_epoch": checks[-1]["epoch"], "strategy": bakeoff["strategy"],
                  "energy_fine_tuning_from_epoch": 76 if fine_tuning else None}
        save_json(state_path, result)
        results[seed] = result
    save_json(RESULTS / "gnn" / "final_training.json", {"seeds": results, "bakeoff_strategy": bakeoff["strategy"]})
    return results


# Compare nested training subsets with the same winning architecture and validation protocol.
def run_learning_curve():
    approval = read_evidence(PREFLIGHT / "learning_curve_approval.json")
    require(approval is not None and approval.get("user_approved") is True, "Learning-curve approval is missing")
    if (RESULTS / "gnn" / "learning_curve.json").exists():
        return read_evidence(RESULTS / "gnn" / "learning_curve.json")
    with training_session("learning_curve"):
        return train_learning_curve()


# Fit the approved smaller subsets and reuse the existing full-data fits.
def train_learning_curve():
    import numpy as np
    import gnn
    import modal
    approval = read_evidence(PREFLIGHT / "learning_curve_approval.json")
    require(approval is not None and approval.get("user_approved") is True,
            "Additional learning-curve fits require the pending user clarification")
    final = read_evidence(RESULTS / "gnn" / "final_results.json")
    training = read_evidence(RESULTS / "gnn" / "final_training.json")
    bakeoff = read_evidence(RESULTS / "gnn" / "bakeoff.json")
    require(final is not None and training is not None and bakeoff is not None,
            "Complete the winning model's full-data results before the learning curve")
    path = RESULTS / "gnn" / "learning_curve.json"
    if path.exists():
        return read_evidence(path)
    spending = reconciled_spend_reservation()
    training_jobs = [read_evidence(item) for item in (RESULTS / "gnn" / "training_jobs").glob("final_s*_e*.json")]
    require(bool(training_jobs), "Learning-curve costing requires measured final-training blocks")
    maximum_block = max(item["elapsed_seconds"] for item in training_jobs)
    validation_apps = [read_evidence(item) for item in (PREFLIGHT / "paid_apps").glob("*.json")]
    validation_apps = [item for item in validation_apps if item["name"].startswith("evaluation final_s")
                       and item["name"].endswith("_val") and item.get("stopped_epoch") is not None]
    require(bool(validation_apps), "Learning-curve costing requires completed validation apps")
    maximum_validation = max((item["stopped_epoch"] - item["created_epoch"] + 30)
                             * item["maximum_hourly_usd"] / 3600 for item in validation_apps)
    blocks = 2 * len(gnn.TRAIN_SEEDS) * (gnn.FINAL_EPOCHS // VALIDATION_INTERVAL)
    hourly = TRAINING_MAX_GPU_HOURLY_USD + 4 * .0473 + 16 * .008
    projection = 1.5 * blocks * ((maximum_block + 120) * hourly / 3600 + maximum_validation) + 5.0
    budget_gate(spending["reported_app_cost_usd"], spending["spend_reservation_usd"] - spending["reported_app_cost_usd"], projection)
    require(spending["spend_reservation_usd"] + projection < SPEND_STOP_USD,
            "Learning-curve projection would reach the actual-spend stop")
    save_json(PREFLIGHT / "learning_curve_budget.json", {"maximum_epochs_per_fit": gnn.FINAL_EPOCHS,
              "fits": 2 * len(gnn.TRAIN_SEEDS), "maximum_gpu_blocks": blocks,
              "measured_maximum_full_training_block_seconds": maximum_block,
              "measured_maximum_validation_upper_cost_usd": maximum_validation,
              "overhead_seconds_per_block": 120, "headroom": 1.5, "test_evaluation_reserve_usd": 5,
              "remaining_projection_usd": projection, "prior_upper_reservation_usd": spending["spend_reservation_usd"],
              "total_projection_usd": spending["spend_reservation_usd"] + projection})
    splits = read_evidence(RESULTS / "gnn" / "splits.json")
    train_rows = read_corpus_rows("full", splits["ids"]["train"])
    test_ids, val_ids = splits["ids"]["test"], splits["ids"]["val_selection"]
    rng = np.random.default_rng(42)
    orders = {n: rng.permutation(sorted(row["id"] for row in train_rows if row["n"] == n)).tolist()
              for n in sorted({row["n"] for row in train_rows})}
    subsets = {}
    for percent in [25, 50, 100]:
        target = int(round(len(train_rows) * percent / 100))
        allocation = {n: int(len(ids) * percent // 100) for n, ids in orders.items()}
        remainder = target - sum(allocation.values())
        priority = sorted(orders, key=lambda n: (-(len(orders[n]) * percent / 100 - allocation[n]), n))
        for n in priority[:remainder]:
            allocation[n] += 1
        subsets[str(percent)] = sorted(i for n, ids in orders.items() for i in ids[:allocation[n]])
    require(set(subsets["25"]) <= set(subsets["50"]) <= set(subsets["100"])
            and subsets["100"] == splits["ids"]["train"], "Learning-curve subsets are not nested")
    subset_path = RESULTS / "gnn" / "learning_curve_subsets.json"
    if subset_path.exists():
        require(read_evidence(subset_path)["ids"] == subsets, "Learning-curve subset ids changed")
    else:
        save_json(subset_path, {"seed": 42, "ids": subsets, "stratification": "n, within the frozen training split",
                  "splits_sha256": sha256(RESULTS / "gnn" / "splits.json")})
    outcomes = {"100": {"graphs": len(train_rows), "reused_full_final_training": True,
                         "summary": final["sets"]["test_split"][bakeoff["strategy"]]["summary"],
                         "seed_spread": final["sets"]["test_split"][bakeoff["strategy"]]["seed_spread"]}}
    fine_tuning = bakeoff["selection"]["chosen"] in ["C03", "C10"]
    for percent in [25, 50]:
        selected = subsets[str(percent)]
        selected_ids = set(selected)
        subset_rows = [row for row in train_rows if row["id"] in selected_ids]
        architecture = dict(bakeoff["configuration"])
        if architecture.get("classification"):
            schema = {"theta_other_fraction": .03 if len(architecture["theta_classes"]) > 2 else 0}
            architecture["theta_classes"] = gnn.classification_values(subset_rows, schema)
        seed_results, evaluations = {}, []
        for seed in gnn.TRAIN_SEEDS:
            prefix = "curve%d_s%d" % (percent, seed)
            state_path = RESULTS / "gnn" / (prefix + ".json")
            done = read_evidence(state_path)
            if done is None:
                previous, best, stale, checks = None, None, 0, []
                for epoch in range(VALIDATION_INTERVAL, gnn.FINAL_EPOCHS + 1, VALIDATION_INTERVAL):
                    config = dict(architecture, seed=seed, maximum_epochs=gnn.FINAL_EPOCHS)
                    if fine_tuning and epoch <= 75:
                        config["energy_weight"] = 0.0
                    job = {"name": prefix + "_e%04d" % epoch, "config": config, "target_epoch": epoch,
                           "start_epoch": epoch - VALIDATION_INTERVAL, "train_ids": selected}
                    if previous is not None:
                        job["input_checkpoint"] = previous["checkpoint"]
                    trained = launch_training_job(job)
                    evaluation = launch_evaluation(job["name"] + "_val", "full", "val", val_ids,
                                                   trained["predictions"]["val"], 20)
                    metrics = evaluation["summaries"][bakeoff["strategy"]]["metrics"]
                    score, hit = metrics["geometric_mean_cost_ratio"]["value"], metrics["hit_rate"]["value"]
                    eligible = not fine_tuning or epoch > 75
                    improved = eligible and (best is None or score < best["score"] - 1e-6
                                              or abs(score - best["score"]) <= 1e-12 and hit > best["hit"])
                    if improved:
                        best = {"score": score, "hit": hit, "epoch": epoch, "training": trained}
                        stale = 0
                    elif eligible:
                        stale += 1
                    checks.append({"epoch": epoch, "score": score, "hit": hit, "eligible": eligible, "improved": improved})
                    save_json(RESULTS / "gnn" / (prefix + "_progress.json"), {"checks": checks, "best": best, "stale": stale})
                    previous = trained
                    write_billing_summary()
                    if stale >= EARLY_STOP_CHECKS:
                        break
                require(best is not None, "Learning curve has no eligible checkpoint")
                prediction = launch_training_job({"name": prefix + "_predictions", "config": best["training"]["config"],
                    "target_epoch": best["epoch"], "start_epoch": best["epoch"], "input_checkpoint": best["training"]["checkpoint"],
                    "train_ids": selected, "prediction_sets": {"test_split": {"corpus": "full", "ids": test_ids}}})
                checkpoint = "gnn/checkpoints/" + prefix + ".pt"
                modal.Volume.from_name(VOLUME_NAME).copy_files([best["training"]["checkpoint"]], checkpoint)
                done = {"best": best, "checks": checks, "prediction": prediction, "checkpoint": checkpoint,
                        "train_graphs": len(selected), "stopped_epoch": checks[-1]["epoch"]}
                save_json(state_path, done)
            evaluation = launch_evaluation(prefix + "_test", "full", "test_split", test_ids,
                                           done["prediction"]["predictions"]["test_split"], 50)
            rows = read_corpus_rows("full", test_ids)
            verify_evaluation_energies(rows, evaluation["results"][bakeoff["strategy"]])
            evaluations.append(evaluation["results"][bakeoff["strategy"]])
            seed_results[str(seed)] = done
        aggregate = gnn.aggregate_model_seeds(evaluations, test_ids)
        outcomes[str(percent)] = {"graphs": len(selected), "seeds": seed_results,
                                 "summary": aggregate["summary"], "seed_spread": aggregate["seed_spread"]}
        save_json(RESULTS / "gnn" / "learning_curve_progress.json", outcomes)
        write_billing_summary()
    result = {"model": bakeoff["selection"]["chosen"], "strategy": bakeoff["strategy"], "fractions": outcomes,
              "train_split_graphs": len(train_rows), "test_graphs": len(test_ids), "seeds": list(gnn.TRAIN_SEEDS),
              "subset_manifest_sha256": sha256(subset_path),
              "protocol": "Nested n-stratified subsets; same architecture, seeds, 150-epoch cap and five-epoch validation/patience; subset-only class centroids and weights; fixed validation/test sets; R=50 test; reuse 100% final fits",
              "remaining_36328_graphs_generated": False}
    save_json(path, result)
    return result


# Independently rebuild every reported initial and final polished circuit energy.
def verify_evaluation_energies(rows, records):
    import check_rows
    import numpy as np
    indexed = {row["id"]: row for row in rows}
    require({row["id"] for row in records} == set(indexed), "Independent energy check omits a graph")
    maximum_before, maximum_after, checks = 0.0, 0.0, 0
    maximum_unreduced_after, maximum_angle, reduced_points = 0.0, 0.0, 0
    for record in records:
        row = indexed[record["id"]]
        n, edges, weights = row["n"], row["edges"], np.asarray(row["w_sorted"])
        m = len(edges)
        for polish in record["polishes"]:
            phase = np.asarray(polish["candidate_phase"])
            initial = np.r_[phase[:m] / weights, phase[m:]]
            final = np.asarray(polish["angles_after"])
            # Each pi shift of an edge phase or node angle changes only the
            # global state phase. Bounded representatives prevent expm's
            # scaling/squaring roundoff at optimizer angles above 1e6 radians.
            # The checker and the 1e-10 energy threshold remain unchanged.
            bounded_final = period_equivalent_angles(final, weights)
            maximum_angle = max(maximum_angle, float(np.abs(final).max()))
            reduced_points += bool(np.max(np.abs(final - bounded_final)) > 1e-12)
            if np.max(np.abs(final)) > 1000:
                unreduced = check_rows.energy(n, edges, weights, final[:m], final[m:])
                maximum_unreduced_after = max(maximum_unreduced_after, abs(unreduced - polish["energy_after"]))
            before = check_rows.energy(n, edges, weights, initial[:m], initial[m:])
            after = check_rows.energy(n, edges, weights, bounded_final[:m], bounded_final[m:])
            error_before, error_after = abs(before - polish["energy_before"]), abs(after - polish["energy_after"])
            require(error_before <= DATASET_ENERGY_TOL and error_after <= DATASET_ENERGY_TOL,
                    "Independent prediction/polish energy differs at graph %d" % row["id"])
            maximum_before, maximum_after = max(maximum_before, error_before), max(maximum_after, error_after)
            checks += 1
    return {"graphs": len(rows), "polishes_checked": checks,
            "max_initial_energy_error": maximum_before, "max_final_energy_error": maximum_after,
            "final_verification": "Pinned checker at period-equivalent bounded final angles; saved angles, optimizer work and hit decisions unchanged; corrected energies retain their original optimizer value",
            "period_reduced_final_points": reduced_points, "max_abs_original_final_angle": maximum_angle,
            "max_unreduced_final_energy_error_for_angles_over_1000": maximum_unreduced_after}


# Evaluate all held-out sets after model selection and export checked Ian-order predictions.
def run_final_evaluation():
    import gnn
    import numpy as np
    import modal
    training = json.loads((RESULTS / "gnn" / "final_training.json").read_text())
    splits = json.loads((RESULTS / "gnn" / "splits.json").read_text())
    manifest = json.loads((RESULTS / "gnn" / "corpus_manifest.json").read_text())
    sets = {"test_split": ("full", splits["ids"]["test"]),
            "large": ("large", sorted(int(key) for key in json.loads((RESULTS / manifest["corpora"]["large"]["index"]).read_text()))),
            "ian": ("ian", list(range(20)))}
    all_evaluations = {name: [] for name in sets}
    for seed in gnn.TRAIN_SEEDS:
        model = training["seeds"][str(seed)]
        for name, (corpus, ids) in sets.items():
            evaluation = launch_evaluation("final_s%d_%s" % (seed, name), corpus, name, ids,
                                           model["prediction"]["predictions"][name], 50)
            all_evaluations[name].append(evaluation)
            rows = read_corpus_rows(corpus, ids)
            for method in ["raw", "snap", "better"]:
                csv_path = RESULTS / "gnn" / "predictions_csv" / ("seed%d_%s_%s.csv" % (seed, name, method))
                gnn.prediction_csv(csv_path, rows, evaluation["results"][method])
                csv_result = csv_gate(csv_path, ids)
                energy = verify_evaluation_energies(rows, evaluation["results"][method])
                save_json(PREFLIGHT / ("prediction_gate_seed%d_%s_%s.json" % (seed, name, method)),
                          {"csv": csv_result, "polish_energies": energy})
    summary = {}
    for name, evaluations in all_evaluations.items():
        corpus, ids = sets[name]
        rows = read_corpus_rows(corpus, ids)
        summary[name] = {}
        for method in evaluations[0]["results"]:
            combined = gnn.aggregate_model_seeds([evaluation["results"][method] for evaluation in evaluations], ids)
            if method in ["raw", "snap", "better"] and corpus != "ian":
                combined["label_summary"] = gnn.label_summary(combined["records"], ids)
            combined["by_n"], combined["by_type"] = {}, {}
            combined["label_by_n"], combined["label_by_type"] = {}, {}
            for field, destination in [("n", "by_n"), ("type", "by_type")]:
                for value in sorted({row[field] for row in rows}):
                    selected = {row["id"] for row in rows if row[field] == value}
                    combined[destination][str(value)] = gnn.metric_summary(
                        [row for row in combined["records"] if row["id"] in selected], sorted(selected))
                    if method in ["raw", "snap", "better"] and corpus != "ian":
                        combined["label_" + destination][str(value)] = gnn.label_summary(
                            [row for row in combined["records"] if row["id"] in selected], sorted(selected))
            summary[name][method] = combined
    paper_rows = read_corpus_rows("ian", list(range(20)))
    records = []
    for row in paper_rows:
        point = np.asarray(row["paper_prediction_raw"])
        m = len(row["edges"])
        phase = np.r_[point[:m] * row["w_sorted"], point[m:]]
        records.append({"id": row["id"], "n": 8, "phase_candidates": [phase.tolist()]})
    paper_path = RESULTS / "gnn" / "predictions" / "paper_ian.json"
    save_json(paper_path, records)
    with modal.Volume.from_name(VOLUME_NAME).batch_upload(force=True) as batch:
        batch.put_file(paper_path, "gnn/predictions/paper_ian.json")
    paper = launch_evaluation("paper_ian", "ian", "ian", list(range(20)),
                              {"path": "gnn/predictions/paper_ian.json", "sha256": sha256(paper_path), "graphs": 20}, 50)
    original = {row["id"]: row["paper_csv_energy"] for row in paper_rows}
    require(all(abs(row["energy_before"] - original[row["id"]]) <= IAN_ENERGY_TOL
                for row in paper["results"]["raw"]), "Old paper GNN initial energies differ from Ian's CSV")
    gnn.prediction_csv(RESULTS / "gnn" / "predictions_csv" / "paper_ian_raw.csv", paper_rows, paper["results"]["raw"])
    summary["paper_ian"] = {"summary": paper["summaries"]["raw"],
                             "independent_energy_check": verify_evaluation_energies(paper_rows, paper["results"]["raw"])}
    save_json(PREFLIGHT / "paper_prediction_csv_gate.json", csv_gate(
        RESULTS / "gnn" / "predictions_csv" / "paper_ian_raw.csv", list(range(20))))
    path = RESULTS / "gnn" / "final_results.json"
    save_json(path, {"strategy": training["bakeoff_strategy"], "sets": summary,
                     "floor_recognition_assumption": "Every method recognizes the independent run_filter floor when it is reached",
                     "random_cost_zero_hit_policy": "R times mean random cost is a conservative lower bound when no random start hits"})
    return summary


# Read optional evidence without inventing a value for an unfinished stage.
def read_evidence(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# Follow current jobs from launch/result records instead of polling an obsolete app id.
def experiment_progress():
    active_training, active_evaluation, seeds = [], [], {}
    for path in (PREFLIGHT / "training_launches").glob("*.json"):
        launch = read_evidence(path)
        name = launch["job"]["name"]
        paid = read_evidence(PREFLIGHT / "paid_apps" / (launch["app_id"] + ".json")) or {}
        if not (RESULTS / "gnn" / "training_jobs" / (name + ".json")).exists() and not paid.get("stopped_epoch"):
            active_training.append({"name": name, "app_id": launch["app_id"],
                                    "target_epoch": launch["job"]["target_epoch"],
                                    "age_seconds": time.time() - launch["created_epoch"]})
    for path in (PREFLIGHT / "evaluation_launches").glob("*.json"):
        name = path.stem
        launch = read_evidence(path)
        paid = read_evidence(PREFLIGHT / "paid_apps" / (launch["app_id"] + ".json")) or {}
        if (RESULTS / "gnn" / "evaluations" / (name + ".json")).exists() or paid.get("stopped_epoch"):
            continue
        batches = list((RESULTS / "gnn" / "evaluations" / name).glob("batch_*.json"))
        active_evaluation.append({"name": name, "app_id": launch["app_id"],
                                  "received_graphs": sum(len(read_evidence(p)["oracle_hits"]) for p in batches),
                                  "expected_graphs": sum(len(job["ids"]) for job in launch["jobs"])})
    paths = set((RESULTS / "gnn").glob("curve*_s*_progress.json")) | set((RESULTS / "gnn").glob("final_seed*_progress.json"))
    for path in sorted(paths):
        value = read_evidence(path)
        if value.get("checks"):
            seeds[path.stem] = {"latest_check": value["checks"][-1], "best_epoch": value["best"]["epoch"] if value.get("best") else None,
                                "stale_checks": value.get("stale", 0)}
    return {"captured_epoch": time.time(), "main": read_evidence(PREFLIGHT / "continuation_status.json"),
            "comparison": read_evidence(PREFLIGHT / "learning_curve_continuation_status.json"),
            "active_training": active_training, "active_evaluation": active_evaluation, "seed_checks": seeds}


# Inspect committed corpus progress without allocating a compute container.
def corpus_progress(mode="full"):
    import modal
    from concurrent.futures import ThreadPoolExecutor
    launch = read_evidence(PREFLIGHT / (mode + "_corpus_launch.json"))
    require(launch is not None, "The requested corpus has not launched")
    volume = modal.Volume.from_name(VOLUME_NAME)
    unavailable = []

    # Treat an uncommitted worker snapshot as unknown progress.
    def read_batch(job):
        prefix = "worker_proofs/%s/batch_%04d/" % (mode, job["batch"])
        for name in ["result.json", "progress.json"]:
            try:
                data = b"".join(volume.read_file(prefix + name))
                value = json.loads(data.decode("utf-8"))
                if name == "result.json":
                    value["completed_records"] = sum(item["records"] for item in value["finished"])
                    value["usable_records"] = sum(item["usable"] for item in value["finished"])
                return value
            except (FileNotFoundError, modal.exception.NotFoundError):
                pass
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                unavailable.append({"path": prefix + name, "reason": str(error),
                                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        batches = [value for value in pool.map(read_batch, launch["jobs"]) if value is not None]
    remaining = modal.Queue.from_name(launch["budget_queue"]).len()
    summary = {"expected_batches": len(launch["jobs"]), "visible_batches": len(batches),
               "expected_records": len(launch["expected"]),
               "completed_records": sum(item["completed_records"] for item in batches),
               "usable_records": sum(item["usable_records"] for item in batches),
               "initial_tokens": launch["initial_tokens"], "remaining_tokens": remaining,
               "reserved_compute_spent_upper_bound_usd": (launch["initial_tokens"] - remaining)
                   * launch["token_seconds"] * launch["token_rate_usd_per_second"]}
    result = {"captured_epoch": time.time(), "mode": mode, "summary": summary, "batches": batches,
              "unavailable_status_files": unavailable}
    save_json(PREFLIGHT / (mode + "_progress_snapshot.json"), result)
    return result


# Save expected families and audit ids independently of which solves produced a record.
def expected_graph_metadata(mode, config):
    path = PREFLIGHT / (mode + "_expected_graph_metadata.json")
    dataset = configured_dataset(mode, config)
    expected = expected_manifest(dataset.selected(mode))
    config_hash = sha256(PREFLIGHT / "frozen_plan.json")
    result = read_evidence(path)
    if result is None:
        values = {}
        for index, (graph_id, n) in enumerate(expected.items(), 1):
            kind, edges, weights, audit = dataset.build(graph_id, n)
            values[graph_id] = {"n": n, "type": kind, "audit": bool(audit)}
            if index % 10000 == 0:
                LOG.info("Expected %s graph metadata: %d/%d", mode, index, len(expected))
        result = {"mode": mode, "frozen_config_sha256": config_hash,
                  "input_hashes": EXPECTED_HASHES, "graphs": values}
        save_json(path, result)
    require(result["frozen_config_sha256"] == config_hash and result["input_hashes"] == EXPECTED_HASHES,
            "Expected graph metadata uses a different frozen plan or source")
    values = {int(key): value for key, value in result["graphs"].items()}
    require(set(values) == set(expected) and all(values[key]["n"] == n for key, n in expected.items()),
            "Expected graph metadata omits or changes an expected id")
    return values


# Report saved-record quality separately from the original generation coverage.
def describe_corpus(mode, config):
    from collections import Counter
    import numpy as np
    dataset = configured_dataset(mode, config)
    expected = expected_manifest(dataset.selected(mode))
    metadata = expected_graph_metadata(mode, config)
    folder = RESULTS / ("dataset" if mode == "full" else "dataset_test")
    records = load_records(folder, expected)
    saved_population = mode == "full" and (PREFLIGHT / "training_population.json").exists()
    population = {key: expected[key] for key in records} if saved_population else expected
    groups = {"n": {}, "type": {}}
    causes, excluded = Counter(), []
    for graph_id, n in expected.items():
        row = records.get(graph_id)
        kind = metadata[graph_id]["type"]
        if row is not None:
            require(row["type"] == kind, "A solved graph family differs from its expected generator")
            require("audit" not in row or row["audit"] == metadata[graph_id]["audit"],
                    "A solved graph audit flag differs from its expected manifest")
        good = usable(row, n) if has_pick(row, n) else False
        for field, value in [("n", str(n)), ("type", kind)]:
            group = groups[field].setdefault(value, {"expected": 0, "usable": 0, "planned": 0, "saved": 0})
            group["planned"] += 1
            group["saved"] += int(row is not None)
            group["expected"] += int(graph_id in population)
            group["usable"] += int(good)
        if saved_population and row is None:
            continue
        if good:
            continue
        reasons = []
        if row is None:
            reasons.append("missing")
        else:
            if not has_pick(row, n):
                reasons.append("pickless")
            if row.get("confirmed") is not True:
                reasons.append("not_confirmed")
            if row.get("filter") != 1:
                reasons.append("filter_not_one")
            if row.get("groups") != 1:
                reasons.append("groups_not_one")
            if len(set(row.get("seeds", []))) < 2:
                reasons.append("fewer_than_two_seeds")
            if row.get("audit") and row.get("audit_same") is not True:
                reasons.append("audit_disagreement")
            if row.get("grad_norm") is not None and row["grad_norm"] > 1e-6:
                reasons.append("gradient_above_tolerance")
            if row.get("floor") is not None and row.get("energy") is not None:
                if abs(row["energy"] - row["floor"]) > ENERGY_TOL:
                    reasons.append("energy_off_floor")
            if row.get("skipped"):
                reasons.append(str(row["skipped"]))
        if not reasons:
            reasons.append("other_usable_guard")
        causes.update(reasons)
        excluded.append({"id": graph_id, "n": n, "type": kind, "causes": reasons})
    good = sum(usable(row, expected[key]) for key, row in records.items())
    for field in groups:
        require(sum(item["expected"] for item in groups[field].values()) == len(population), "Dataset report denominator differs")
        require(sum(item["usable"] for item in groups[field].values()) == good, "Independent dataset usable counts differ")
    require(good == len(population) - len(excluded), "Dataset exclusion count differs")
    all_audited = [key for key, value in metadata.items() if value["audit"]]
    audited = [key for key in all_audited if key in population]
    result = {"expected": len(population), "planned": len(expected), "present": len(records), "usable": good,
              "not_run": len(expected) - len(records), "not_run_ids": sorted(set(expected) - set(records)),
              "quality_denominator": "saved records" if saved_population else "all expected ids",
              "usable_fraction": good / len(population), "by_n": groups["n"], "by_type": groups["type"],
              "causes_overlap": dict(causes), "excluded": excluded,
              "audited": len(audited), "audit_agreed": sum(has_pick(records.get(key), expected[key])
                  and records[key].get("audit_same") is True for key in audited),
              "expected_audit_ids": audited, "planned_audits": len(all_audited),
              "audits_not_run": len(set(all_audited) - set(records))}
    save_json(RESULTS / "gnn" / (mode + "_dataset_statistics.json"), result)
    return result


# Reconcile app and storage charges while keeping unreported usage distinct from the bill.
def write_billing_summary(refresh=True):
    from collections import defaultdict
    from decimal import Decimal
    billing = reconciled_spend_reservation() if refresh else read_evidence(PREFLIGHT / "billing_reconciliation.json")
    require(billing is not None, "No billing evidence exists")
    grouped = defaultdict(list)
    for row in billing["billing_rows"]:
        grouped[row["app_id"]].append(row)
    entries = []
    gpu_rates = {"L4": .8, "A10G": 1.1, "L40S": 1.95, "T4": .59, "A100": 2.1, "A100-40GB": 2.1}
    for app_id, rows in sorted(grouped.items()):
        cost = math.fsum(float(row["cost_usd"]) for row in rows)
        exact = sum((Decimal(row["cost_usd"]) for row in rows), Decimal(0))
        require(abs(cost - float(exact)) <= 1e-8, "Independent report billing sum differs")
        resources = {}
        for row in rows:
            for name, value in row["cost_by_resource"].items():
                resources[name] = resources.get(name, Decimal(0)) + Decimal(value)
        gpu_hours = {name: float(value) / gpu_rates[name] for name, value in resources.items()
                     if name in gpu_rates and value > 0}
        entries.append({"app_id": app_id, "name": billing["apps"].get(app_id, {}).get("launch", rows[0]["description"]),
                        "date_utc": rows[0]["interval"][:10], "cost_usd": cost,
                        "physical_core_hours": float(resources.get("CPU", 0)) / .0473,
                        "memory_gib_hours": float(resources.get("Memory", 0)) / .008,
                        "l4_hours": float(resources.get("L4", 0)) / .8,
                        "gpu_hours_by_type": gpu_hours,
                        "cost_by_resource": {key: str(value) for key, value in resources.items()}})
    require(abs(math.fsum(item["cost_usd"] for item in entries) - billing["reported_app_cost_usd"]) <= 1e-8,
            "Final ledger sum differs from workspace billing")
    result = {"entries": entries, "reported_usd": billing["reported_app_cost_usd"],
              "unreported_reservation_usd": billing["unreported_resource_reservation_usd"],
              "overhead_reservation_usd": billing["image_and_startup_reserve_usd"],
              "upper_reservation_usd": billing["spend_reservation_usd"], "checked_epoch": billing["checked_epoch"],
              "resource_hours_source": "Resource charges divided by the verified workspace rates; includes idle time and failures"}
    save_json(RESULTS / "gnn" / "billing_summary.json", result)
    lines = ["## Latest billing reconciliation", "", "Reported charges include failed runs, probes, CPU, GPU, memory, storage and egress where Modal has posted them.", "",
             "| UTC date | App or storage | Physical core-hours | GiB-hours | GPU hours by type | Reported USD |",
             "|---|---|---:|---:|---:|---:|"]
    for row in entries:
        hours = ", ".join("%s: %.6f" % item for item in sorted(row["gpu_hours_by_type"].items())) or "none posted"
        lines.append("| %s | %s (%s) | %.6f | %.6f | %s | %.8f |" %
                     (row["date_utc"], row["name"], row["app_id"], row["physical_core_hours"], row["memory_gib_hours"], hours, row["cost_usd"]))
    lines.extend(["", "Reported total: $%.8f. Unreported usage reservation: $%.8f. Additional overhead reservation: $%.2f. Upper reservation: $%.8f."
                  % (result["reported_usd"], result["unreported_reservation_usd"], result["overhead_reservation_usd"], result["upper_reservation_usd"]),
                  "", "Modal can revise recent hourly records. Reservations are upper estimates, not final charges. Resource-hours come from the charge breakdown and the account rates."])
    path = RESULTS / "cost_ledger.md"
    history = path.read_text(encoding="utf-8").split("\n## Latest billing reconciliation", 1)[0]
    path.write_text(history.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return result


# Format a measured estimate and its graph-bootstrap interval for report tables.
def report_interval(metric, percent=False):
    if metric is None or metric.get("value") is None:
        return "not measured"
    scale = 100 if percent else 1
    suffix = "%" if percent else ""
    return ("%.*f%s [%.*f, %.*f]" % (REPORT_DECIMALS, scale * metric["value"], suffix,
            REPORT_DECIMALS, scale * metric["low95"], REPORT_DECIMALS, scale * metric["high95"]))


# Draw the label frequencies that determine the discrete prediction heads.
def draw_label_frequencies(labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = RESULTS / "gnn" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    theta = labels["theta_counts"]
    positive = sum(count for value, count in theta.items() if float(value) > 0)
    theta_counts = [labels["theta_negative_pi4"], theta["0.0"], positive,
                    labels["edge_count"] - labels["theta_negative_pi4"] - sum(theta.values())]
    beta_counts = list(labels["beta_counts"].values())
    beta_counts.append(labels["node_count"] - sum(beta_counts))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for axis, counts, total, ticks, title in [
            (axes[0], theta_counts, labels["edge_count"], ["−π/4", "0", "+π/4", "Other"], "Edge phases θ"),
            (axes[1], beta_counts, labels["node_count"], ["−π/4", "−π/8", "0", "+π/8", "+π/4", "Other"], "Node angles β")]:
        values = [100 * count / total for count in counts]
        bars = axis.bar(ticks, values, color="#2463a2", width=.7)
        axis.bar_label(bars, labels=[format(count, ",") for count in counts], padding=3, fontsize=8)
        axis.set(title=title + " (%s labels)" % format(total, ","), ylabel="Labels (%)", ylim=(0, max(values) * 1.18))
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("Frozen usable training population: 63,288 graphs\nCounts within 0.001 radians of each listed value")
    for extension in ["png", "pdf"]:
        fig.savefig(folder / ("label_frequencies." + extension), dpi=180)
    plt.close(fig)


# Draw the three requested scientific figures from held-out graph aggregates.
def draw_result_figures(final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = RESULTS / "gnn" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    methods = ["raw", "snap", "better", "random", "zero", "oracle"]
    labels = {"raw": "Model raw", "snap": "Model snapped", "better": "Model better", "random": "Random", "zero": "Zero + noise", "oracle": "Oracle"}
    paths = []
    for metric, name, ylabel in [("hit_rate", "hit_rate_by_n", "Polish hit rate (%)"),
                                 ("geometric_mean_cost_ratio", "cost_ratio_by_n", "Geometric mean expected cost / random")]:
        fig, axis = plt.subplots(figsize=(9, 5.4), layout="constrained")
        for method in methods:
            points = []
            for tag in ["test_split", "large"]:
                for size, summary in final["sets"][tag][method]["by_n"].items():
                    points.append((int(size), summary["metrics"][metric]))
            points.sort()
            x = np.asarray([item[0] for item in points])
            scale = 100 if metric == "hit_rate" else 1
            y = scale * np.asarray([item[1]["value"] for item in points])
            low = scale * np.asarray([item[1]["low95"] for item in points])
            high = scale * np.asarray([item[1]["high95"] for item in points])
            line = axis.plot(x, y, marker="o", label=labels[method])[0]
            axis.fill_between(x, low, high, color=line.get_color(), alpha=.10)
        axis.set(xlabel="Nodes", ylabel=ylabel, xticks=list(range(5, 15)))
        if metric != "hit_rate":
            axis.set_yscale("log")
            axis.axhline(1, color="black", linewidth=.7, alpha=.4)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, ncols=2)
        for extension in ["png", "pdf"]:
            path = folder / (name + "." + extension)
            fig.savefig(path, dpi=180)
            paths.append(path)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for axis, tag, title in zip(axes, ["test_split", "large", "ian"], ["Held-out n=5–12", "Larger n=13–14", "Ian's 20 graphs"]):
        for method in [final["strategy"], "random", "zero", "oracle"]:
            values = [row["energy_ratio_before"] for row in final["sets"][tag][method]["records"]]
            axis.hist(values, bins=np.linspace(0, 1.05, 43), density=True, histtype="step", label=labels[method])
        axis.set(xlabel="Mean initial energy / reference energy", ylabel="Graph density", title=title)
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    for extension in ["png", "pdf"]:
        path = folder / ("initial_energy_distribution." + extension)
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


# Render a complete metric table with confidence intervals and explicit random-zero-hit counts.
def append_metric_table(lines, label, methods):
    lines.extend(["", "### " + label, "",
        "95% intervals resample graphs. Model values average the three seeds within each graph.", "",
        "| Method | Graphs | Hit | Median cost/random | Geometric mean cost/random | Initial E/floor | Final approximation | Mean nfev | Random zero-hit graphs |",
        "|---|---:|---|---|---|---|---|---|---:|"])
    for name, summary in methods.items():
        metrics = summary["metrics"]
        values = [report_interval(metrics[key], key == "hit_rate") for key in
                  ["hit_rate", "median_cost_ratio", "geometric_mean_cost_ratio", "mean_energy_ratio_before", "mean_approximation_ratio_after", "mean_nfev"]]
        lines.append("| %s | %d | %s | %d |" % (name, summary["graphs"], " | ".join(values), summary["zero_random_hit_graphs"]))


# Complete Ian's reference-angle tables from already evaluated predictions.
def ian_label_statistics():
    import gnn
    folder = RESULTS / "gnn"
    paths = {str(seed): folder / "evaluations" / ("final_s%d_ian.json" % seed) for seed in gnn.TRAIN_SEEDS}
    paths["paper"] = folder / "evaluations" / "paper_ian.json"
    require(all(path.exists() for path in paths.values()), "Ian label tables need every completed model-seed and paper evaluation")
    sources = {name: sha256(path) for name, path in paths.items()}
    sources.update({"model_source": sha256(ROOT / "src" / "gnn.py"),
                    "corpus_manifest": sha256(folder / "corpus_manifest.json")})
    output = folder / "ian_label_statistics.json"
    cached = read_evidence(output)
    if cached is not None and cached["source_hashes"] == sources:
        return cached
    ids = list(range(20))
    rows = {row["id"]: row for row in read_corpus_rows("ian", ids)}
    evaluated = {name: read_evidence(path) for name, path in paths.items()}
    require(all(value["expected_ids"] == ids for value in evaluated.values()), "Ian label tables change the twenty-row manifest")
    methods = {}
    for method in ["raw", "snap", "better"]:
        seed_records = []
        for seed in gnn.TRAIN_SEEDS:
            records = []
            for record in evaluated[str(seed)]["results"][method]:
                phase = record["polishes"][0]["candidate_phase"]
                records.append(dict(record, label_metrics=gnn.label_metrics(rows[record["id"]], phase)))
            seed_records.append(records)
        combined = gnn.aggregate_model_seeds(seed_records, ids)["records"]
        methods[method] = {"summary": gnn.label_summary(combined, ids), "by_type": {}}
        for kind in sorted({row["type"] for row in rows.values()}):
            selected = [i for i in ids if rows[i]["type"] == kind]
            methods[method]["by_type"][kind] = gnn.label_summary([row for row in combined if row["id"] in selected], selected)
    paper = []
    for record in evaluated["paper"]["results"]["raw"]:
        phase = record["polishes"][0]["candidate_phase"]
        paper.append(dict(record, label_metrics=gnn.label_metrics(rows[record["id"]], phase)))
    methods["old paper raw"] = {"summary": gnn.label_summary(paper, ids)}
    result = {"source_hashes": sources, "expected_ids": ids, "seeds": list(gnn.TRAIN_SEEDS), "methods": methods,
              "reference": "The single recorded run_filter floor pick for each unweighted Ian graph. Symmetries and multiple minima can give other good angle vectors; label mismatch does not imply an energy failure. Some reference coordinates are continuous, so snapping can change even an exact reference point.",
              "new_paid_compute": False}
    save_json(output, result)
    return result


# Compare the two completed regression-head ablations on matched validation graphs.
def label_ablation_comparison():
    import gnn
    folder = RESULTS / "gnn"
    paths = {name: folder / "evaluations" / (name + "_s42_val.json") for name in ["B", "E"]}
    paths.update({name + "_training": folder / "training_jobs" / (name + "_s42_e0040.json") for name in ["B", "E"]})
    if not all(path.exists() for path in paths.values()):
        return None
    sources = {name: sha256(path) for name, path in paths.items()}
    output = folder / "label_ablation.json"
    cached = read_evidence(output)
    if cached is not None and cached["source_hashes"] == sources:
        return cached
    values = {name: read_evidence(path) for name, path in paths.items()}
    left, right = values["B_training"], values["E_training"]
    require(left["training_ids_sha256"] == right["training_ids_sha256"]
            and left["training_graphs"] == right["training_graphs"] and left["epoch"] == right["epoch"] == 40
            and left["source_hashes"]["src/gnn.py"] == right["source_hashes"]["src/gnn.py"],
            "Label ablation uses different training data, epochs or model source")
    for key in ["baseline", "classification", "hypotheses", "seed", "maximum_epochs", "hidden", "layers"]:
        require(left["config"].get(key) == right["config"].get(key), "Label ablation changes encoder or head configuration")
    require(not left["config"]["classification"] and left["config"].get("label_weight", 1) == 1
            and left["config"].get("energy_weight", 0) == 0
            and right["config"]["label_weight"] == 0 and right["config"]["energy_weight"] == 1,
            "Label ablation is not the supervised-regression versus energy-only comparison")
    expected = read_evidence(folder / "splits.json")["ids"]["val_selection"]
    require(all(values[name]["expected_ids"] == expected for name in ["B", "E"]), "Label ablation changes validation ids")
    strategies = {name: validation_strategy(values[name])["chosen"] for name in ["B", "E"]}
    paired = gnn.choose_variant({name: values[name]["results"][strategies[name]] for name in ["B", "E"]}, {"B": 0, "E": 1})
    comparison = paired["paired_log_cost_comparisons"]["B" if paired["lowest_observed_cost"] == "E" else "E"]
    if paired["lowest_observed_cost"] == "E":
        point, low, high = [math.exp(comparison[key]) for key in ["mean_log_cost_difference", "low95", "high95"]]
    else:
        point, low, high = [math.exp(-comparison[key]) for key in ["mean_log_cost_difference", "high95", "low95"]]
    independent = paired["geometric_mean_cost_ratio"]["B"] / paired["geometric_mean_cost_ratio"]["E"]
    require(math.isclose(point, independent, rel_tol=1e-12), "Independent label-ablation cost ratios differ")
    result = {"source_hashes": sources, "graphs": len(expected), "training_graphs": left["training_graphs"],
              "training_ids_sha256": left["training_ids_sha256"], "seed": 42, "epochs": 40, "strategies": strategies,
              "supervised_regression_over_energy_only": {"value": point, "low95": low, "high95": high},
              "interpretation": "Supervised regression used less validation cost" if high < 1 else
                                "Energy-only training used less validation cost" if low > 1 else "The paired interval includes equal cost",
              "limits": "One matched training seed; graph-bootstrap intervals do not estimate training-seed variability. Classification changes the heads and is reported separately.",
              "new_paid_compute": False}
    save_json(output, result)
    return result


# Analyze only saved learning-curve evaluations, without any new fits or graph searches.
def learning_curve_diagnostics(curve, final):
    import numpy as np
    import gnn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = RESULTS / "gnn"
    expected = read_evidence(folder / "splits.json")["ids"]["test"]
    method = curve["strategy"]
    records = {"100": final["sets"]["test_split"][method]["records"]}
    sources = {"gnn/final_results.json": sha256(folder / "final_results.json")}
    for percent in ["25", "50"]:
        evaluations = []
        for seed in curve["seeds"]:
            path = folder / "evaluations" / ("curve%s_s%d_test.json" % (percent, seed))
            evaluation = read_evidence(path)
            require(evaluation is not None and evaluation["expected_ids"] == expected,
                    "Learning-curve diagnostics need every fixed test id")
            evaluations.append(evaluation["results"][method])
            sources[str(path.relative_to(RESULTS)).replace("\\", "/")] = sha256(path)
        records[percent] = gnn.aggregate_model_seeds(evaluations, expected)["records"]
    indexed = {percent: {row["id"]: row for row in rows} for percent, rows in records.items()}
    require(all(len(records[p]) == len(expected) and set(indexed[p]) == set(expected) for p in indexed),
            "Learning-curve diagnostics omit or duplicate held-out graphs")
    comparisons, by_n = {}, {}
    for earlier, later in [("25", "50"), ("50", "100"), ("25", "100")]:
        differences = np.asarray([math.log(indexed[later][i]["cost_ratio"])
                                  - math.log(indexed[earlier][i]["cost_ratio"]) for i in expected])
        rng = np.random.default_rng(gnn.MODEL_SEED)
        samples = [float(np.exp(differences[rng.integers(len(expected), size=len(expected))].mean()))
                   for _ in range(gnn.BOOTSTRAP_SAMPLES)]
        low, high = np.quantile(samples, [.025, .975])
        estimate = math.exp(math.fsum(differences) / len(expected))
        comparisons[earlier + "_to_" + later] = {
            "graphs": len(expected), "later_over_earlier_cost_ratio": {
                "value": estimate, "low95": float(low), "high95": float(high)},
            "interpretation": "lower cost" if high < 1 else "higher cost" if low > 1 else "interval includes one"}
    for size in sorted({row["n"] for row in records["100"]}):
        ids = [i for i in expected if indexed["100"][i]["n"] == size]
        by_n[str(size)] = {percent: gnn.metric_summary([indexed[percent][i] for i in ids], ids)
                           for percent in ["25", "50", "100"]}
    subsets = read_evidence(folder / "learning_curve_subsets.json")["ids"]
    train = {row["id"]: row for row in read_corpus_rows("full", subsets["100"])}
    diversity = {percent: {"graphs": len(ids), "unique_unweighted_wl_groups": len({train[i]["wl_hash"] for i in ids}),
                          "by_n": {str(size): sum(train[i]["n"] == size for i in ids)
                                   for size in sorted({row["n"] for row in train.values()})}}
                 for percent, ids in subsets.items()}
    result = {"paired_comparisons": comparisons, "by_n": by_n, "training_diversity": diversity,
              "source_hashes": sources, "expected_test_ids_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
              "bootstrap_samples": gnn.BOOTSTRAP_SAMPLES,
              "protocol": "Average linear costs within graph over three model seeds, then paired graph bootstrap of mean log later/earlier cost; exploratory per-size intervals are not adjusted for multiple comparisons",
              "new_paid_compute": False, "missing_graphs_generated": False}
    save_json(folder / "learning_curve_diagnostics.json", result)
    figure_folder = folder / "figures"
    figure_folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    percentages = ["25", "50", "100"]
    x = [curve["fractions"][p]["graphs"] for p in percentages]
    for axis, metric, label, scale in [(axes[0], "geometric_mean_cost_ratio", "Geometric mean per-graph cost / random", 1),
                                       (axes[1], "hit_rate", "Polish hit rate (%)", 100)]:
        points = [curve["fractions"][p]["summary"]["metrics"][metric] for p in percentages]
        axis.plot(x, [scale * point["value"] for point in points], marker="o", color="#2463a2")
        axis.fill_between(x, [scale * point["low95"] for point in points],
                          [scale * point["high95"] for point in points], color="#2463a2", alpha=.15)
        axis.set(xlabel="Training graphs", ylabel=label, xticks=x,
                 xticklabels=["%s%%\n%s" % (p, format(n, ",")) for p, n in zip(percentages, x)])
        axis.grid(alpha=.2)
    axes[0].set_yscale("log")
    axes[0].axhline(1, color="black", linewidth=.7, alpha=.4)
    fig.suptitle("Winning model: fixed held-out graphs, three training seeds\nShading: 95% graph-bootstrap interval")
    for extension in ["png", "pdf"]:
        fig.savefig(figure_folder / ("learning_curve." + extension), dpi=180)
    plt.close(fig)
    return result


# Write measured results and an unsent email draft for either a completed or halted run.
def write_research_report(reason=None, refresh_billing=True):
    import gnn
    folder = RESULTS / "gnn"
    folder.mkdir(parents=True, exist_ok=True)
    final = read_evidence(folder / "final_results.json")
    if reason is None and final is None:
        stop = read_evidence(PREFLIGHT / "continuation_stop.json") or read_evidence(PREFLIGHT / "stop.json")
        reason = stop.get("reason") if stop else None
    billing = write_billing_summary(refresh=refresh_billing)
    config = load_frozen_plan() if (PREFLIGHT / "frozen_plan_lock.json").exists() else None
    complete = final is not None and reason is None
    curve_approval = read_evidence(PREFLIGHT / "learning_curve_approval.json")
    curve_pending = not (folder / "learning_curve.json").exists() and not (
        curve_approval is not None and curve_approval.get("deferred_by_user") is True)
    curve_authorized = curve_approval is not None and curve_approval.get("user_approved") is True
    status = "main experiment complete; training comparison pending" if complete and curve_pending else "completed" if complete else "halted" if reason else "in progress"
    lines = ["# Weighted Max-Cut MA-QAOA", "", "Status: " + status + ".", ""]
    if reason:
        lines.extend(["I stopped further compute because: " + str(reason), ""])
    if complete:
        chosen = final["strategy"]
        point = final["sets"]["test_split"][chosen]["summary"]["metrics"]["geometric_mean_cost_ratio"]
        lines.append("The selected model uses %s post-processing. Its held-out geometric mean cost/random ratio is %s. Values below one save circuit evaluations." % (chosen, report_interval(point)))
        if point["value"] >= 1:
            lines.append("The model did not beat random starts on the headline held-out expected-cost metric.")
        elif point["high95"] >= 1:
            lines.append("The point estimate favours the model, but the graph-bootstrap interval includes one. This run does not establish a reduction in expected circuit evaluations.")
        else:
            lines.append("The model reduced expected circuit evaluations on this held-out set, and the graph-bootstrap interval lies below one.")
        if curve_pending:
            lines.append("The main experiment is complete. The approved 25%/50% training comparison is still pending, and the 100% final fits are available to reuse. This delivery is incomplete for that comparison." if curve_authorized else
                         "The main experiment is complete. The additional 25%/50% fits remain blocked pending clarification of the user's scheduling instruction; automatic approval review rejected including them in this launch. The 100% final fits are available to reuse. This delivery is incomplete for the requested training comparison.")
    else:
        lines.append("I have not measured a final GNN warm-start result. I make no claim about model savings or larger-size generalization.")
    lines.extend(["", "## Gates", "", "Training data quality uses the user-approved saved-record denominator. The frozen usable ids define the training population; 36,328 ungenerated training-size ids are not run. Other gates retain their complete expected-id denominators, including missing and pickless records.", "",
                  "| Gate | Measured result | Requirement | Evidence |", "|---|---|---|---|"])
    zero = read_evidence(PREFLIGHT / "stage_zero.json")
    if zero:
        lines.append("| Ian CSV energy and order | %d rows; maximum energy error %.17g; traversal order checked | error ≤1e-12; all orders correct | [Stage 0](../preflight/stage_zero.json) |" % (zero["original_csv"]["rows"], zero["original_csv"]["max_energy_error"]))
    builds = read_evidence(PREFLIGHT / "reference_build.json")
    if builds:
        lines.append("| Reference graph reconstruction | %d/%d exact edges and weights | 21/21 | [Build check](../preflight/reference_build.json) |" % (sum(row["passed"] for row in builds["results"]), builds["expected"]))
    references = read_evidence(PREFLIGHT / "reference_results.json")
    if references:
        reference_source = {row["id"]: row for row in read_evidence(REFERENCE_TARGET)["graphs"]}
        floor_error = max(abs(row["record"]["floor"] - reference_source[row["id"]]["floor"]) for row in references)
        distance = max(row["pick_distance"] for row in references)
        lines.append("| Reference solves | %d/12 passed; maximum floor error %.17g; maximum pick distance %.17g | floor ≤1e-6; distance ≤1e-3 | [Reference results](../preflight/reference_results.json) |" % (sum(row["passed"] for row in references), floor_error, distance))
    pilot = read_evidence(RESULTS / "dataset_pilot" / "runner_gates.json")
    if pilot:
        for name, label, threshold in [("usable", "Pilot usable", "≥95%; target ≥97%"), ("audit", "Pilot audit agreement", "≥98%"), ("compare", "STALL 100/400 agreement", "≥98%, or costed STALL 400")]:
            value = pilot[name]
            lines.append("| %s | %d/%d (%.6f%%) | %s | [Pilot gates](../dataset_pilot/runner_gates.json) |" % (label, value["passed"], value["expected"], 100 * value["fraction"], threshold))
    for name, label in [("pilot_csv_gate.json", "Pilot CSV"), ("pilot400_csv_gate.json", "Pilot400 CSV"), ("full_corpus_csv_gate.json", "Full CSV"), ("test_corpus_csv_gate.json", "Larger-test CSV"), ("full_partial_csv_gate.json", "Saved full-size rows CSV, partial"), ("test_partial_csv_gate.json", "Saved larger-size rows CSV, partial")]:
        value = read_evidence(PREFLIGHT / name)
        lines.append("| %s | %s | correct traversal order; error ≤1e-10 | %s |" % (label,
                     "%d rows; max error %.17g" % (value["rows"], value["max_energy_error"]) if value else "not measured",
                     "[Checker evidence](../preflight/%s)" % name if value else "pending"))
    smoke = read_evidence(PREFLIGHT / "test_smoke_gate.json")
    if smoke:
        lines.append("| Runner batch completion test | %d assigned graphs saved; %d/%d batch completions returned; %d total test rows retained | every assigned graph and batch completes | [Paid runner check](../preflight/test_smoke_gate.json) |" %
                     (len(smoke["expected_ids"]), smoke["returned_batches"], smoke["expected_batches"], smoke["saved_total"]))
    statistics = {}
    if config:
        lines.extend(["| Frozen plan | %d training-size and %d larger-test graphs; shrink factor %.10f; STALL %d | counts fixed before full generation | [Plan](../preflight/frozen_plan.json) |" %
                     (sum(config["plan_counts"].values()), sum(config["test_counts"].values()), config["uniform_factor"], config["stall"])])
        for mode in ["full", "test"]:
            if (RESULTS / ("dataset" if mode == "full" else "dataset_test")).exists():
                stats = describe_corpus(mode, config)
                statistics[mode] = stats
                lines.append("| %s usable | %d/%d (%.6f%%) | ≥95%% overall | [Counts](%s_dataset_statistics.json) |" % (mode, stats["usable"], stats["expected"], 100 * stats["usable_fraction"], mode))
    for path, label in [(PREFLIGHT / "ian_floor_gate.json", "Ian floor list"), (PREFLIGHT / "stage3_gate.json", "Label energy, box, exact cuts and split"), (folder / "torch_energy_check.json", "Local torch energy and gradient"), (folder / "torch_energy_check_cuda.json", "GPU torch energy and gradient")]:
        value = read_evidence(path)
        measured = "not measured"
        if value:
            if "max_gradient_error" in value:
                measured = "%d points; maximum energy/gradient errors %.17g / %.17g" % (value["points"], value["max_energy_error"], value["max_gradient_error"])
            elif "max_label_floor_error" in value:
                measured = "%d graphs; max label-floor/cut errors %.17g / %.17g; box violations %d; split counts %s" % (value["verified_graphs"], value["max_label_floor_error"], value["max_independent_cut_error"], value["box_violations"], json.dumps(value["split_counts"], sort_keys=True))
            else:
                measured = json.dumps(value, sort_keys=True).replace("|", "/")
        lines.append("| %s | %s | all specified checks pass | %s |" % (label, measured, "[Evidence](%s)" % os.path.relpath(path, folder).replace("\\", "/") if value else "pending"))
    oracle_seen = False
    for path in sorted((folder / "evaluations").glob("*.json")):
        value = read_evidence(path)
        if "oracle_gate" in value:
            oracle_seen = True
            gate = value["oracle_gate"]
            lines.append("| Oracle %s | %d/%d (%.6f%%) | ≥99%% | [Evaluation](evaluations/%s) |" % (path.stem, gate["passed"], gate["expected"], 100 * gate["fraction"], path.name))
    if not oracle_seen:
        lines.append("| Oracle polish success | pending | ≥99% over the complete evaluation manifest | no completed model evaluation yet |")
    lines.append("| Spend limit | $%.8f posted; $%.8f upper reservation | actual below $280; total projection ≤$300 | [Billing](billing_summary.json) |" % (billing["reported_usd"], billing["upper_reservation_usd"]))
    if config:
        lines.append("| Original 100,000-graph forecast (historical) | $%.8f original reservation | superseded by the approved saved-population scope; hard cap $300 | [Original frozen plan](../preflight/frozen_plan.json) |" % config["cost"]["projected_total_usd"])
    lines.extend(["", "The torch numerical requirements are energy error ≤1e-10 and phase-gradient error ≤1e-8 on all 1,000 points.", "", "## Dataset", ""])
    for mode, stats in statistics.items():
        lines.extend(["### " + mode, "", "I retained %d of %d %s as usable. Of %d originally planned ids, %d are not run. The archive keeps rejected saved records and their evidence." % (stats["usable"], stats["expected"], stats["quality_denominator"], stats["planned"], stats["not_run"]), "",
                      "| Nodes | Quality denominator | Usable | Usable % |", "|---:|---:|---:|---:|"])
        for size, value in sorted(stats["by_n"].items(), key=lambda pair: int(pair[0])):
            lines.append("| %s | %d | %d | %.6f |" % (size, value["expected"], value["usable"], 100 * value["usable"] / value["expected"]))
        lines.extend(["", "| Graph family | Expected | Usable | Usable % |", "|---|---:|---:|---:|"])
        for kind, value in sorted(stats["by_type"].items()):
            lines.append("| %s | %d | %d | %.6f |" % (kind, value["expected"], value["usable"], 100 * value["usable"] / value["expected"]))
        lines.extend(["", "I observed %d agreeing audits out of %d audit ids in the quality population. There are %d designated audits among ids not run." % (stats["audit_agreed"], stats["audited"], stats["audits_not_run"]), "",
                      "Exclusion causes can overlap: " + json.dumps(stats["causes_overlap"], sort_keys=True) + ". See the counts JSON for each excluded id."])
    bias = read_evidence(PREFLIGHT / "saved_data_bias_check.json")
    if bias:
        lines.extend(["", "### Interruption bias check", "", "This check used saved files only and launched no Modal compute. The pilot is the complete random sample of 40 graphs per n. Intervals below independently bootstrap both timing samples (5,000 resamples). Timing is a difficulty proxy; this check cannot establish that every graph property is representative.", "",
                      "| n | Saved graphs | Completed mean seconds | Pilot mean seconds | Mean ratio [95% interval] |",
                      "|---:|---:|---:|---:|---|"])
        for n, item in bias["by_n"].items():
            lo, hi = item["bootstrap_mean_ratio_95ci"]
            lines.append("| %s | %d | %.4f | %.4f | %.4f [%.4f, %.4f] |" % (n, item["saved"], item["saved_mean_secs_main"], item["pilot_mean_secs_main"], item["mean_ratio"], lo, hi))
        shards = bias["shards"]
        lines.extend(["", "%d missing ids belong to %d wholly unstarted shards. The remaining %d are ascending-id tails of %d partial shards; every partial shard is an exact prefix. %d shards completed. Larger sizes are underrepresented because they occur later in each shard. The within-size comparisons do not show clear selection of easy graphs: n=5 has almost identical medians and 90th percentiles despite a lower mean, while n=6 is slower than the pilot. Family-level means and counts are retained in the evidence; several pilot family cells are small. I proceeded under the user's bias rule." % (shards["missing_in_whole_shards"], shards["unstarted"], shards["missing_in_started_shard_tails"], shards["partial"], shards["complete"]), "",
                      "The frozen training population contains 63,288 usable ids. The original 100,000-id recipe remains unchanged so graph identities do not move. [Population and row hashes](../preflight/training_population.json); [bias evidence](../preflight/saved_data_bias_check.json)."])
    comparison = read_evidence(RESULTS / "dataset_pilot" / "runner_compare.json")
    if comparison:
        disagreements = [row for row in comparison if not row["same"]]
        lines.extend(["", "Pilot stopping-rule disagreements: %d." % len(disagreements)])
        for row in disagreements:
            lines.append("id %d, n=%d, %s: distance %s." % (row["id"], row["n"], row["type"], row["distance"]))
    labels = read_evidence(folder / "label_statistics.json") or read_evidence(folder / "label_statistics_preview.json")
    if labels:
        lines.extend(["", "### Labels", "", "I counted %d edge phases and %d node angles. I found %d box violations. Values count as discrete when they lie within 1e-3 radians." % (labels["edge_count"], labels["node_count"], labels["box_violations"]), "",
                      "| Angle family | Value | Count | Fraction |", "|---|---:|---:|---:|"])
        for family, values, total in [("theta", labels["theta_counts"], labels["edge_count"]), ("beta", labels["beta_counts"], labels["node_count"])]:
            for angle, count in values.items():
                lines.append("| %s | %s | %d | %.8f |" % (family, angle, count, count / total))
        lines.extend(["", "Theta near −π/4: %d. Theta outside the 0/+π/4 classes: %.8f. Beta outside the five classes: %.8f." % (labels["theta_negative_pi4"], labels["theta_other_fraction"], labels["beta_other_fraction"])])
        positive = sum(count for angle, count in labels["theta_counts"].items() if float(angle) > 0)
        lines.append("The fraction of theta within 1e-3 of either sign of π/4 is %.8f." % ((positive + labels["theta_negative_pi4"]) / labels["edge_count"]))
        if "box_tolerance_units" in labels:
            lines.append("Box tolerance: " + labels["box_tolerance_units"] + ". A fixed 1e-6 tolerance in phase units would flag %d coordinates. The largest phase excess is %.12g, and the largest excess in raw gamma is %.12g. Correcting the tolerance units changed no labels or ids." % (labels["fixed_phase_tolerance_violations"], labels["max_phase_box_excess"], labels["max_raw_gamma_box_excess"]))
        lines.append("Node class indices 0–4 correspond to β = −π/4, −π/8, 0, +π/8 and +π/4. The leaf_sign_accuracy metric measures sign accuracy among labels near β = ±π/4. A class with no reference examples is reported as not measured.")
        draw_label_frequencies(labels)
        lines.extend(["", "![Label frequencies in the frozen usable population](figures/label_frequencies.png)"])
    split = read_evidence(folder / "splits.json")
    if split:
        lines.extend(["", "### Split", "", "I grouped unweighted Weisfeiler–Lehman hashes before splitting. Actual counts are %s. Seed %d fixes the split and the 2,000-graph validation subset. I forced %d weighted graphs with Ian-test hashes into the test split. WL hashes can merge distinct regular graphs; this grouping prevents leakage conservatively." % (json.dumps(split["counts"], sort_keys=True), split["seed"], split["forced_test_graphs"])])
    curve = read_evidence(folder / "learning_curve.json")
    if curve:
        lines.extend(["", "## Training data learning curve", "", curve["protocol"] + ". The table uses the geometric mean of per-graph estimated circuit-cost ratios; smaller is better. The aggregate-work comparison below separately reports the ratio of summed estimated costs. These fits do not generate any of the 36,328 unrun graphs.", "",
                      "| Training split fraction | Training graphs | Held-out cost/random [95% interval] | Held-out hit [95% interval] |",
                      "|---:|---:|---|---|"])
        for percent in ["25", "50", "100"]:
            value = curve["fractions"][percent]
            metrics = value["summary"]["metrics"]
            lines.append("| %s%% | %d | %s | %s |" % (percent, value["graphs"], report_interval(metrics["geometric_mean_cost_ratio"]), report_interval(metrics["hit_rate"], True)))
        lines.extend(["", "Intervals resample held-out graphs after averaging model seeds within each graph. The 100% row reuses the final fits. This curve measures gains within the completed population; it cannot guarantee the value of the interrupted tails, whose size mix differs.", "",
                      "The following means and sample standard deviations summarize the three seed-level estimates. They use a different averaging order from the graph-level intervals above.", "",
                      "| Training fraction | Metric | Mean across seeds | Sample standard deviation | Seed values |", "|---|---|---:|---:|---|"])
        for percent in ["25", "50", "100"]:
            for metric, spread in curve["fractions"][percent]["seed_spread"].items():
                lines.append("| %s%% | %s | %.8f | %.8f | %s |" % (percent, metric, spread["mean"], spread["std"], json.dumps(spread["values"])))
        diagnostics = learning_curve_diagnostics(curve, final)
        lines.extend(["", "![Training data learning curve](figures/learning_curve.png)", "",
                      "Paired comparisons keep the same held-out graphs and random-start evidence. A ratio below one favours the larger training subset. These intervals resample graphs and do not include uncertainty from new training seeds.", "",
                      "| Training change | Later / earlier geometric mean cost [95% interval] | Interpretation |",
                      "|---|---|---|"])
        for change, value in diagnostics["paired_comparisons"].items():
            lines.append("| %s%% → %s%% | %s | %s |" % (*change.split("_to_"), report_interval(value["later_over_earlier_cost_ratio"]), value["interpretation"]))
        lines.extend(["", "Exploratory results by graph size use the same graph-bootstrap estimator. Intervals are not adjusted for multiple comparisons.", "",
                      "| Nodes | Held-out graphs | 25% cost/random | 50% cost/random | 100% cost/random |", "|---:|---:|---|---|---|"])
        for size, values in diagnostics["by_n"].items():
            lines.append("| %s | %d | %s | %s | %s |" % (size, values["100"]["graphs"],
                         *(report_interval(values[p]["metrics"]["geometric_mean_cost_ratio"]) for p in ["25", "50", "100"])))
        lines.extend(["", "Training diversity (WL grouping may merge distinct unweighted structures):", "",
                      "| Training fraction | Weighted graphs | Unique unweighted WL groups |", "|---|---:|---:|"])
        for percent in ["25", "50", "100"]:
            value = diagnostics["training_diversity"][percent]
            lines.append("| %s%% | %d | %d |" % (percent, value["graphs"], value["unique_unweighted_wl_groups"]))
    bakeoff = read_evidence(folder / "bakeoff.json")
    if bakeoff:
        lines.extend(["", "## Architecture selection", "", "I selected %s using only the fixed validation graphs. I selected %s post-processing. The paired bootstrap identified %s as within noise of the lowest observed cost. I used the declared simplicity order for that tie." % (bakeoff["selection"]["chosen"], bakeoff["strategy"], ", ".join(bakeoff["selection"]["within_noise"])), "",
                      "| Variant | Validation geometric mean cost/random | Hit | Post-processing |", "|---|---:|---:|---|"])
        for name, ratio in bakeoff["selection"]["geometric_mean_cost_ratio"].items():
            strategy = bakeoff["strategies"][name]["chosen"]
            evaluation = read_evidence(folder / "evaluations" / (name + "_s42_val.json"))
            metrics = evaluation["summaries"][strategy]["metrics"]
            lines.append("| %s | %s | %s | %s |" % (name, report_interval(metrics["geometric_mean_cost_ratio"]),
                                                    report_interval(metrics["hit_rate"], True), strategy))
        lines.extend(["", "A uses five GINE layers, hidden width 192, and tie-averaged canonical positions sorted by degree and weighted degree. B uses six GPS layers, width 256 and eight attention heads. Both predict period-π phases. I used graph-invariant anchors to orient degenerate Laplacian eigenspaces and zeroed modes that no anchor resolved. I applied random sign flips during training. C uses 20 label epochs followed by 20 energy epochs for the bake-off. D uses four hypotheses. E uses energy alone. I ran no optional parameter sweep."])
    else:
        preliminary = {name: read_evidence(folder / "evaluations" / (name + "_s42_val.json"))
                       for name in ["A", "B", "Bcls", "E", "C03", "C10", "D"]}
        if any(value is not None for value in preliminary.values()):
            lines.extend(["", "## Architecture comparison in progress", "",
                          "These are completed 40-epoch, seed-42 validation results. The comparison is incomplete. I have not selected a final model or evaluated the held-out results. Post-processing follows the declared paired-bootstrap simplicity rule within each completed variant.", "",
                          "| Variant | Validation geometric mean cost/random [95% interval] | Hit [95% interval] | Post-processing |", "|---|---|---|---|"])
            for name, evaluation in preliminary.items():
                if evaluation is None:
                    lines.append("| %s | pending | pending | pending |" % name)
                    continue
                strategy = validation_strategy(evaluation)["chosen"]
                metrics = evaluation["summaries"][strategy]["metrics"]
                lines.append("| %s | %s | %s | %s |" % (name, report_interval(metrics["geometric_mean_cost_ratio"]),
                                                       report_interval(metrics["hit_rate"], True), strategy))
    ablation = label_ablation_comparison()
    if ablation:
        lines.extend(["", "### Do the labels help?", "",
                      "The paired supervised-regression B / energy-only E validation cost ratio is %s. %s in this matched 40-epoch, seed-42 comparison on %d graphs. Both use the same encoder and regression heads, training ids and optimizer settings. Each uses its validation-selected post-processing. %s [Paired evidence](label_ablation.json)." % (
                          report_interval(ablation["supervised_regression_over_energy_only"]), ablation["interpretation"], ablation["graphs"], ablation["limits"])])
    if final:
        lines.extend(["", "## Held-out results", "", "Each actual joint energy/gradient call receives a parameter-shift-equivalent charge of 2(m+n)+1 circuit evaluations; the simulator itself uses adjoint gradients. Ranking costs one energy evaluation per candidate. Expected cost adds estimated, unexecuted random fallback after a warm-start failure. Each method assumes that it recognises the fixed empirical reference energy when reached. A graph with zero random hits uses R times the mean restart cost as a conservative lower bound for the random reference. Test R=50; validation R=20. Zero plus noise averages five draws of standard deviation 0.05 in phase units."])
        training = read_evidence(folder / "final_training.json")
        lines.extend(["", "I checked validation every five epochs. I stopped a seed after four checks without improvement, with a maximum of 150 epochs. I restored its best eligible checkpoint. For C, I reserved the first 75 epochs for labels and assessed early stopping during the energy-fine-tuning phase.", "",
                      "| Seed | Best epoch | Stopped epoch | Best validation cost/random | Final checkpoint |", "|---:|---:|---:|---:|---|"])
        for seed, value in training["seeds"].items():
            lines.append("| %s | %d | %d | %.8f | %s |" % (seed, value["best"]["epoch"], value["stopped_epoch"], value["best"]["score"], value["final_checkpoint"]))
        for tag in ["test_split", "large", "ian"]:
            methods = final["sets"][tag]
            append_metric_table(lines, tag, {name: value["summary"] for name, value in methods.items()})
            for field in ["by_n", "by_type"]:
                for value in methods["raw"][field]:
                    append_metric_table(lines, tag + " " + field + "=" + value,
                                        {name: method[field][value] for name, method in methods.items()})
                    if "label_" + field in methods["raw"] and value in methods["raw"]["label_" + field]:
                        lines.extend(["", "Label metrics for " + tag + " " + field + "=" + value + ":", "",
                                      "| Method | Metric | Estimate [95% interval] |", "|---|---|---|"])
                        for name in ["raw", "snap", "better"]:
                            for metric, estimate in methods[name]["label_" + field][value].items():
                                lines.append("| %s | %s | %s |" % (name, metric, report_interval(estimate, metric != "mean_pick_dist")))
            lines.extend(["", "### " + tag + " seed spread", "", "These standard deviations describe model seeds. The intervals above describe resampling graphs.", "",
                          "| Method | Metric | Mean across seeds | Sample standard deviation | Seed values |", "|---|---|---:|---:|---|"])
            for name in ["raw", "snap", "better"]:
                for metric, spread in methods[name]["seed_spread"].items():
                    lines.append("| %s | %s | %.8f | %.8f | %s |" % (name, metric, spread["mean"], spread["std"], json.dumps(spread["values"])))
                if "label_summary" in methods[name]:
                    lines.extend(["", "Label accuracy for %s (%s):" % (name, tag), "", "| Metric | Estimate [95% interval] |", "|---|---|"])
                    for metric, value in methods[name]["label_summary"].items():
                        lines.append("| %s | %s |" % (metric, report_interval(value, metric != "mean_pick_dist")))
        append_metric_table(lines, "Old paper GNN on Ian's 20 rows", {"paper raw": final["sets"]["paper_ian"]["summary"]})
        ian_labels = ian_label_statistics()
        lines.extend(["", "### Ian reference-angle accuracy", "", ian_labels["reference"], "",
                      "The model rows average three seeds within each of the same twenty graphs. The old-paper row uses its recorded predictions. Intervals resample whole graphs. [Complete angle tables](ian_label_statistics.json).", "",
                      "| Method | Metric | Estimate [95% interval] |", "|---|---|---|"])
        for name, value in ian_labels["methods"].items():
            for metric, estimate in value["summary"].items():
                lines.append("| %s | %s | %s |" % (name, metric, report_interval(estimate, metric != "mean_pick_dist")))
        draw_result_figures(final)
        lines.extend(["", "![Hit rate by size](figures/hit_rate_by_n.png)", "", "![Expected cost ratio by size](figures/cost_ratio_by_n.png)", "", "![Initial energy distribution](figures/initial_energy_distribution.png)"])
    lines.extend(["", "## Cost", "", "I used only Modal workspace tejas-16842. The original 100,000-graph forecast was $%.8f; it predates the interruption and the approved saved-population training scope. Modal has reported $%.8f. I retain $%.8f for unreported intervals and $%.2f for overhead. The current upper reservation is $%.8f. The [ledger](../cost_ledger.md) lists app costs and resource-hours, including failed runs." %
                  (config["cost"]["projected_total_usd"] if config else 0, billing["reported_usd"], billing["unreported_reservation_usd"], billing["overhead_reservation_usd"], billing["upper_reservation_usd"]),
                  "", "## Limits and failures", "",
                  "The initial pilot used too many numerical threads because Modal replaced image environment settings. I stopped it, preserved its data, excluded its timings and charged it to this task. I then forced one thread in the parent and child processes and checked the loaded libraries. Two corrected-pilot graphs exceeded the first time limit; I resumed them without changing their parameters. One audited-pilot row failed the gradient threshold and remains unusable.", "",
                  "The first GPU workers failed to import modal_app because the GPU image omitted PYTHONPATH. I first misread the empty container snapshots as a scheduling restriction. Logs identified the import failure. I repaired the source path and passed the numerical gate on an A100 under the separately approved $0.55 cap. That approval excluded model training.", "",
                  "An early angle-box check used the raw-gamma tolerance directly on weighted phases and flagged 12 coordinates. I corrected the units: phase tolerance is edge weight times the unchanged run_filter gamma tolerance. All frozen usable labels pass. I changed no labels or training ids.", "",
                  "A later L4-only training continuation remained unallocated. I stopped that request before replacing it with the permitted L4/A10 preference. The replacement resumed the saved checkpoint with unchanged model, data and optimizer settings. The first final fit then waited unallocated on both types, so I added L40S after costing all remaining fits and comparisons below the $280 stop. Each training result records the actual GPU. The ledger uses the maximum eligible rate for reservations; model training never requests an A100.", "",
                  "Final and learning-curve training reuse at most one GPU worker across validation checks, with a 90-second idle window. This reduces repeated allocation waits. CPU containers still perform run_filter evaluation. The cost forecast includes paid GPU idle time, and the training-session context stops its app on completion or failure.", "",
                  "A live progress-file read returned undecodable bytes and the first continuation stopped the full run. After shutdown, I parsed all 653 saved shards, containing 12,347 records, without an error. All 80 status-file rereads passed. I changed the monitor to skip an unreadable status snapshot while keeping scientific data checks strict. The exact cause of the transient read is unknown. Any resumed attempt keeps the same frozen counts and spends only the remaining resource tokens plus a separately costed startup reservation.", "",
                  "The host C: drive filled after the first final seed's epoch-40 evaluation. The runner stopped its GPU session automatically. All 80 batches (2,000 validation graphs), predictions and checkpoints were already saved. I copied all 14,480 result files to D:, verified every file's SHA-256, preserved the original results path with a directory junction, and rebuilt the aggregate from the complete saved batches without repeating paid polishes. Temporary files for resumed coordinators also use the D: result directory. The frozen model source and data were unchanged.", "",
                  "Two independent seeds and the unrelated-seed audits support reproducibility. They do not prove a mathematical global minimum or a unique label. The n=13–14 set tests this run's larger graphs. It does not establish generalisation to other sizes, depths or graph distributions. WL grouping can combine non-isomorphic graphs. The training plan uses only weighted p=1 graphs.", "",
                  "Volume commits protect completed records every two minutes. A preemption can still discard the search currently running inside an immutable run_filter call. A long individual graph can therefore lose more than ten minutes of work.", "",
                  "Historical scope: the 174 instances contain 105 weighted Max-Cut and 69 spin-glass instances. The relabel result, 19/20, and the 2% weight result, 16/20, pool ten Max-Cut and ten spin-glass graphs. The 21 reference graphs are weighted Max-Cut. Earlier boosted-tree experiments are not GNN results.", "",
                  "Security: early SDK debug output exposed authorization headers. I disabled transport debug logging and removed it from saved logs. The exposed credential needs rotation after this run. I did not include credential values in scientific uploads or this report."])
    interruption = read_evidence(PREFLIGHT / "transport_failure_analysis.json")
    if interruption:
        lines.extend(["", "## Interrupted full run", "",
                      "In the interrupted attempt, the runner received zero batch-completion results out of 80 expected. It saved 63,672 records. At that point, 63,288 of the 100,000 expected ids met the usable criteria, 384 had recorded failures and 36,328 were missing. Usable coverage was 63.288%, below the 95% gate. I stopped that attempt before GNN training.", "",
                      "The client log records a TLS handshake timeout in the Modal output task. The installed Modal 1.6.3 code suppresses a connection error when detached mode and live output are both enabled. I infer that a transport interruption allowed the batch context to end without completion results. I have not established the full SDK failure chain.", "",
                      "I removed live log streaming from the runner. I added persisted call ids, retries that read the same call, and an explicit check for every batch and shard completion. Local checks cover a transport retry and a missing completion. The twenty-graph test's paid verification is recorded in test_smoke_gate.json once it passes.", "",
                      "The [stop analysis](../preflight/full_stop_analysis.json) records counts and failure causes. The [transport analysis](../preflight/transport_failure_analysis.json) separates observations from inference. The saved CSV passed the independent energy and traversal-order checks. The approved Python inputs and run_filter remain unchanged."])
    path = folder / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paragraphs = ["Hi Ian,", "I ran the weighted Max-Cut p=1 data pipeline on Presonance's Modal workspace. The 100- and 400-restart pilots agreed on 320/320 picks. The audited pilot had 319/320 usable rows and 320/320 audit agreement."]
    if "full" in statistics:
        paragraphs.append("I saved %d main-dataset graphs out of %d planned. Of these, %d meet the usable criteria and %d fail them. The other %d planned ids are not run. The separate larger-size set has %s. I kept rejected saved records and their failure evidence." % (statistics["full"]["present"], statistics["full"]["planned"], statistics["full"]["usable"], statistics["full"]["present"] - statistics["full"]["usable"], statistics["full"]["not_run"], "%d usable out of %d" % (statistics["test"]["usable"], statistics["test"]["expected"]) if "test" in statistics else "not completed"))
    else:
        paragraphs.append("The approved plan contains 100,000 graphs at n=5–12 and 300 graphs each at n=13 and n=14. Full data generation has not passed its completion gates.")
    if complete:
        method = final["strategy"]
        small = final["sets"]["test_split"][method]["summary"]["metrics"]
        large = final["sets"]["large"][method]["summary"]["metrics"]
        paragraphs.append("The selected model's geometric mean cost relative to random starts is %s on the held-out n=5–12 set. Its hit rate is %s. I counted candidate ranking, polishing and fallback restarts." % (report_interval(small["geometric_mean_cost_ratio"]), report_interval(small["hit_rate"], True)))
        if small["geometric_mean_cost_ratio"]["value"] >= 1:
            paragraphs[-1] += " It did not beat random starts on this metric."
        elif small["geometric_mean_cost_ratio"]["high95"] >= 1:
            paragraphs[-1] += " The interval includes one, so the reduction remains uncertain."
        paragraphs.append("On n=13–14, the geometric mean cost ratio is %s and the hit rate is %s. These intervals resample graphs; the report also gives the spread across three training seeds and the comparison on your 20 graphs." % (report_interval(large["geometric_mean_cost_ratio"]), report_interval(large["hit_rate"], True)))
        if curve:
            values = [curve["fractions"][percent] for percent in ["25", "50", "100"]]
            paragraphs.append("I also trained the winning model on 25%%, 50%% and 100%% of the training split: %s graphs. The held-out cost/random ratios were %s, using three seeds at each size. I reused the final fits for 100%%. The 36,328 missing training-size graphs remain not run." % (
                ", ".join(format(value["graphs"], ",") for value in values),
                "; ".join(report_interval(value["summary"]["metrics"]["geometric_mean_cost_ratio"]) for value in values)))
    else:
        paragraphs.append("I have no final GNN savings or size-generalization result to report. " + ("I stopped the run. " + str(reason).rstrip(". ") + "." if reason else "The remaining stages are still in progress."))
        if "full" in statistics:
            stats = statistics["full"]
            paragraphs.append("Among the saved main-dataset records, %d fail the usable checks. The largest cause is %d gradients above the threshold. The seed guards and audits support reproducibility, but they do not prove global optimality or uniqueness." % (stats["present"] - stats["usable"], stats["causes_overlap"].get("gradient_above_tolerance", 0)))
        else:
            paragraphs.append("One pilot row failed the gradient threshold. I excluded it. The seed guards and audits support reproducibility, but they do not prove global optimality or uniqueness.")
    paragraphs.extend(["Modal has reported $%.8f so far. The report separates posted charges from unreported reservations and includes failed runs. I have attached the data, checks and results that completed." % billing["reported_usd"], "Thanks,\nAshay Sarda"])
    (folder / "email_draft.txt").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    floor_review = read_evidence(PREFLIGHT / "reference_floor_review.json")
    if complete and curve is not None and floor_review and not floor_review.get("partial_model_review", True):
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "src/report_comparison_additions.py"),
                        "--require-complete", "--append-report"], check=True, cwd=ROOT)
    pending_reason = "Approved training comparison is still pending" if curve_authorized else "Additional training comparison awaits explicit user clarification after automatic approval review"
    return {"report": str(path), "completed": complete and not curve_pending,
            "main_experiment_completed": complete,
            "reason": pending_reason if complete and curve_pending else None if reason is None else str(reason)}


# Select completed bake-off and final weights while omitting temporary optimizer snapshots.
def delivery_checkpoints():
    selected = set()
    bakeoff = read_evidence(RESULTS / "gnn" / "bakeoff.json")
    if bakeoff:
        for epochs in bakeoff["checkpoint_paths"].values():
            selected.update(epochs.values())
    for path in (RESULTS / "gnn" / "training_jobs").glob("*.json"):
        job = read_evidence(path)
        if not job["name"].startswith(("final_", "curve")):
            selected.add(job["checkpoint"])
    curve = read_evidence(RESULTS / "gnn" / "learning_curve.json")
    if curve:
        for fraction in ["25", "50"]:
            selected.update(seed["checkpoint"] for seed in curve["fractions"][fraction]["seeds"].values())
    final = read_evidence(RESULTS / "gnn" / "final_training.json")
    if final:
        selected.update(value["final_checkpoint"] for value in final["seeds"].values())
    else:
        for path in (RESULTS / "gnn").glob("final_seed*_progress.json"):
            progress = read_evidence(path)
            if progress.get("best"):
                selected.add(progress["best"]["training"]["checkpoint"])
    return {name.replace("\\", "/") for name in selected}


# Keep this task's research outputs and reject unrelated local files or transport logs.
def delivery_result_path(relative, checkpoints):
    parts = Path(relative).parts
    if not parts or parts[0] not in ["dataset", "dataset_test", "dataset_pilot", "dataset_pilot400",
                                   "dataset_pilot_oversubscribed", "reference_stage", "ian_floors",
                                   "worker_proofs", "preflight", "gnn", "cost_ledger.md"]:
        return False
    if any(part in ["__pycache__", ".git", ".venv", ".codex", ".claude"] for part in parts):
        return False
    if parts[:2] == ("preflight", "delivery_inventory"):
        return False
    if parts[:2] == ("worker_proofs", "evaluation") and len(parts) >= 4 and parts[3] != "batch_0000":
        return False
    if Path(relative).name in DELIVERY_OPERATIONAL_EXCLUSIONS or Path(relative).suffix.lower() == ".log":
        return False
    if Path(relative).suffix in [".tmp", ".partial", ".download"] or ".working." in str(relative):
        return False
    if parts[:2] == ("gnn", "checkpoints"):
        return str(relative).replace("\\", "/") in checkpoints
    return True


# Synchronize only selected research artifacts and stream large model files to disk.
def sync_delivery_artifacts():
    import modal
    from concurrent.futures import ThreadPoolExecutor
    lock_profile()
    verify_workspace()
    check_input_hashes()
    volume = modal.Volume.from_name(VOLUME_NAME)
    checkpoints = delivery_checkpoints()
    entries, directories = [], ["/"]
    stale_status = {"preflight/stop.json", "preflight/execution_error.json", "preflight/continuation_stop.json"}
    inventory_cache = PREFLIGHT / "delivery_inventory"
    frozen_volume = all(read_evidence(path).get("stopped_epoch") is not None
                        for path in (PREFLIGHT / "paid_apps").glob("*.json"))
    inventory_tag = sha256(RESULTS / "gnn" / "learning_curve.json") if frozen_volume and (RESULTS / "gnn" / "learning_curve.json").exists() else None

    # Bound each remote inventory request to one directory.
    def list_directory(path):
        from types import SimpleNamespace
        parts = Path(path.lstrip("/")).parts
        recursive = False
        cache_path = inventory_cache / (hashlib.sha256(path.encode("utf-8")).hexdigest() + ".json")
        cached = read_evidence(cache_path) if inventory_tag else None
        if cached and cached.get("frozen_curve_sha256") == inventory_tag and cached.get("recursive", False) == recursive:
            return [SimpleNamespace(**entry) for entry in cached["entries"]]
        for attempt in range(3):
            try:
                result = list(volume.iterdir(path, recursive=recursive))
                if recursive:
                    result = [entry for entry in result if entry.type == modal.volume.FileEntryType.FILE]
                if inventory_tag:
                    save_json(cache_path, {"path": path, "frozen_curve_sha256": inventory_tag, "recursive": recursive,
                              "entries": [{"path": entry.path, "type": int(entry.type), "size": entry.size}
                                          for entry in result], "captured_epoch": time.time()})
                return result
            except Exception as error:
                LOG.warning("Delivery inventory retry %d for %s: %s", attempt + 1, path, type(error).__name__)
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=8) as pool:
        while directories:
            batches = list(pool.map(list_directory, directories))
            directories = []
            for batch in batches:
                for entry in batch:
                    relative = Path(entry.path.lstrip("/"))
                    allowed = delivery_result_path(relative, checkpoints)
                    if entry.type != modal.volume.FileEntryType.FILE:
                        if allowed or relative.as_posix() == "gnn/checkpoints":
                            directories.append(entry.path)
                    elif allowed and relative.as_posix() not in stale_status:
                        entries.append(entry)
            LOG.info("Delivery inventory: %d files; %d directories pending", len(entries), len(directories))

    # Preserve downloads atomically and reject an unexpected remote traversal.
    delivery_root = RESULTS.resolve(strict=True)

    def copy_entry(entry):
        relative = Path(entry.path.lstrip("/"))
        target = delivery_root / relative
        require(target.resolve().is_relative_to(delivery_root), "Delivery path escapes the results folder: " + relative.as_posix())
        if target.is_file():
            return {"path": relative.as_posix(), "downloaded": False}
        require(not target.exists(), "Delivery target exists but is not a file")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".download")
        with temporary.open("wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
        require(temporary.stat().st_size == entry.size, "Downloaded delivery file size differs")
        temporary.replace(target)
        return {"path": relative.as_posix(), "downloaded": True}

    with ThreadPoolExecutor(max_workers=8) as pool:
        downloaded = list(pool.map(copy_entry, entries))
    missing = [name for name in checkpoints if not (RESULTS / name).is_file()]
    require(not missing, "A selected model checkpoint is missing from the delivery")
    expected_hashes = {}
    for path in (RESULTS / "gnn" / "training_jobs").glob("*.json"):
        job = read_evidence(path)
        if job["checkpoint"].replace("\\", "/") in checkpoints:
            expected_hashes[job["checkpoint"].replace("\\", "/")] = job["checkpoint_sha256"]
    final = read_evidence(RESULTS / "gnn" / "final_training.json")
    if final:
        for seed in final["seeds"].values():
            expected_hashes[seed["final_checkpoint"]] = seed["best"]["training"]["checkpoint_sha256"]
    curve = read_evidence(RESULTS / "gnn" / "learning_curve.json")
    if curve:
        for fraction in ["25", "50"]:
            for seed in curve["fractions"][fraction]["seeds"].values():
                expected_hashes[seed["checkpoint"]] = seed["best"]["training"]["checkpoint_sha256"]
    require(set(expected_hashes) == checkpoints, "A selected model checkpoint lacks its original training checksum")
    for relative, expected_hash in expected_hashes.items():
        require(sha256(RESULTS / relative) == expected_hash, "Downloaded model weights differ from their training checksum")
    save_json(PREFLIGHT / "delivery_sync.json", {"files": len(entries),
              "downloaded": sum(item["downloaded"] for item in downloaded), "checkpoints": sorted(checkpoints),
              "policy": "Download missing task artifacts only; retain authoritative local results",
              "evaluation_environment_proofs": "One per evaluation job (batch_0000); repeated per-batch environment snapshots remain preserved locally and on the Volume. All scientific evaluation records are included separately.",
              "individual_artifact_uploads": 0, "checkpoint_sha256_verified": expected_hashes})
    return downloaded


# Check selected-profile credential literals without logging their values.
def verify_delivery_archive_secrets(archive):
    import zipfile
    from modal.config import config
    lock_profile()
    credentials = [config.get(name) for name in ["token_id", "token_secret"]]
    require(all(isinstance(value, str) and len(value) >= 8 for value in credentials),
            "The selected-profile credential scan cannot run")
    needles = [value.encode("utf-8") for value in credentials]
    overlap = max(map(len, needles)) - 1
    findings, files, scanned_bytes = [], 0, 0
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            if entry.is_dir():
                continue
            files += 1
            tail = b""
            with package.open(entry) as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    scanned_bytes += len(chunk)
                    block = tail + chunk
                    if any(needle in block for needle in needles):
                        findings.append(entry.filename)
                        break
                    tail = block[-overlap:]
    require(not findings, "Delivery archive contains a saved credential literal: " + ", ".join(findings))
    result = {"archive_sha256": sha256(archive), "scanned_files": files, "scanned_bytes": scanned_bytes,
              "selected_profile_credential_values_checked": len(needles), "literal_credential_findings": 0,
              "credential_values_printed": False, "captured_epoch": time.time()}
    save_json(archive.parent / "credential_scan_verified.json", result)
    return result


# Package reviewed local results and verify one archive through a Modal roundtrip.
def package_delivery(reason=None):
    import modal
    import zipfile
    sync_delivery_artifacts()
    report = write_research_report(reason=reason, refresh_billing=reason is None)
    lock_profile()
    volume = modal.Volume.from_name(VOLUME_NAME)
    checkpoints = delivery_checkpoints()
    paths = {relative: ROOT / relative for relative in UPLOAD_FILES + DELIVERY_ONLY_FILES}
    for path in RESULTS.rglob("*"):
        if path.is_file() and delivery_result_path(path.relative_to(RESULTS), checkpoints):
            paths["results/" + path.relative_to(RESULTS).as_posix()] = path
    manifest_path = RESULTS / "gnn" / "delivery_manifest.json"
    paths.pop("results/gnn/delivery_manifest.json", None)
    manifest = {"profile": MODAL_PROFILE, "volume": VOLUME_NAME, "created_epoch": time.time(),
                "completed": report["completed"], "reason": report["reason"],
                "files": {relative: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                          for relative, path in sorted(paths.items())},
                "manifest_self_hash_policy": "This manifest omits itself to avoid a recursive checksum"}
    save_json(manifest_path, manifest)
    paths["results/gnn/delivery_manifest.json"] = manifest_path
    destination = RESULTS / "delivery"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (DELIVERY_NAME + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as handle:
        for relative, path in sorted(paths.items()):
            handle.write(path, relative)
    with zipfile.ZipFile(archive) as handle:
        require(set(handle.namelist()) == set(paths), "Archive contents differ from the delivery manifest")
        require(handle.testzip() is None, "Archive CRC verification failed")
        for relative, details in manifest["files"].items():
            with handle.open(relative) as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            require(actual == details["sha256"], "Archived research file SHA-256 differs")
    digest = sha256(archive)
    secret_scan = verify_delivery_archive_secrets(archive)
    checksum = archive.with_suffix(".zip.sha256")
    checksum.write_text(digest + "  " + archive.name + "\n", encoding="ascii")
    remote = "delivery/" + archive.name
    with volume.batch_upload(force=True) as batch:
        batch.put_file(archive, remote)
        batch.put_file(checksum, remote + ".sha256")
    received = archive.with_suffix(".zip.download")
    with received.open("wb") as handle:
        for chunk in volume.read_file(remote):
            handle.write(chunk)
    require(sha256(received) == digest, "The archive downloaded from Modal differs from the uploaded file")
    received.replace(archive)
    result = {"path": str(archive), "sha256": digest, "bytes": archive.stat().st_size,
              "files": len(paths), "volume_path": remote, "download_verified": True,
              "credential_scan_verified": secret_scan,
              "completed": report["completed"], "reason": report["reason"]}
    save_json(destination / "delivery_verified.json", result)
    return result


# Serialize the newly approved comparison after the active driver's final archive is complete.
def run_approved_learning_curve_after_main():
    approval = read_evidence(PREFLIGHT / "learning_curve_approval.json")
    require(approval is not None and approval.get("user_approved") is True, "Learning-curve approval is missing")
    started = time.time()
    deadline = started + FULL_TIMEOUT
    marker = PREFLIGHT / "learning_curve_continuation_status.json"
    save_json(marker, {"stage": "waiting for main experiment and archive", "status": "waiting", "created_epoch": started})
    try:
        while True:
            stop = read_evidence(PREFLIGHT / "continuation_stop.json")
            require(stop is None or stop.get("captured_epoch", 0) < started,
                    "The main experiment stopped; the additional fits must not launch")
            state = read_evidence(PREFLIGHT / "continuation_status.json") or {}
            if state.get("stage") == "complete" and state.get("captured_epoch", 0) >= started:
                break
            require(time.time() < deadline, "Main-experiment wait exceeded its one-day deadline")
            time.sleep(30)
        save_json(marker, {"stage": "learning curve", "status": "running", "created_epoch": started})
        run_learning_curve()
        write_billing_summary()
        delivery = package_delivery()
        save_json(marker, {"stage": "complete", "status": "complete", "created_epoch": started,
                           "captured_epoch": time.time(), "delivery": delivery})
        return 0
    except Exception as error:
        stop_task_apps()
        save_json(marker, {"stage": "stopped", "status": "stopped", "reason": str(error),
                           "type": type(error).__name__, "created_epoch": started, "captured_epoch": time.time()})
        write_billing_summary()
        write_research_report(reason=str(error) or type(error).__name__)
        return 1


# Continue the approved stages after the existing full run, stopping and packaging on a failed gate.
def run_approved_remainder():
    try:
        population = load_training_population()
        require((PREFLIGHT / "full_corpus_csv_gate.json").exists(), "Saved-population CSV gate has not passed")
        bias = read_evidence(PREFLIGHT / "saved_data_bias_check.json")
        require(bias is not None and not bias["clearly_easy"], "Complete the no-compute bias check first")
        archive = PREFLIGHT / "runner_repairs" / ("before_saved_population_" + str(int(time.time())))
        archive.mkdir(parents=True, exist_ok=True)
        for name in ["stop.json", "execution_error.json", "continuation_stop.json"]:
            path = PREFLIGHT / name
            if path.exists():
                require(not (archive / name).exists(), "Prior stop evidence is already archived; inspect the new stop")
                path.rename(archive / name)
        save_json(PREFLIGHT / "saved_population_approval.json", {
            "training_population_sha256": sha256(PREFLIGHT / "training_population.json"),
            "usable": population["usable"], "saved": population["saved"], "not_run": population["not_run"],
            "quality_denominator": "saved records", "remaining_training_size_generation_authorized": False,
            "learning_curve_scheduling": "explicitly approved; comparison coordinator follows the final fits and reuses 100%",
            "learning_curve_approval_sha256": sha256(PREFLIGHT / "learning_curve_approval.json"), "created_epoch": time.time()})
        write_billing_summary()
        stages = [(PREFLIGHT / "test_smoke_gate.json", "twenty-graph runner completion test", lambda: launch_test_completion(smoke=True)),
                  (PREFLIGHT / "test_corpus_csv_gate.json", "remaining larger held-out corpus", launch_test_completion),
                  (PREFLIGHT / "ian_floor_gate.json", "Ian floors", launch_ian_floors),
                  (PREFLIGHT / "stage3_gate.json", "data verification and splits", prepare_training_artifacts),
                  (RESULTS / "gnn" / "torch_energy_check_cuda.json", "GPU numerical verification", launch_gpu_energy_check),
                  (RESULTS / "gnn" / "bakeoff.json", "architecture bake-off", run_bakeoff),
                  (RESULTS / "gnn" / "final_training.json", "three-seed final training", run_final_training),
                  (RESULTS / "gnn" / "final_results.json", "held-out evaluation", run_final_evaluation)]
        for marker, name, action in stages:
            if marker.exists():
                LOG.info("Already completed: %s", name)
                continue
            save_json(PREFLIGHT / "continuation_status.json", {"stage": name, "status": "running", "captured_epoch": time.time()})
            LOG.info("Starting %s", name)
            action()
            write_billing_summary()
        save_json(PREFLIGHT / "continuation_status.json", {"stage": "delivery", "status": "running", "captured_epoch": time.time()})
        package_delivery()
        save_json(PREFLIGHT / "continuation_status.json", {"stage": "complete", "status": "complete", "captured_epoch": time.time()})
        return 0
    except Exception as error:
        stop_task_apps()
        save_json(PREFLIGHT / "continuation_stop.json", {"type": type(error).__name__, "reason": str(error), "captured_epoch": time.time()})
        LOG.error("Approved continuation stopped: %s", error)
        write_stop_report(error)
        try:
            package_delivery(reason=str(error))
        except Exception as delivery_error:
            save_json(PREFLIGHT / "delivery_error.json", {"type": type(delivery_error).__name__, "reason": str(delivery_error)})
            LOG.error("Partial delivery needs repair: %s", delivery_error)
        return 1


# Run the debugger default through every local gate before the first Modal stage.
# Run the debugger default through every local gate before the first Modal stage.
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "shard":
        child_shard(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6] if len(sys.argv) > 6 else None)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "pilot_record":
        child_pilot_record(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        return 0
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(PREFLIGHT / "runner.log", encoding="utf-8")])
    LOG.setLevel(logging.DEBUG)
    for name in ["hpack", "h2", "grpclib", "modal", "asyncio"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    save_json(PREFLIGHT / "research_provenance.json", RESEARCH_PROVENANCE)
    try:
        install_reference()
        check_input_hashes()
        lock_profile()
        stage_zero()
        LOG.info("Local preflight passed.")
        if RUN_STAGE == "references":
            launch_references()
        elif RUN_STAGE == "pilot":
            launch_pilots()
        elif RUN_STAGE == "pilot400":
            launch_pilot_mode("pilot400")
            finalize_pilots()
        elif RUN_STAGE == "recover_pilot":
            resume_pilot_records("pilot")
        elif RUN_STAGE == "finalize_pilots":
            finalize_pilots()
        elif RUN_STAGE == "test_timing":
            launch_test_timings()
        elif RUN_STAGE == "freeze":
            freeze_plan()
        elif RUN_STAGE in ["full", "test"]:
            launch_sharded_mode(RUN_STAGE)
        elif RUN_STAGE == "resume_full":
            launch_sharded_mode("full", resume=True)
        elif RUN_STAGE == "ian":
            launch_ian_floors()
        elif RUN_STAGE == "prepare":
            prepare_training_artifacts()
        elif RUN_STAGE == "bakeoff":
            run_bakeoff()
        elif RUN_STAGE == "final_training":
            run_final_training()
        elif RUN_STAGE == "final_evaluation":
            run_final_evaluation()
        elif RUN_STAGE == "learning_curve":
            run_learning_curve()
        elif RUN_STAGE == "remaining":
            return run_approved_remainder()
        elif RUN_STAGE == "report":
            write_research_report(refresh_billing=False)
        elif RUN_STAGE == "deliver":
            package_delivery()
        else:
            require(False, "Unsupported execution stage: %s" % RUN_STAGE)
    except GateFailure as error:
        stop_task_apps()
        save_json(PREFLIGHT / "stop.json", {"status": "stopped", "reason": str(error),
                                          "allowed_modal_profile": MODAL_PROFILE,
                                          "modal_compute_launched": (PREFLIGHT / "reference_launch.json").exists()})
        LOG.error("Execution stopped: %s", error)
        write_stop_report(error)
        return 1
    except Exception as error:
        stop_task_apps()
        save_json(PREFLIGHT / "execution_error.json", {"status": "stopped", "type": type(error).__name__,
                  "reason": str(error), "allowed_modal_profile": MODAL_PROFILE})
        LOG.error("Execution error: %s", error)
        write_stop_report(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
