"""
Paper-based fixed-depth GINE reproduction and matched 30-epoch model training.
The original implementation was unavailable; documented weighted adaptations are explicit.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import gnn

ROOT = Path(__file__).resolve().parent.parent
EPOCHS = 30
REGULARIZATION = 0.02
FEATURES = {}
DIFFUSION_STEPS = 1000
SAMPLING_STEPS = 50
PAPER_ANGLE_UNITS = "RZZ r=-2*w*gamma_filter; RX(2*b), b=-beta_filter; decode theta=-r/2, beta=-b"
DIFFUSION_HIDDEN = 128
DIFFUSION_LAYERS = 4


# Collate identical numerical features in NumPy before one transfer per tensor.
def feature_batch(features, device="cpu", random_signs=None, baseline=False):
    batch, nodes, edges = len(features), max(row["n"] for row in features), max(len(row["edges"]) for row in features)
    result = {"node": np.zeros((batch, nodes, features[0]["node"].shape[1]), dtype=np.float32),
              "edge": np.zeros((batch, edges, features[0]["edge"].shape[1]), dtype=np.float32),
              "reverse_edge": np.zeros((batch, edges, features[0]["edge"].shape[1]), dtype=np.float32),
              "index": np.zeros((batch, edges, 2), dtype=np.int64),
              "node_mask": np.zeros((batch, nodes), dtype=bool), "edge_mask": np.zeros((batch, edges), dtype=bool)}
    for k, row in enumerate(features):
        n, m = row["n"], len(row["edges"])
        result["node"][k, :n] = row["node"]
        result["edge"][k, :m] = row["edge"]
        result["reverse_edge"][k, :m] = row["reverse_edge"]
        result["index"][k, :m] = row["edges"]
        result["node_mask"][k, :n] = True
        result["edge_mask"][k, :m] = True
        if random_signs is not None:
            signs = random_signs.choice([-1., 1.], size=8)
            result["node"][k, :n, -8:] *= signs
            if baseline:
                for field in ["edge", "reverse_edge"]:
                    result[field][k, :m, 2:10] *= signs
                    result[field][k, :m, 10:18] *= signs
            else:
                for field in ["edge", "reverse_edge"]:
                    result[field][k, :m, -24:-16] *= signs
    return {name: torch.from_numpy(value).to(device) for name, value in result.items()}


# Decode actual RZZ rotations and paper mixer parameters with period 2*pi.
class PaperModel(gnn.WarmStartModel):
    # Keep the reported GINE architecture and directed edge prediction heads.
    def __init__(self, node_features, edge_features):
        super().__init__(node_features, edge_features, baseline=True, hidden=192, layers=5)

    # Recover gate rotation r and mixer b without our period-pi transformation.
    def forward(self, data, expected_angles=False):
        result = super().forward(data)
        result["theta"] = torch.atan2(result["edge_raw"][..., 1], result["edge_raw"][..., 0])
        result["beta"] = torch.atan2(result["node_raw"][..., 1], result["node_raw"][..., 0])
        return result


# Denoise circular edge and node phases while retaining a variable-size weighted graph.
class GraphDiffusion(nn.Module):
    # Use a graph network inside the diffusion model rather than a fixed six-angle MLP.
    def __init__(self, node_features, edge_features):
        super().__init__()
        self.baseline, self.classification, self.hypotheses = False, False, 4
        self.network = gnn.WarmStartModel(node_features + 10, edge_features + 10,
                                         hidden=DIFFUSION_HIDDEN, layers=DIFFUSION_LAYERS)
        self.register_buffer("alpha_bar", torch.cumprod(1 - torch.linspace(1e-4, .02, DIFFUSION_STEPS), 0))

    # Add noise coordinates and a common time embedding to each active node and edge.
    def forward(self, data, noisy_edge, noisy_node, timestep):
        angle = timestep.float()[:, None] / (DIFFUSION_STEPS - 1) * torch.tensor(
            [1., 2., 4., 8.], device=timestep.device)[None, :] * np.pi
        embedding = torch.cat([angle.sin(), angle.cos()], -1)
        updated = dict(data)
        for field, noisy in [("node", noisy_node), ("edge", noisy_edge), ("reverse_edge", noisy_edge)]:
            extra = torch.cat([noisy, embedding[:, None, :].expand(-1, noisy.shape[1], -1)], -1)
            updated[field] = torch.cat([data[field], extra], -1)
        prediction = self.network(updated)
        return prediction["edge_raw"][:, 0], prediction["node_raw"][:, 0]


# Embed each physical period-pi phase on the unit circle; weights act through theta=w*gamma.
def circular_targets(rows, data):
    edge, node = targets(rows, data, False)
    return tuple(torch.stack([(2 * value).cos(), (2 * value).sin()], -1) for value in [edge, node])


# Learn Gaussian noise with graph-normalized losses and fixed validation noise when requested.
def diffusion_loss(model, rows, data, generator=None):
    clean_edge, clean_node = circular_targets(rows, data)
    timestep = torch.randint(DIFFUSION_STEPS, (len(rows),), device=data["node"].device, generator=generator)
    signal = model.alpha_bar[timestep, None, None]
    noise_edge = torch.randn(clean_edge.shape, device=clean_edge.device, generator=generator)
    noise_node = torch.randn(clean_node.shape, device=clean_node.device, generator=generator)
    noisy_edge = signal.sqrt() * clean_edge + (1 - signal).sqrt() * noise_edge
    noisy_node = signal.sqrt() * clean_node + (1 - signal).sqrt() * noise_node
    predicted = model(data, noisy_edge, noisy_node, timestep)
    loss = 0.
    for value, noise, mask in zip(predicted, [noise_edge, noise_node], [data["edge_mask"], data["node_mask"]]):
        mse = (value - noise).square().mean(-1)
        loss = loss + ((mse * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    return loss


# Sample four proposals by seeded DDIM, projecting to phases only after denoising.
def diffusion_sample(model, data, seed, candidates=4, deadline_epoch=float("inf")):
    device = data["node"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    results = []
    steps = np.unique(np.linspace(DIFFUSION_STEPS - 1, 0, SAMPLING_STEPS).round().astype(int))[::-1]
    for candidate in range(candidates):
        edge = torch.randn((*data["edge_mask"].shape, 2), device=device, generator=generator)
        node = torch.randn((*data["node_mask"].shape, 2), device=device, generator=generator)
        for position, step in enumerate(steps):
            if time.time() >= deadline_epoch:
                raise RuntimeError("Diffusion sampling resource deadline reached")
            timestep = torch.full((len(edge),), int(step), device=device, dtype=torch.long)
            predicted = model(data, edge, node, timestep)
            alpha = model.alpha_bar[step]
            previous = model.alpha_bar[int(steps[position + 1])] if position + 1 < len(steps) else torch.ones((), device=device)
            updated = []
            for noisy, noise in zip([edge, node], predicted):
                clean = (noisy - (1 - alpha).sqrt() * noise) / alpha.sqrt()
                updated.append(previous.sqrt() * clean + (1 - previous).sqrt() * noise)
            edge, node = updated
        results.append(tuple(torch.atan2(value[..., 1], value[..., 0]) / 2 for value in [edge, node]))
    return {"theta": torch.stack([r[0] for r in results], 1), "beta": torch.stack([r[1] for r in results], 1)}


# Build paper-style features with an explicit optional edge-weight adaptation.
def paper_features(row, weighted=True):
    source = row if weighted else dict(row, w_sorted=[1.] * len(row["edges"]))
    result = gnn.graph_features(source, baseline=True)
    if not weighted:
        result["edge"] = result["edge"][:, :-1]
        result["reverse_edge"] = result["reverse_edge"][:, :-1]
    return result


# Pad physical labels in the output coordinates of the requested model.
def targets(rows, data, paper):
    edge = torch.zeros(data["edge_mask"].shape, device=data["node"].device)
    node = torch.zeros(data["node_mask"].shape, device=data["node"].device)
    for k, row in enumerate(rows):
        m = len(row["edges"])
        gamma = np.asarray(row["pick"][:m])
        phase = gamma * row["w_sorted"]
        values = -2 * phase if paper else phase
        edge[k, :m] = torch.as_tensor(values, dtype=torch.float32, device=edge.device)
        beta = np.asarray(row["pick"][m:])
        node[k, :row["n"]] = torch.as_tensor(-beta if paper else beta, dtype=torch.float32, device=node.device)
    return edge, node


# Implement the PDF's circular cosine loss and lambda=0.02 radial regularizer.
def paper_loss(prediction, edge, node, data):
    result = 0.
    for raw_values, truth, mask in [(prediction["edge_directions"], edge, data["edge_mask"]),
                                    ([prediction["node_raw"]], node, data["node_mask"])]:
        losses = []
        for raw in raw_values:
            angle = torch.atan2(raw[..., 1], raw[..., 0])
            value = 1 - torch.cos(angle - truth[:, None, :])
            value = value + REGULARIZATION * (torch.linalg.vector_norm(raw, dim=-1) - 1).abs()
            losses.append((value * mask[:, None, :]).sum() / mask.sum().clamp_min(1))
        result = result + torch.stack(losses).mean()
    return result


# Compute each model's native supervised loss without changing graph or label identities.
def model_loss(model, prediction, rows, data, paper, class_weights):
    edge, node = targets(rows, data, paper)
    return paper_loss(prediction, edge, node, data) if paper else gnn.label_loss(
        model, prediction, edge, node, data, *(class_weights or [None, None]))


# Evaluate native validation loss on a fixed manifest without test-label selection.
def validation_loss(model, rows, features, paper, class_weights, deadline_epoch=float("inf")):
    model.eval()
    total, seen = 0., 0
    generator = torch.Generator(device="cuda").manual_seed(817231)
    with torch.inference_mode():
        for start in range(0, len(rows), gnn.BATCH_SIZE):
            if time.time() >= deadline_epoch:
                raise RuntimeError("Validation resource deadline reached")
            selected = rows[start:start + gnn.BATCH_SIZE]
            data = feature_batch([features[r["id"]] for r in selected], device="cuda", baseline=model.baseline)
            loss = diffusion_loss(model, selected, data, generator) if isinstance(model, GraphDiffusion) else model_loss(
                model, model(data), selected, data, paper, class_weights)
            total += float(loss) * len(selected)
            seen += len(selected)
    return total / seen


# Record synchronized prediction times and convert all outputs to common physical phases.
def predict(model, rows, features, paper, seed=42, deadline_epoch=float("inf")):
    model.eval()
    records = []
    with torch.inference_mode():
        warmup = rows[:min(gnn.BATCH_SIZE, len(rows))]
        data = feature_batch([features[r["id"]] for r in warmup], device="cuda", baseline=model.baseline)
        if isinstance(model, GraphDiffusion):
            diffusion_sample(model, data, seed, candidates=1, deadline_epoch=deadline_epoch)
        else:
            model(data)
        torch.cuda.synchronize()
        for start in range(0, len(rows), gnn.BATCH_SIZE):
            if time.time() >= deadline_epoch:
                raise RuntimeError("Prediction resource deadline reached")
            selected = rows[start:start + gnn.BATCH_SIZE]
            torch.cuda.synchronize()
            started = time.perf_counter()
            data = feature_batch([features[r["id"]] for r in selected], device="cuda", baseline=model.baseline)
            result = diffusion_sample(model, data, seed + start, deadline_epoch=deadline_epoch) if isinstance(model, GraphDiffusion) else model(data)
            theta = result["theta"].cpu().numpy()
            beta = result["beta"].cpu().numpy()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            for k, row in enumerate(selected):
                m, n = len(row["edges"]), row["n"]
                candidates = []
                for j in range(model.hypotheses):
                    phases = -.5 * theta[k, j, :m] if paper else theta[k, j, :m]
                    mixers = -beta[k, j, :n] if paper else beta[k, j, :n]
                    candidates.append(np.r_[phases, mixers].astype(float).tolist())
                records.append({"id": row["id"], "n": n, "phase_candidates": candidates,
                                "generated_candidates": model.hypotheses,
                                "inference_batch_graphs": len(selected), "inference_batch_seconds": elapsed,
                                "inference_amortized_seconds": elapsed / len(selected)})
    return records


# Train one documented comparison fit on the exact expanded graph and label manifest.
def remote_train(job, source_hashes, manifest_hash):
    import modal
    import modal_app as runner
    from paper_comparison import read_large_all
    runner.configure_threads()
    runner.PREFLIGHT = runner.RESULTS / "worker_proofs/comparison" / job["name"]
    for relative, digest in source_hashes.items():
        runner.require(runner.sha256(ROOT / relative) == digest, "Comparison worker source differs")
    runner.require(runner.sha256(runner.RESULTS / "gnn/corpus_manifest.json") == manifest_hash,
                   "Comparison corpus manifest differs")
    runner.check_input_hashes()
    runner.stage_zero()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    torch.cuda.manual_seed_all(job["seed"])
    np.random.seed(job["seed"])
    torch.use_deterministic_algorithms(True)
    splits = runner.read_evidence(runner.RESULTS / "gnn/splits.json")
    manifest = runner.read_evidence(runner.RESULTS / "gnn/corpus_manifest.json")
    all_rows = runner.read_corpus_rows("full", manifest=manifest)
    by_id = {r["id"]: r for r in all_rows}
    train_rows = [by_id[i] for i in splits["ids"]["train"]]
    val_rows = [by_id[i] for i in splits["ids"]["val"]]
    paper = job["kind"].startswith("paper")
    weighted = job["kind"] != "paper_literal"
    cache_key = (manifest_hash, job["kind"])
    feature_started = time.perf_counter()
    if cache_key not in FEATURES:
        if paper:
            FEATURES[cache_key] = {r["id"]: paper_features(r, weighted) for r in all_rows}
        else:
            info = manifest["corpora"]["full"]["features"]["A" if job["configuration"].get("baseline") else "B"]
            path = runner.RESULTS / info["path"]
            runner.require(runner.sha256(path) == info["sha256"], "Comparison feature cache differs")
            FEATURES[cache_key] = gnn.load_feature_cache(path)
    features = FEATURES[cache_key]
    feature_seconds = time.perf_counter() - feature_started
    example = features[train_rows[0]["id"]]
    architecture = {k: v for k, v in job["configuration"].items()
                    if k in ["baseline", "classification", "hypotheses", "hidden", "layers", "theta_classes"]}
    train_only_vocabulary = None
    if architecture.get("classification"):
        statistics = gnn.label_statistics(train_rows)
        values = gnn.classification_values(train_rows, statistics)
        previous = architecture.get("theta_classes", [0., float(np.pi / 4)])
        train_only_vocabulary = {"theta_classes": values, "theta_other_fraction": statistics["theta_other_fraction"],
                                 "matches_core_configuration": len(previous) == len(values) and bool(np.allclose(previous, values))}
        architecture["theta_classes"] = values
    if paper:
        model = PaperModel(example["node"].shape[-1], example["edge"].shape[-1])
        architecture = {"baseline": True, "hidden": 192, "layers": 5, "classification": False, "hypotheses": 1}
    elif job["kind"] == "diffusion":
        model = GraphDiffusion(example["node"].shape[-1], example["edge"].shape[-1])
        architecture = {"denoiser": "GPS", "hidden": DIFFUSION_HIDDEN, "layers": DIFFUSION_LAYERS,
                        "circle_coordinates": 2, "time_features": 8, "diffusion_steps": DIFFUSION_STEPS,
                        "sampling_steps": SAMPLING_STEPS, "hypotheses": 4}
    else:
        model = gnn.WarmStartModel(example["node"].shape[-1], example["edge"].shape[-1], **architecture)
        architecture.update({"hidden": model.node_input.out_features, "layers": len(model.layers)})
    architecture.update({"node_features": example["node"].shape[-1], "edge_features": example["edge"].shape[-1]})
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_weights = gnn.training_class_weights(train_rows, model.theta_classes.cpu().numpy(), "cuda") if model.classification else None
    history, best, best_loss = [], None, float("inf")
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng([job["seed"], epoch])
        order = rng.permutation(len(train_rows))
        losses = []
        epoch_started = time.perf_counter()
        for start in range(0, len(order), gnn.BATCH_SIZE):
            runner.require(time.time() < job["deadline_epoch"], "Comparison GPU resource deadline reached")
            selected = [train_rows[int(k)] for k in order[start:start + gnn.BATCH_SIZE]]
            data = feature_batch([features[r["id"]] for r in selected], device="cuda", random_signs=rng, baseline=model.baseline)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss(model, selected, data) if isinstance(model, GraphDiffusion) else model_loss(
                model, model(data), selected, data, paper, class_weights)
            runner.require(bool(torch.isfinite(loss)), "Nonfinite comparison loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = validation_loss(model, val_rows, features, paper, class_weights, job["deadline_epoch"])
        improved = validation < best_loss
        if improved:
            best_loss = validation
            best = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch,
                    "validation_loss": validation}
        history.append({"epoch": epoch, "train_batch_mean_loss": float(np.mean(losses)),
                        "validation_loss": validation, "selected": improved,
                        "elapsed_seconds": time.perf_counter() - epoch_started})
        runner.save_json(runner.RESULTS / "paper_comparison/training" / (job["name"] + "_progress.json"),
                         {"job": job, "history": history, "best_epoch": best["epoch"]})
        checkpoint_progress = runner.RESULTS / "paper_comparison/checkpoints" / (job["name"] + "_resume.pt")
        checkpoint_progress.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "best": best,
                    "job": job, "history": history, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, checkpoint_progress)
        if epoch % 5 == 0:
            runner.commit_volume(modal.Volume.from_name(runner.VOLUME_NAME))
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best["model"])
    checkpoint = runner.RESULTS / "paper_comparison/checkpoints" / (job["name"] + ".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**best, "job": job, "history": history, "architecture": architecture,
                "angle_units": PAPER_ANGLE_UNITS if paper else "filter theta=w*gamma and beta",
                "source_hashes": source_hashes, "corpus_manifest_sha256": manifest_hash,
                "train_ids_sha256": hashlib.sha256(json.dumps(splits["ids"]["train"]).encode()).hexdigest()}, checkpoint)
    predictions = {}
    sets = {"test_split": ("full", splits["ids"]["test"]), "large": ("large", None)}
    for tag, (corpus, ids) in sets.items():
        rows = [by_id[i] for i in ids] if corpus == "full" else read_large_all(runner, job["large_all"])
        if corpus == "full":
            cache = features
        elif paper:
            cache = {r["id"]: paper_features(r, weighted) for r in rows}
        else:
            info = manifest["corpora"][corpus]["features"]["A" if model.baseline else "B"]
            runner.require(runner.sha256(runner.RESULTS / info["path"]) == info["sha256"], "Larger feature cache differs")
            cache = gnn.load_feature_cache(runner.RESULTS / info["path"])
            for row in rows:
                if row["id"] not in cache:
                    cache[row["id"]] = gnn.graph_features(row, baseline=model.baseline)
        path = runner.RESULTS / "paper_comparison/predictions" / (job["name"] + "_" + tag + ".json")
        runner.save_json(path, predict(model, rows, cache, paper, job["seed"], job["deadline_epoch"]))
        predictions[tag] = {"path": path.relative_to(runner.RESULTS).as_posix(), "sha256": runner.sha256(path), "graphs": len(rows)}
    own_total_seconds = time.perf_counter() - started
    core_predictions, core_seconds = {}, 0.
    if job.get("core_checkpoint"):
        specification = job["core_checkpoint"]
        path = runner.RESULTS / specification["path"]
        runner.require(runner.sha256(path) == specification["sha256"], "Full-recipe GNN checkpoint differs")
        core_started = time.perf_counter()
        core_model, unused_kind = load_predictor(path, "cuda")
        for tag, (corpus, ids) in sets.items():
            rows = [by_id[i] for i in ids] if corpus == "full" else read_large_all(runner, job["large_all"])
            if corpus == "full":
                cache = features
            else:
                info = manifest["corpora"][corpus]["features"]["A" if core_model.baseline else "B"]
                runner.require(runner.sha256(runner.RESULTS / info["path"]) == info["sha256"], "Core inference feature cache differs")
                cache = gnn.load_feature_cache(runner.RESULTS / info["path"])
                for row in rows:
                    if row["id"] not in cache:
                        cache[row["id"]] = gnn.graph_features(row, baseline=core_model.baseline)
            output_path = runner.RESULTS / "paper_comparison/predictions" / ("core_final_s%d_%s.json" % (job["seed"], tag))
            runner.save_json(output_path, predict(core_model, rows, cache, False, job["seed"], job["deadline_epoch"]))
            core_predictions[tag] = {"path": output_path.relative_to(runner.RESULTS).as_posix(),
                                     "sha256": runner.sha256(output_path), "graphs": len(rows)}
        core_seconds = time.perf_counter() - core_started
    result = {"job": job, "history": history, "best_epoch": best["epoch"], "best_validation_loss": best_loss,
              "checkpoint": checkpoint.relative_to(runner.RESULTS).as_posix(), "checkpoint_sha256": runner.sha256(checkpoint),
              "predictions": predictions, "training_seconds": training_seconds,
              "training_and_prediction_seconds": own_total_seconds, "additional_core_prediction_seconds": core_seconds,
              "core_predictions": core_predictions,
              "feature_seconds": feature_seconds, "parameters": sum(p.numel() for p in model.parameters()),
              "train_graphs": len(train_rows), "validation_graphs": len(val_rows),
              "train_ids_sha256": hashlib.sha256(json.dumps(splits["ids"]["train"]).encode()).hexdigest(),
              "corpus_manifest_sha256": manifest_hash, "source_hashes": source_hashes,
              "train_only_vocabulary": train_only_vocabulary,
              "angle_units": PAPER_ANGLE_UNITS if paper else "filter theta=w*gamma and beta",
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "optimizer": "AdamW; constant lr=1e-3; weight_decay=1e-4; batch=256; clip=1; 30 epochs",
              "checkpoint_selection": "minimum native validation loss, no test selection"}
    runner.save_json(runner.RESULTS / "paper_comparison/training" / (job["name"] + ".json"), result)
    runner.commit_volume(modal.Volume.from_name(runner.VOLUME_NAME))
    return result


# Reconstruct a comparison or core checkpoint for inference, without submitting any jobs.
def load_predictor(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = payload.get("source_hashes", {}).get("src/gnn.py")
    if expected and hashlib.sha256(Path(gnn.__file__).read_bytes()).hexdigest() != expected:
        raise ValueError("The checkpoint's graph-feature/model implementation differs from this gnn.py")
    state = payload["model"]
    kind = payload.get("job", {}).get("kind", "core")
    if kind == "diffusion":
        node_features = state["network.node_input.weight"].shape[1] - 10
        edge_features = state["network.edge_input.weight"].shape[1] - 10
        model = GraphDiffusion(node_features, edge_features)
    elif kind.startswith("paper"):
        node_features, edge_features = state["node_input.weight"].shape[1], state["edge_input.weight"].shape[1]
        model = PaperModel(node_features, edge_features)
    else:
        node_features, edge_features = state["node_input.weight"].shape[1], state["edge_input.weight"].shape[1]
        config = payload.get("config", payload.get("architecture", {}))
        architecture = {key: value for key, value in config.items()
                        if key in ["baseline", "classification", "hypotheses", "hidden", "layers", "theta_classes"]}
        architecture["hidden"] = state["node_input.weight"].shape[0]
        architecture["layers"] = len({key.split(".")[1] for key in state if key.startswith("layers.")})
        architecture["theta_classes"] = state["theta_classes"].tolist()
        model = gnn.WarmStartModel(node_features, edge_features, **architecture)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), kind


# Predict edge gammas and node betas from a connected positive-weight graph, with no labels.
def predict_graph(checkpoint, graph, device="cpu", seed=42):
    import networkx as nx
    n = graph["n"]
    edges = graph["edges"]
    weights = graph.get("w_sorted", graph.get("weights"))
    if type(n) is not int or n < 2 or not isinstance(weights, list) or len(weights) != len(edges):
        raise ValueError("Provide n>=2, edges and one positive weight per edge")
    pairs = {}
    for edge, weight in zip(edges, weights):
        if len(edge) != 2 or any(type(v) is not int or not 0 <= v < n for v in edge) or edge[0] == edge[1]:
            raise ValueError("Edges must join distinct integer vertices in 0..n-1")
        key = tuple(sorted(edge))
        if key in pairs or not np.isfinite(weight) or weight <= 0:
            raise ValueError("Duplicate edges or nonpositive/nonfinite weights are unsupported")
        pairs[key] = float(weight)
    topology = nx.Graph()
    topology.add_nodes_from(range(n))
    topology.add_edges_from(pairs)
    if not nx.is_connected(topology):
        raise ValueError("These predictors were trained on connected weighted graphs")
    sorted_edges = sorted(pairs)
    row = {"id": int(graph.get("id", 0)), "n": n, "edges": sorted_edges,
           "w_sorted": [pairs[edge] for edge in sorted_edges]}
    model, kind = load_predictor(checkpoint, device)
    features = paper_features(row, kind != "paper_literal") if kind.startswith("paper") else gnn.graph_features(row, baseline=model.baseline)
    data = feature_batch([features], device=device, baseline=model.baseline)
    with torch.inference_mode():
        output = diffusion_sample(model, data, seed) if isinstance(model, GraphDiffusion) else model(data)
    theta, beta = output["theta"][0].cpu().numpy(), output["beta"][0].cpu().numpy()
    candidates = []
    m = len(sorted_edges)
    for k in range(model.hypotheses):
        paper = kind.startswith("paper")
        phase = -.5 * theta[k, :m] if paper else theta[k, :m]
        mixers = -beta[k, :n] if paper else beta[k, :n]
        candidates.append({"gamma": (phase / row["w_sorted"]).astype(float).tolist(),
                           "beta": mixers.astype(float).tolist(), "edge_phase": phase.astype(float).tolist()})
    return {"n": n, "edges": sorted_edges, "weights": row["w_sorted"], "p": 1, "kind": kind,
            "candidates": candidates, "seed": seed, "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "note": "Predictions only; no filter, energy evaluation, local optimization or Modal job was run."}


# Keep command-line inference local and make the default debugger action a usage message.
def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Predict p=1 MA-QAOA parameters for a new weighted graph")
    parser.add_argument("mode", choices=["predict"])
    parser.add_argument("checkpoint")
    parser.add_argument("graph", help="JSON containing n, edges, weights (or w_sorted)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(predict_graph(args.checkpoint, json.loads(Path(args.graph).read_text()), args.device, args.seed), indent=2))


if __name__ == "__main__":
    main()
