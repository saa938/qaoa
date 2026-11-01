"""
Usage: python src/gnn.py
Validate the batched differentiable p=1 circuit against run_filter before GPU training.
Angles are edge phases theta=w*gamma followed by node betas, all with period pi.
"""

import json
import os
import time
import copy
import csv
import math
import statistics
import hashlib
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODE = "check_energy"
OUT = ROOT / "results" / "gnn"
ENERGY_POINTS = 1000
ENERGY_SEED = 42
ENERGY_BATCH = 25
TORCH_VERSION = "2.15.0"
SPLIT_SEED = 42
CLASS_TOL = 1e-3
BOX_TOL = 1e-6
MODEL_SEED = 42
TRAIN_SEEDS = [42, 43, 44]
BAKEOFF_EPOCHS = 40
FINAL_EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
REGULARIZATION_WEIGHT = 0.1
RANDOM_TEST_STARTS = 50
RANDOM_VAL_STARTS = 20
ZERO_STARTS = 5
ZERO_NOISE = 0.05
BOOTSTRAP_SAMPLES = 1000
HIT_TOL = 1e-6
CLASSICAL_STATE_CACHE = {}

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
try:
    import torch
    from torch import nn
    from torch.nn import functional as functional
except ImportError:
    torch = None
    nn = None
    functional = None

MODEL_BASE = nn.Module if nn is not None else object


# Evolve independent graphs of one node count with the exact positive-phase convention.
def torch_energy(n, edges, weights, theta, beta):
    if torch is None:
        raise RuntimeError("The differentiable circuit requires the separate PyTorch environment")
    batch = theta.shape[0]
    dim = 1 << n
    indices = torch.arange(dim, device=theta.device)
    qubits = torch.arange(n, device=theta.device)
    spins = (1 - 2 * ((indices[:, None] >> qubits[None, :]) & 1)).to(theta.dtype)
    zz = (spins[:, edges[..., 0]] * spins[:, edges[..., 1]]).permute(1, 0, 2)
    present = (weights != 0).to(theta.dtype)
    phase = torch.einsum("bm,bsm->bs", theta * present, zz)
    psi = torch.exp(1j * phase) / np.sqrt(dim)
    for j in range(n):
        shaped = psi.reshape(batch, 1 << (n - j - 1), 2, 1 << j)
        a, b = shaped[:, :, 0, :], shaped[:, :, 1, :]
        cosine = torch.cos(beta[:, j])[:, None, None]
        sine = 1j * torch.sin(beta[:, j])[:, None, None]
        psi = torch.stack((cosine * a + sine * b, sine * a + cosine * b), dim=2).reshape(batch, dim)
    probability = psi.real.square() + psi.imag.square()
    return 0.5 * torch.einsum("bs,bsm,bm->b", probability, zz, weights) - 0.5 * weights.sum(dim=1)


# Pad variable edge counts within a graph-size cohort without adding artificial terms.
def circuit_tensors(rows, device="cpu", dtype=None):
    dtype = torch.float64 if dtype is None else dtype
    n = rows[0]["n"]
    if any(row["n"] != n for row in rows):
        raise ValueError("A circuit cohort must have one node count")
    maximum = max(len(row["edges"]) for row in rows)
    edges = torch.zeros((len(rows), maximum, 2), dtype=torch.long)
    weights = torch.zeros((len(rows), maximum), dtype=dtype)
    for k, row in enumerate(rows):
        m = len(row["edges"])
        edges[k, :m] = torch.as_tensor(row["edges"], dtype=torch.long)
        weights[k, :m] = torch.as_tensor(row["w_sorted"], dtype=dtype)
    return edges.to(device), weights.to(device)


# Compare every random point and every coordinate derivative against the adjoint circuit.
def check_energy(device="cpu"):
    import make_dataset as dataset
    import run_filter as run_filter
    import modal_app as runner
    runner.check_input_hashes()
    torch.set_num_threads(1)
    rng = np.random.default_rng(ENERGY_SEED)
    dataset.setup("full")
    checks = []
    started = time.monotonic()
    for n in range(5, 13):
        for start in range(0, ENERGY_POINTS // 8, ENERGY_BATCH):
            count = min(ENERGY_BATCH, ENERGY_POINTS // 8 - start)
            rows, points = [], []
            for index in range(count):
                graph_id = 9_000_000 + n * 1000 + start + index
                kind, edges, weights, audit = dataset.build(graph_id, n)
                rows.append({"n": n, "edges": edges, "w_sorted": weights.tolist()})
                points.append((rng.random(len(edges) + n) - 0.5) * np.pi)
            edges, weights = circuit_tensors(rows, device=device)
            theta = torch.zeros_like(weights)
            beta = torch.empty((count, n), dtype=torch.float64, device=device)
            for index, (row, point) in enumerate(zip(rows, points)):
                m = len(row["edges"])
                theta[index, :m] = torch.as_tensor(point[:m], device=device)
                beta[index] = torch.as_tensor(point[m:], device=device)
            theta.requires_grad_()
            beta.requires_grad_()
            energy = torch_energy(n, edges, weights, theta, beta)
            dtheta, dbeta = torch.autograd.grad(energy.sum(), (theta, beta))
            for index, (row, point) in enumerate(zip(rows, points)):
                m = len(row["edges"])
                w = np.asarray(row["w_sorted"])
                raw = np.r_[point[:m] / w, point[m:]]
                reference, unused_batch, grad, dimension = run_filter.make_energy(n, row["edges"], w, 1)
                value, derivative = grad(raw, True)
                correct = np.r_[derivative[:m] / w, derivative[m:]]
                actual = np.r_[dtheta[index, :m].detach().cpu().numpy(), dbeta[index].detach().cpu().numpy()]
                checks.append({"n": n, "energy_error": abs(float(energy[index].detach()) - value),
                               "gradient_error": float(np.max(np.abs(actual - correct)))})
        print("Torch circuit: n=%d, %d/%d random points checked" % (n, len(checks), ENERGY_POINTS), flush=True)
    result = {"checks": checks, "points": len(checks), "torch": torch.__version__, "device": device,
              "complex_dtype": "complex128", "max_energy_error": max(row["energy_error"] for row in checks),
              "max_gradient_error": max(row["gradient_error"] for row in checks),
              "elapsed_seconds": time.monotonic() - started}
    OUT.mkdir(parents=True, exist_ok=True)
    runner.save_json(OUT / "torch_energy_check.json", result)
    runner.save_json(OUT / ("torch_energy_check_" + str(device).replace(":", "_") + ".json"), result)
    runner.torch_gate(checks)
    print("PASS: max energy error %.3g, max phase-gradient error %.3g" %
          (result["max_energy_error"], result["max_gradient_error"]), flush=True)
    return result


# Describe discrete labels directly from usable records before choosing classification heads.
def label_statistics(rows):
    theta, beta, weights = [], [], []
    for row in rows:
        if row.get("usable") is not True:
            continue
        m = len(row["edges"])
        point = np.asarray(row["pick"])
        theta.extend((point[:m] * row["w_sorted"]).tolist())
        weights.extend(row["w_sorted"])
        beta.extend(point[m:].tolist())
    theta, beta, weights = np.asarray(theta), np.asarray(beta), np.asarray(weights)
    if not len(theta) or not len(beta):
        raise ValueError("No usable labels")
    edge_classes = np.array([0, np.pi / 4])
    node_classes = np.arange(-2, 3) * np.pi / 8
    edge_distances = np.abs(theta[:, None] - edge_classes)
    node_distances = np.abs(beta[:, None] - node_classes)
    return {"edge_count": len(theta), "node_count": len(beta),
            "theta_counts": {str(value): int((edge_distances[:, k] <= CLASS_TOL).sum())
                             for k, value in enumerate(edge_classes)},
            "theta_other_fraction": float((edge_distances.min(axis=1) > CLASS_TOL).mean()),
            "theta_negative_pi4": int((np.abs(theta + np.pi / 4) <= CLASS_TOL).sum()),
            "beta_counts": {str(value): int((node_distances[:, k] <= CLASS_TOL).sum())
                            for k, value in enumerate(node_classes)},
            "beta_other_fraction": float((node_distances.min(axis=1) > CLASS_TOL).mean()),
            "box_tolerance_units": "run_filter.reduce_point applies TIE_TOL=1e-6 in raw gamma; phase tolerance is w*TIE_TOL, beta tolerance is 1e-6",
            "fixed_phase_tolerance_violations": int((np.abs(theta) > np.pi / 4 + BOX_TOL).sum()
                                                     + (np.abs(beta) > np.pi / 4 + BOX_TOL).sum()),
            "max_phase_box_excess": float(max(0, (np.abs(theta) - np.pi / 4).max())),
            "max_raw_gamma_box_excess": float(max(0, ((np.abs(theta) - np.pi / 4) / weights).max())),
            "box_violations": int((np.abs(theta) > np.pi / 4 + weights * BOX_TOL).sum()
                                  + (np.abs(beta) > np.pi / 4 + BOX_TOL).sum())}


# Orient a repeated eigenspace using graph-invariant anchors rather than node order.
def invariant_eigenspace_basis(projector, anchors, dimension):
    candidates = projector @ anchors
    output = np.zeros((len(projector), dimension))
    for k in range(dimension):
        for repeat in range(2):
            if k:
                candidates -= output[:, :k] @ (output[:, :k].T @ candidates)
        norms = np.linalg.norm(candidates, axis=0)
        maximum = float(norms.max())
        if maximum <= 1e-9:
            break
        choices = np.flatnonzero(norms >= maximum - 1e-10 * max(1.0, maximum))
        column = int(choices[0])
        output[:, k] = candidates[:, column] / norms[column]
    return output


# Preserve informative Laplacian vectors with a deterministic equivariant spectral basis.
def laplacian_features(adjacency, weighted=None):
    n = len(adjacency)
    degree = adjacency.sum(axis=1)
    values, vectors = np.linalg.eigh(np.diag(degree) - adjacency)
    weighted = adjacency if weighted is None else weighted
    wdegree = weighted.sum(axis=1)
    spread = float(np.ptp(wdegree))
    z = 2 * (wdegree - wdegree.min()) / spread - 1 if spread > 1e-12 else np.zeros(n)
    moments = np.polynomial.chebyshev.chebvander(z, n - 1)
    structural = walk_features((weighted != 0).astype(float), weighted)
    anchors = np.c_[moments, weighted @ moments / np.maximum(wdegree[:, None], 1e-12), structural]
    anchors -= anchors.mean(axis=0)
    anchor_norms = np.linalg.norm(anchors, axis=0)
    anchors = np.divide(anchors, anchor_norms, out=np.zeros_like(anchors), where=anchor_norms > 1e-10)
    output = np.zeros((n, 8))
    index = 1
    while index < min(n, 9):
        scale = max(1.0, abs(values[index]))
        block = np.flatnonzero((np.arange(n) >= index) & (np.abs(values - values[index]) <= 1e-9 * scale))
        if len(block) > 1:
            projector = vectors[:, block] @ vectors[:, block].T
            basis = invariant_eigenspace_basis(projector, anchors, len(block))
            for k, column in enumerate(block):
                if column <= 8:
                    output[:, column - 1] = basis[:, k]
        else:
            vector = vectors[:, index]
            signs = list(vector @ anchors) + [float((vector ** 3).sum())]
            sign = next((np.sign(moment) for moment in signs if abs(moment) > 1e-10), 0.0)
            output[:, index - 1] = sign * vector
        index = int(block[-1]) + 1
    return output


# Compute both unweighted and weighted return probabilities for steps one to sixteen.
def walk_features(adjacency, weighted):
    result = []
    for matrix in [adjacency, weighted]:
        transition = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)
        power = np.eye(len(matrix))
        for step in range(16):
            power = power @ transition
            result.append(np.diag(power))
    return np.stack(result, axis=1)


# Build graph features without relying on node ids or the orientation of a sorted edge.
def graph_features(row, baseline=False):
    n = row["n"]
    edges = np.asarray(row["edges"], dtype=np.int64)
    weights = np.asarray(row["w_sorted"], dtype=float)
    u, v = edges.T
    adjacency = np.zeros((n, n))
    weighted = np.zeros((n, n))
    adjacency[u, v] = adjacency[v, u] = 1
    weighted[u, v] = weighted[v, u] = weights
    degree = adjacency.sum(axis=1)
    wdegree = weighted.sum(axis=1)
    pe = laplacian_features(adjacency if baseline else weighted, weighted=weighted)
    if baseline:
        keys = [(degree[j], round(float(wdegree[j]), 10)) for j in range(n)]
        order = sorted(set(keys))
        rank = np.array([sum(keys.count(key) for key in order if key < current)
                         + 0.5 * (keys.count(current) - 1) for current in keys])
        nodes = np.c_[degree / (n - 1), rank / (n - 1), pe]
        features = np.c_[degree[u] / (n - 1), degree[v] / (n - 1), pe[u], pe[v], weights]
        reverse = np.c_[degree[v] / (n - 1), degree[u] / (n - 1), pe[v], pe[u], weights]
    else:
        triangles = np.diag(adjacency @ adjacency @ adjacency) / 2
        clustering = np.divide(2 * triangles, degree * (degree - 1), out=np.zeros(n), where=degree > 1)
        mean_incident = wdegree / degree
        nodes = np.c_[degree / (n - 1), wdegree, wdegree / mean_incident,
                      np.full(n, 1 / n), triangles, clustering, walk_features(adjacency, weighted), pe]
        maximum = weighted.max(axis=1)
        relative = np.c_[weights / maximum[u], weights / maximum[v]]
        ranks = np.zeros((len(edges), 2))
        for k, (a, b) in enumerate(edges):
            for side, node in enumerate([a, b]):
                incident = weighted[node, adjacency[node] != 0]
                rank = (incident < weights[k]).sum() + 0.5 * ((incident == weights[k]).sum() - 1)
                ranks[k, side] = rank / max(1, len(incident) - 1)
        features = np.c_[weights, weights / weights.mean(), relative.sum(axis=1), np.abs(np.diff(relative, axis=1)),
                         ranks.sum(axis=1), np.abs(np.diff(ranks, axis=1)), degree[u] + degree[v],
                         np.abs(degree[u] - degree[v]), wdegree[u] + wdegree[v], np.abs(wdegree[u] - wdegree[v]),
                         (adjacency[u] * adjacency[v]).sum(axis=1), pe[u] + pe[v], np.abs(pe[u] - pe[v]), pe[u] * pe[v]]
        reverse = features.copy()
    return {"n": n, "edges": edges, "weights": weights, "node": nodes.astype(np.float32),
            "edge": features.astype(np.float32), "reverse_edge": reverse.astype(np.float32)}


# Pad a minibatch and apply one coherent eigenvector sign augmentation per graph.
def feature_batch(features, device="cpu", random_signs=None, baseline=False):
    batch = len(features)
    nodes = max(row["n"] for row in features)
    edges = max(len(row["edges"]) for row in features)
    result = {"node": torch.zeros(batch, nodes, features[0]["node"].shape[1]),
              "edge": torch.zeros(batch, edges, features[0]["edge"].shape[1]),
              "reverse_edge": torch.zeros(batch, edges, features[0]["edge"].shape[1]),
              "index": torch.zeros(batch, edges, 2, dtype=torch.long),
              "node_mask": torch.zeros(batch, nodes, dtype=torch.bool),
              "edge_mask": torch.zeros(batch, edges, dtype=torch.bool)}
    for k, row in enumerate(features):
        n, m = row["n"], len(row["edges"])
        node = row["node"].copy()
        edge, reverse = row["edge"].copy(), row["reverse_edge"].copy()
        if random_signs is not None:
            signs = random_signs.choice([-1.0, 1.0], size=8)
            node[:, -8:] *= signs
            if baseline:
                edge[:, 2:10] *= signs
                edge[:, 10:18] *= signs
                reverse[:, 2:10] *= signs
                reverse[:, 10:18] *= signs
            else:
                edge[:, -24:-16] *= signs
                reverse[:, -24:-16] *= signs
        result["node"][k, :n] = torch.as_tensor(node)
        result["edge"][k, :m] = torch.as_tensor(edge)
        result["reverse_edge"][k, :m] = torch.as_tensor(reverse)
        result["index"][k, :m] = torch.as_tensor(row["edges"])
        result["node_mask"][k, :n] = True
        result["edge_mask"][k, :m] = True
    return {name: value.to(device) for name, value in result.items()}


# Identify the graph-feature implementation independently of unrelated source changes.
def feature_implementation_hash():
    source = "".join(inspect.getsource(function) for function in
                     [invariant_eigenspace_basis, laplacian_features, walk_features, graph_features, feature_batch])
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# Store ragged numerical features without object arrays or executable serialization.
def save_feature_cache(rows, baseline, path):
    features = [graph_features(row, baseline=baseline) for row in rows]
    arrays = {"ids": np.asarray([row["id"] for row in rows], dtype=np.int64),
              "node_offsets": np.r_[0, np.cumsum([row["n"] for row in rows])],
              "edge_offsets": np.r_[0, np.cumsum([len(row["edges"]) for row in rows])],
              "implementation_hash": np.asarray(feature_implementation_hash()),
              "baseline": np.asarray(baseline)}
    for field in ["node", "edge", "reverse_edge", "edges", "weights"]:
        arrays[field] = np.concatenate([row[field] for row in features])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"graphs": len(rows), "baseline": baseline, "implementation_hash": feature_implementation_hash()}


# Rebuild graph views from a numerical cache while retaining canonical edge order.
def load_feature_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if str(arrays["implementation_hash"]) != feature_implementation_hash():
        raise ValueError("Cached graph features use a different implementation")
    result = {}
    for k, graph_id in enumerate(arrays["ids"]):
        a, b = (int(value) for value in arrays["node_offsets"][k:k + 2])
        u, v = (int(value) for value in arrays["edge_offsets"][k:k + 2])
        row = {"n": b - a, "node": arrays["node"][a:b]}
        row.update({field: arrays[field][u:v] for field in ["edge", "reverse_edge", "edges", "weights"]})
        result[int(graph_id)] = row
    if len(result) != len(arrays["ids"]):
        raise ValueError("Feature cache repeats a graph id")
    return result


# Verify exact classical cuts using bits rather than the builder's spin matrix.
def independent_max_cut(row):
    n = row["n"]
    if n not in CLASSICAL_STATE_CACHE:
        CLASSICAL_STATE_CACHE[n] = np.arange(1 << n, dtype=np.uint32)
    states = CLASSICAL_STATE_CACHE[n]
    cut = np.zeros(len(states))
    for (u, v), weight in zip(row["edges"], row["w_sorted"]):
        cut += weight * (((states >> u) ^ (states >> v)) & 1)
    return float(cut.max())


# Aggregate GINE messages in both edge directions while ignoring padded edges.
class GINE(MODEL_BASE):
    # Initialize the learnable self weight and message transformation.
    def __init__(self, hidden):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden))

    # Apply shared edge-conditioned messages to each endpoint.
    def forward(self, node, edge, reverse, index, mask):
        hidden = node.shape[-1]
        u = index[..., 0, None].expand(-1, -1, hidden)
        v = index[..., 1, None].expand(-1, -1, hidden)
        hu, hv = node.gather(1, u), node.gather(1, v)
        active = mask[..., None]
        aggregate = torch.zeros_like(node)
        aggregate.scatter_add_(1, u, functional.relu(hv + reverse) * active)
        aggregate.scatter_add_(1, v, functional.relu(hu + edge) * active)
        return self.mlp((1 + self.epsilon) * node + aggregate)


# Combine local messages with full attention among nodes of each graph.
class GPS(MODEL_BASE):
    # Initialize the specified eight-head attention and residual feed-forward blocks.
    def __init__(self, hidden, heads=8, dropout=0.1):
        super().__init__()
        self.local = GINE(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for k in range(3)])
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))

    # Keep padded nodes out of attention and every subsequent graph representation.
    def forward(self, node, edge, reverse, data):
        node = self.norms[0](node + self.drop(self.local(node, edge, reverse, data["index"], data["edge_mask"])))
        attention = self.attention(node, node, node, key_padding_mask=~data["node_mask"], need_weights=False)[0]
        node = self.norms[1](node + self.drop(attention))
        node = self.norms[2](node + self.drop(self.ff(node)))
        return node * data["node_mask"][..., None]


# Provide the paper-style and global encoders with one or four circular or discrete heads.
class WarmStartModel(MODEL_BASE):
    # Construct the configured bake-off variant without choosing classes from test labels.
    def __init__(self, node_features, edge_features, baseline=False, classification=False,
                 hypotheses=1, hidden=None, layers=None, theta_classes=None):
        super().__init__()
        if torch is None:
            raise RuntimeError("Model construction requires the separate PyTorch environment")
        self.baseline, self.classification, self.hypotheses = baseline, classification, hypotheses
        hidden = hidden or (192 if baseline else 256)
        layers = layers or (5 if baseline else 6)
        self.node_input = nn.Linear(node_features, hidden)
        self.edge_input = nn.Linear(edge_features, hidden)
        self.layers = nn.ModuleList([GINE(hidden) if baseline else GPS(hidden) for k in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for k in range(layers)]) if baseline else None
        self.drop = nn.Dropout(0.1)
        self.register_buffer("theta_classes", torch.tensor([0.0, np.pi / 4] if theta_classes is None else theta_classes,
                                                           dtype=torch.float32))
        self.register_buffer("beta_classes", torch.tensor(np.arange(-2, 3) * np.pi / 8, dtype=torch.float32))
        edge_outputs = len(self.theta_classes) if classification else 2
        node_outputs = len(self.beta_classes) if classification else 2
        self.edge_outputs, self.node_outputs = edge_outputs, node_outputs
        self.theta_head = nn.Sequential(nn.Linear((3 if baseline else 4) * hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hypotheses * edge_outputs))
        self.beta_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hypotheses * node_outputs))

    # Return all candidate angles and the raw logits needed for graph-normalized losses.
    def forward(self, data, expected_angles=False):
        node = self.node_input(data["node"])
        edge = self.edge_input(data["edge"])
        reverse = self.edge_input(data["reverse_edge"])
        for k, layer in enumerate(self.layers):
            if self.baseline:
                updated = node + self.drop(layer(node, edge, reverse, data["index"], data["edge_mask"]))
                active = data["node_mask"]
                node = updated.clone()
                node[active] = self.norms[k](updated[active])
                node = functional.relu(node) * active[..., None]
            else:
                node = layer(node, edge, reverse, data)
        hidden = node.shape[-1]
        u = data["index"][..., 0, None].expand(-1, -1, hidden)
        v = data["index"][..., 1, None].expand(-1, -1, hidden)
        hu, hv = node.gather(1, u), node.gather(1, v)
        directions = None
        if self.baseline:
            forward = self.theta_head(torch.cat((hu, hv, edge), dim=-1))
            backward = self.theta_head(torch.cat((hv, hu, reverse), dim=-1))
            edge_raw = 0.5 * (forward + backward)
            directions = [value.reshape(*value.shape[:2], self.hypotheses, self.edge_outputs).transpose(1, 2)
                          for value in [forward, backward]]
        else:
            edge_raw = self.theta_head(torch.cat((hu + hv, hu * hv, (hu - hv).abs(), edge), dim=-1))
        edge_raw = edge_raw.reshape(*edge_raw.shape[:2], self.hypotheses, self.edge_outputs).transpose(1, 2)
        node_raw = self.beta_head(node).reshape(*node.shape[:2], self.hypotheses, self.node_outputs).transpose(1, 2)
        if self.classification:
            if expected_angles:
                theta = (edge_raw.softmax(-1) * self.theta_classes).sum(-1)
                beta = (node_raw.softmax(-1) * self.beta_classes).sum(-1)
            else:
                theta = self.theta_classes[edge_raw.argmax(-1)]
                beta = self.beta_classes[node_raw.argmax(-1)]
        else:
            theta = torch.atan2(edge_raw[..., 1], edge_raw[..., 0]) / 2
            beta = torch.atan2(node_raw[..., 1], node_raw[..., 0]) / 2
        return {"theta": theta, "beta": beta, "edge_raw": edge_raw, "node_raw": node_raw,
                "edge_directions": directions}


# Average each angle-block loss within its graph before comparing hypotheses.
def label_loss(model, prediction, target_theta, target_beta, data, edge_class_weights=None, node_class_weights=None):
    masks = [data["edge_mask"], data["node_mask"]]
    targets = [target_theta, target_beta]
    raw_values = [prediction["edge_raw"], prediction["node_raw"]]
    class_values = [model.theta_classes, model.beta_classes]
    class_weights = [edge_class_weights, node_class_weights]
    total = None
    for side, (raw, target, mask) in enumerate(zip(raw_values, targets, masks)):
        if model.classification:
            labels = (target[..., None] - class_values[side]).abs().argmin(dim=-1)
            if side == 0 and len(model.theta_classes) > 2:
                other = (target[..., None] - model.theta_classes[:2]).abs().min(dim=-1).values > CLASS_TOL
                labels = torch.where(other, len(model.theta_classes) - 1, labels)
            labels = labels[:, None, :].expand(-1, model.hypotheses, -1)
            loss = functional.cross_entropy(raw.permute(0, 3, 1, 2), labels,
                                             weight=class_weights[side], reduction="none")
        else:
            directions = prediction["edge_directions"] if side == 0 and prediction["edge_directions"] is not None else [raw]
            terms = []
            for value in directions:
                angle = torch.atan2(value[..., 1], value[..., 0]) / 2
                terms.append(1 - torch.cos(2 * (angle - target[:, None, :]))
                             + REGULARIZATION_WEIGHT * (torch.linalg.vector_norm(value, dim=-1) - 1).abs())
            loss = torch.stack(terms).mean(dim=0)
        block = (loss * mask[:, None, :]).sum(dim=-1) / mask.sum(dim=-1)[:, None].clamp_min(1)
        total = block if total is None else total + block
    return total.min(dim=1).values.mean()


# Polish with the shared tolerances and obtain the initial energy from the first gradient call.
def measured_polish(energy, gradient, start, floor):
    import run_filter
    initial = []

    # Capture the optimizer's first joint energy-gradient evaluation without extra circuit calls.
    def recorded_gradient(point, with_energy=False):
        result = gradient(point, with_energy)
        if with_energy and not initial:
            initial.append(float(result[0]))
        return result

    result = run_filter.polish(energy, recorded_gradient, np.asarray(start, dtype=float))
    evaluations = int(result.nfev)
    return {"energy_before": initial[0], "energy_after": float(result.fun), "nfev": evaluations,
            "circuit_evaluations": evaluations * (2 * len(start) + 1),
            "hit": bool(result.fun <= floor + HIT_TOL), "angles_after": result.x.tolist()}


# Estimate random restart cost while explicitly retaining zero-hit lower bounds.
def random_cost(runs):
    if not runs:
        raise ValueError("Random restart evidence is empty")
    successes = sum(run["hit"] for run in runs)
    cost = sum(run["circuit_evaluations"] for run in runs)
    return {"cost": cost / successes if successes else float(cost), "hit_fraction": successes / len(runs),
            "zero_hit_lower_bound": successes == 0, "starts": len(runs), "total_nfev": sum(run["nfev"] for run in runs)}


# Build paired random, noisy-zero and oracle baselines once for each graph.
def graph_baselines(row, restarts=RANDOM_TEST_STARTS, seed=MODEL_SEED):
    import run_filter
    n, edges, w = row["n"], row["edges"], np.asarray(row["w_sorted"])
    m = len(edges)
    energy, unused_batch, gradient, dimension = run_filter.make_energy(n, edges, w, 1)
    rng = np.random.default_rng([seed, int(row["id"])])
    period = run_filter.periods(m, n, 1, w)
    random = [measured_polish(energy, gradient, (rng.random(dimension) - 0.5) * period, row["floor"])
              for k in range(restarts)]
    zero = []
    for k in range(ZERO_STARTS):
        phase_start = rng.normal(0.0, ZERO_NOISE, dimension)
        start = np.r_[phase_start[:m] / w, phase_start[m:]]
        zero.append(measured_polish(energy, gradient, start, row["floor"]))
    oracle = measured_polish(energy, gradient, row["pick"], row["floor"])
    return {"id": row["id"], "n": n, "random": random, "random_cost": random_cost(random),
            "zero": zero, "oracle": oracle}


# Rank raw or snapped candidates and charge every energy-only ranking evaluation.
def model_start(row, phase_candidates, baseline, strategy="better"):
    import run_filter
    n, edges, w = row["n"], row["edges"], np.asarray(row["w_sorted"])
    m = len(edges)
    energy, unused_batch, gradient, dimension = run_filter.make_energy(n, edges, w, 1)
    ranked, ranking_cost = [], 0
    for candidate in phase_candidates:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != (dimension,) or not np.isfinite(candidate).all():
            raise ValueError("Prediction shape or finiteness check failed")
        raw = np.r_[candidate[:m] / w, candidate[m:]]
        snapped_phase = np.round(candidate / (np.pi / 8)) * (np.pi / 8)
        snapped = np.r_[snapped_phase[:m] / w, snapped_phase[m:]]
        choices = []
        if strategy in ["raw", "better"]:
            choices.append((energy(raw), raw, "raw"))
            ranking_cost += 1
        if strategy in ["snap", "better"]:
            choices.append((energy(snapped), snapped, "snap"))
            ranking_cost += 1
        if not choices:
            raise ValueError("Unknown prediction strategy")
        ranked.append(min(choices, key=lambda value: value[0]))
    ranked.sort(key=lambda value: value[0])
    polishes = []
    for initial_energy, candidate, chosen in ranked:
        result = measured_polish(energy, gradient, candidate, row["floor"])
        result["chosen"] = chosen
        result["candidate_phase"] = np.r_[candidate[:m] * w, candidate[m:]].tolist()
        polishes.append(result)
        if result["hit"]:
            break
    hit = any(result["hit"] for result in polishes)
    start_cost = ranking_cost + sum(result["circuit_evaluations"] for result in polishes)
    reference = baseline["random_cost"]["cost"]
    expected = start_cost + (0 if hit else reference)
    best = min(polishes, key=lambda result: result["energy_after"])
    return {"id": row["id"], "n": n, "type": row["type"], "hit": hit,
            "ranking_evaluations": ranking_cost, "nfev": sum(result["nfev"] for result in polishes),
            "circuit_evaluations": start_cost, "expected_cost": expected, "cost_ratio": expected / reference,
            "zero_random_hit_lower_bound": baseline["random_cost"]["zero_hit_lower_bound"],
            "energy_ratio_before": ranked[0][0] / row["floor"],
            "approximation_ratio_after": -best["energy_after"] / row["max_cut"],
            "energy_before": ranked[0][0], "energy_after": best["energy_after"],
            "polishes": polishes}


# Bootstrap complete graphs rather than individual starts or angle coordinates.
def metric_summary(rows, expected_ids, seed=MODEL_SEED):
    records = {row["id"]: row for row in rows}
    if len(records) != len(rows) or set(records) - set(expected_ids):
        raise ValueError("Unexpected or duplicate evaluation ids")
    missing = sorted(set(expected_ids) - set(records))
    if missing:
        raise ValueError("Missing evaluation results count as failures: %s" % missing)
    ordered = [records[graph_id] for graph_id in expected_ids]
    ratios = np.asarray([row["cost_ratio"] for row in ordered])
    hit = np.asarray([row["hit"] for row in ordered], dtype=float)
    before = np.asarray([row["energy_ratio_before"] for row in ordered])
    after = np.asarray([row["approximation_ratio_after"] for row in ordered])
    nfev = np.asarray([row["nfev"] for row in ordered])
    if not len(ratios) or not np.isfinite(np.r_[ratios, hit, before, after, nfev]).all() or (ratios <= 0).any():
        raise ValueError("Evaluation metrics must be nonempty, finite and have positive costs")
    rng = np.random.default_rng(seed)
    samples = []
    for k in range(BOOTSTRAP_SAMPLES):
        draw = rng.integers(len(ordered), size=len(ordered))
        samples.append([hit[draw].mean(), np.median(ratios[draw]), np.exp(np.log(ratios[draw]).mean()),
                        before[draw].mean(), after[draw].mean(), nfev[draw].mean()])
    bounds = np.quantile(np.asarray(samples), [0.025, 0.975], axis=0)
    values = [hit.mean(), np.median(ratios), np.exp(np.log(ratios).mean()), before.mean(), after.mean(), nfev.mean()]
    independent = [math.fsum(hit) / len(hit), statistics.median(ratios.tolist()),
                   math.exp(math.fsum(math.log(value) for value in ratios) / len(ratios)),
                   math.fsum(before) / len(before), math.fsum(after) / len(after), math.fsum(nfev) / len(nfev)]
    if not np.allclose(values, independent, rtol=1e-12, atol=1e-12):
        raise ValueError("Independent metric aggregations disagree")
    for k in range(len(values)):
        ordered_samples = sorted(sample[k] for sample in samples)
        for j, probability in enumerate([.025, .975]):
            index = probability * (len(ordered_samples) - 1)
            left, right = int(math.floor(index)), int(math.ceil(index))
            value = ordered_samples[left] + (index - left) * (ordered_samples[right] - ordered_samples[left])
            if not math.isclose(value, bounds[j, k], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Independent bootstrap quantile calculations disagree")
    names = ["hit_rate", "median_cost_ratio", "geometric_mean_cost_ratio", "mean_energy_ratio_before",
             "mean_approximation_ratio_after", "mean_nfev"]
    return {"graphs": len(ordered), "zero_random_hit_graphs": sum(row["zero_random_hit_lower_bound"] for row in ordered),
            "metrics": {name: {"value": float(value), "low95": float(bounds[0, k]), "high95": float(bounds[1, k])}
                        for k, (name, value) in enumerate(zip(names, values))}}


# Assign entire unweighted WL groups while approaching the requested row proportions.
def grouped_split(rows, holdout_hashes=None):
    usable = [row for row in rows if row.get("usable") is True]
    if not usable or len({row["id"] for row in usable}) != len(usable):
        raise ValueError("Split input is empty or has duplicate usable ids")
    if any(not 5 <= row["n"] <= 12 or not row.get("wl_hash") for row in usable):
        raise ValueError("Only the weighted training-size corpus belongs in the grouped split")
    groups, group_sizes = {}, {}
    for row in usable:
        groups.setdefault(row["wl_hash"], []).append(row["id"])
        if row["wl_hash"] in group_sizes and group_sizes[row["wl_hash"]] != row["n"]:
            raise ValueError("One WL group contains inconsistent node counts")
        group_sizes[row["wl_hash"]] = row["n"]
    forced = set(holdout_hashes or []) & set(groups)
    rng = np.random.default_rng(SPLIT_SEED)
    priority = {key: float(rng.random()) for key in sorted(groups)}
    splits = {name: [] for name in ["train", "val", "test"]}
    split_groups = {name: [] for name in splits}
    names = list(splits)
    forced_count = sum(len(groups[key]) for key in forced)
    test_target = max(.1 * len(usable), forced_count)
    val_target = .1 * len(usable)
    train_target = len(usable) - test_target - val_target
    free_count = len(usable) - forced_count
    if train_target <= 0 or free_count <= 0:
        raise ValueError("Structural holdouts leave too little data for training and validation")
    by_size = {}
    for n in sorted({row["n"] for row in usable}):
        keys = [key for key in groups if group_sizes[key] == n]
        forced_n = sum(len(groups[key]) for key in keys if key in forced)
        available = sum(len(groups[key]) for key in keys) - forced_n
        targets = np.array([train_target, val_target, test_target - forced_count]) * available / free_count
        counts = np.zeros(3)
        for key in sorted(keys, key=lambda value: (-len(groups[value]), priority[value])):
            if key in forced:
                splits["test"].extend(groups[key])
                split_groups["test"].append(key)
                continue
            index = int(np.argmax(targets - counts))
            name = names[index]
            splits[name].extend(groups[key])
            split_groups[name].append(key)
            counts[index] += len(groups[key])
        by_size[n] = {name: int(counts[k]) + (forced_n if name == "test" else 0) for k, name in enumerate(names)}
    for name in splits:
        splits[name].sort()
    if any(set(split_groups[a]) & set(split_groups[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
        raise ValueError("An unweighted structure leaks between splits")
    if set(sum(splits.values(), [])) != {row["id"] for row in usable}:
        raise ValueError("Split id coverage differs from the usable corpus")
    if not forced <= set(split_groups["test"]):
        raise ValueError("An extra-test topology leaked into training or validation")
    if len(splits["val"]) < 2000:
        raise ValueError("Validation split has fewer than the required 2000 warm-start selection graphs")
    subset_rng = np.random.default_rng(SPLIT_SEED)
    splits["val_selection"] = sorted(int(value) for value in subset_rng.choice(splits["val"], size=2000, replace=False))
    return {"seed": SPLIT_SEED, "ids": splits, "wl_groups": split_groups,
            "counts": {name: len(ids) for name, ids in splits.items()}, "expected_usable": len(usable),
            "counts_by_n": by_size, "forced_test_hashes": sorted(forced), "forced_test_graphs": forced_count,
            "requested_proportions": [.8, .1, .1], "extra_test_holdout_hashes": sorted(set(holdout_hashes or []))}


# Set any other-class angle from training labels rather than validation or test labels.
def classification_values(train_rows, full_statistics):
    values = [0.0, float(np.pi / 4)]
    if full_statistics["theta_other_fraction"] > 0.02:
        other = []
        for row in train_rows:
            m = len(row["edges"])
            phase = np.asarray(row["pick"][:m]) * row["w_sorted"]
            other.extend(phase[np.abs(phase[:, None] - np.asarray(values)).min(axis=1) > CLASS_TOL])
        if not other:
            raise ValueError("An other class is required but absent from training labels")
        values.append(float(np.arctan2(np.sin(2 * np.asarray(other)).mean(), np.cos(2 * np.asarray(other)).mean()) / 2))
    return values


# Derive optional inverse-square-root class weights only from the training labels.
def training_class_weights(rows, theta_classes, device):
    classes = [np.asarray(theta_classes), np.arange(-2, 3) * np.pi / 8]
    counts = [np.zeros(len(value), dtype=int) for value in classes]
    for row in rows:
        m = len(row["edges"])
        values = [np.asarray(row["pick"][:m]) * row["w_sorted"], np.asarray(row["pick"][m:])]
        for block, (angles, candidates) in enumerate(zip(values, classes)):
            distance = np.abs(angles[:, None] - candidates)
            index = distance.argmin(axis=1)
            if block == 0 and len(candidates) > 2:
                index[distance[:, :2].min(axis=1) > CLASS_TOL] = len(candidates) - 1
            counts[block] += np.bincount(index, minlength=len(candidates))
    weights = []
    for count in counts:
        positive = count[count > 0]
        if positive.max() / positive.min() <= 3:
            weights.append(None)
        else:
            value = np.zeros(len(count))
            value[count > 0] = 1 / np.sqrt(count[count > 0])
            value /= value[count > 0].mean()
            weights.append(torch.as_tensor(value, dtype=torch.float32, device=device))
    return weights


# Combine independent graph-size cohorts in the normalized differentiable energy loss.
def energy_loss(rows, prediction):
    if prediction["theta"].shape[1] != 1:
        raise ValueError("Energy training is specified for the single-hypothesis variants")
    total = prediction["theta"].sum() * 0
    for n in sorted({row["n"] for row in rows}):
        indices = [k for k, row in enumerate(rows) if row["n"] == n]
        cohort = [rows[k] for k in indices]
        device = prediction["theta"].device
        edges, weights = circuit_tensors(cohort, device, prediction["theta"].dtype)
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        theta = prediction["theta"].index_select(0, index)[:, 0, :weights.shape[1]]
        beta = prediction["beta"].index_select(0, index)[:, 0, :n]
        energy = torch_energy(n, edges, weights, theta, beta)
        floor = torch.as_tensor([row["floor"] for row in cohort], dtype=energy.dtype, device=device)
        total = total + ((energy - floor) / weights.sum(dim=1)).sum() / len(rows)
    return total


# Train one reproducible epoch while leaving optimizer-based validation on CPU workers.
def train_epoch(model, optimizer, rows, feature_cache, epoch, seed, maximum_epochs,
                label_weight=1.0, energy_weight=0.0, device="cuda", class_weights=None):
    model.train()
    rng = np.random.default_rng([seed, epoch])
    order = rng.permutation(len(rows))
    steps = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    total, seen = 0.0, 0
    for batch_index, start in enumerate(range(0, len(order), BATCH_SIZE)):
        selected = [rows[int(k)] for k in order[start:start + BATCH_SIZE]]
        features = [feature_cache[row["id"]] for row in selected]
        data = feature_batch(features, device=device, random_signs=rng, baseline=model.baseline)
        if not model.baseline:
            step = (epoch - 1) * steps + batch_index
            warmup = min(5 * steps, max(1, maximum_epochs * steps // 10))
            progress = max(0.0, (step - warmup) / max(1, maximum_epochs * steps - warmup))
            rate = LEARNING_RATE * min(1.0, (step + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))
            for group in optimizer.param_groups:
                group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data, expected_angles=bool(energy_weight))
        loss = prediction["theta"].sum() * 0
        if label_weight:
            theta = torch.zeros(data["edge_mask"].shape, dtype=torch.float32)
            beta = torch.zeros(data["node_mask"].shape, dtype=torch.float32)
            for k, row in enumerate(selected):
                m, n = len(row["edges"]), row["n"]
                theta[k, :m] = torch.as_tensor(np.asarray(row["pick"][:m]) * row["w_sorted"], dtype=torch.float32)
                beta[k, :n] = torch.as_tensor(row["pick"][m:], dtype=torch.float32)
            theta, beta = theta.to(device), beta.to(device)
            weights = [None, None] if class_weights is None else class_weights
            loss = loss + label_weight * label_loss(model, prediction, theta, beta, data, *weights)
        if energy_weight:
            loss = loss + energy_weight * energy_loss(selected, prediction)
        if not torch.isfinite(loss):
            raise ValueError("Training produced a nonfinite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(selected)
        seen += len(selected)
    return {"epoch": epoch, "graphs": seen, "loss": total / seen}


# Predict phases for a fixed manifest without accessing labels or their energy floors.
def predict_rows(model, rows, feature_cache, device="cuda"):
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            selected = rows[start:start + BATCH_SIZE]
            features = [feature_cache[row["id"]] for row in selected]
            data = feature_batch(features, device=device, baseline=model.baseline)
            prediction = model(data)
            for k, row in enumerate(selected):
                m, n = len(row["edges"]), row["n"]
                phases = torch.cat((prediction["theta"][k, :, :m], prediction["beta"][k, :, :n]), dim=-1)
                records.append({"id": row["id"], "n": n, "phase_candidates": phases.cpu().tolist()})
    if len({row["id"] for row in records}) != len(rows):
        raise ValueError("Prediction manifest contains duplicate ids")
    return records


# Measure coordinate accuracy directly and graph distance under the supplied exact symmetries.
def label_metrics(row, phase):
    import make_dataset
    phase = np.asarray(phase, dtype=float)
    m = len(row["edges"])
    label = np.asarray(row["pick"], dtype=float)
    target = np.r_[label[:m] * row["w_sorted"], label[m:]]
    snapped = np.round(phase / (np.pi / 8)) * np.pi / 8
    delta = np.abs((snapped - target + np.pi / 2) % np.pi - np.pi / 2)
    raw = np.r_[phase[:m] / row["w_sorted"], phase[m:]]
    distance = make_dataset.pick_dist(row["n"], row["edges"], row["w_sorted"], raw, label)
    classes = np.arange(-2, 3) * np.pi / 8
    targets = np.abs(target[m:, None] - classes).argmin(axis=1)
    predicted = np.abs(snapped[m:, None] - classes).argmin(axis=1)
    leaves = np.abs(np.abs(target[m:]) - np.pi / 4) <= CLASS_TOL
    return {"pick_dist": float(distance), "within_1e3": bool(distance <= CLASS_TOL),
            "edge_correct": int((delta[:m] <= CLASS_TOL).sum()), "edges": m,
            "node_correct": int((delta[m:] <= CLASS_TOL).sum()), "nodes": row["n"],
            "beta_class_total": [int((targets == k).sum()) for k in range(5)],
            "beta_class_correct": [int(((targets == k) & (predicted == k)).sum()) for k in range(5)],
            "leaf_sign_total": int(leaves.sum()),
            "leaf_sign_correct": int((np.sign(snapped[m:][leaves]) == np.sign(target[m:][leaves])).sum())}


# Convert every noisy-start draw into the same expected-cost convention as model starts.
def baseline_metric(row, baseline, method):
    runs = baseline[method] if method in ["random", "zero"] else [baseline["oracle"]]
    reference = baseline["random_cost"]["cost"]
    hit = float(np.mean([run["hit"] for run in runs]))
    start_cost = float(np.mean([run["circuit_evaluations"] for run in runs]))
    expected = reference if method == "random" else start_cost + (1 - hit) * reference
    return {"id": row["id"], "n": row["n"], "type": row["type"], "hit": hit,
            "nfev": float(np.mean([run["nfev"] for run in runs])),
            "circuit_evaluations": start_cost, "expected_cost": expected, "cost_ratio": expected / reference,
            "zero_random_hit_lower_bound": baseline["random_cost"]["zero_hit_lower_bound"],
            "energy_ratio_before": float(np.mean([run["energy_before"] / row["floor"] for run in runs])),
            "approximation_ratio_after": float(np.mean([-run["energy_after"] / row["max_cut"] for run in runs]))}


# Aggregate coordinate counts while bootstrapping complete graphs and retaining absent classes.
def label_summary(records, expected_ids, seed=MODEL_SEED):
    indexed = {row["id"]: row for row in records}
    if len(indexed) != len(records) or set(indexed) != set(expected_ids):
        raise ValueError("Label accuracy requires the complete expected prediction manifest")
    labels = [indexed[key]["label_metrics"] for key in expected_ids]

    # Pool a graph resample without treating coordinates from one graph as independent.
    def values(indices):
        chosen = [labels[int(k)] for k in indices]
        edge_total = sum(row["edges"] for row in chosen)
        node_total = sum(row["nodes"] for row in chosen)
        leaf_total = sum(row["leaf_sign_total"] for row in chosen)
        result = [np.mean([row["pick_dist"] for row in chosen]),
                  np.mean([row["within_1e3"] for row in chosen]),
                  sum(row["edge_correct"] for row in chosen) / edge_total,
                  sum(row["node_correct"] for row in chosen) / node_total,
                  sum(row["leaf_sign_correct"] for row in chosen) / leaf_total if leaf_total else np.nan]
        for k in range(5):
            count = sum(row["beta_class_total"][k] for row in chosen)
            result.append(sum(row["beta_class_correct"][k] for row in chosen) / count if count else np.nan)
        return np.asarray(result)

    rng = np.random.default_rng(seed)
    point = values(np.arange(len(labels)))
    boot = np.asarray([values(rng.integers(len(labels), size=len(labels))) for k in range(BOOTSTRAP_SAMPLES)])
    names = ["mean_pick_dist", "within_1e3", "edge_accuracy", "node_accuracy", "leaf_sign_accuracy"]
    names += ["beta_class_%d_accuracy" % k for k in range(5)]
    result = {}
    for k, name in enumerate(names):
        finite = boot[:, k][np.isfinite(boot[:, k])]
        bounds = np.quantile(finite, [.025, .975]) if len(finite) else [None, None]
        result[name] = {"value": float(point[k]) if np.isfinite(point[k]) else None,
                        "low95": None if bounds[0] is None else float(bounds[0]),
                        "high95": None if bounds[1] is None else float(bounds[1])}
    return result


# Select only on paired validation cost, preferring simpler models within bootstrap noise.
def choose_variant(evaluations, simplicity):
    if not evaluations:
        raise ValueError("No complete validation evaluations were provided")
    manifests = [set(row["id"] for row in rows) for rows in evaluations.values()]
    if any(value != manifests[0] for value in manifests):
        raise ValueError("Variants were evaluated on different validation manifests")
    ids = sorted(manifests[0])
    logs, hit = {}, {}
    for name, rows in evaluations.items():
        summary = metric_summary(rows, ids)
        indexed = {row["id"]: row for row in rows}
        logs[name] = np.log([indexed[key]["cost_ratio"] for key in ids])
        hit[name] = summary["metrics"]["hit_rate"]["value"]
    best = min(logs, key=lambda name: (logs[name].mean(), -hit[name]))
    rng = np.random.default_rng(MODEL_SEED)
    draws = rng.integers(len(ids), size=(BOOTSTRAP_SAMPLES, len(ids)))
    paired, tied = {}, []
    for name in logs:
        difference = logs[name] - logs[best]
        bounds = np.quantile(difference[draws].mean(axis=1), [.025, .975])
        paired[name] = {"mean_log_cost_difference": float(difference.mean()),
                        "low95": float(bounds[0]), "high95": float(bounds[1])}
        if bounds[0] <= 0 <= bounds[1]:
            tied.append(name)
    chosen = min(tied, key=lambda name: (simplicity[name], logs[name].mean(), -hit[name]))
    return {"chosen": chosen, "lowest_observed_cost": best, "within_noise": tied,
            "paired_log_cost_comparisons": paired, "expected_ids": ids,
            "geometric_mean_cost_ratio": {name: float(np.exp(value.mean())) for name, value in logs.items()},
            "hit_rate": hit}


# Average model seeds within each graph before resampling graphs for uncertainty.
def aggregate_model_seeds(evaluations, expected_ids):
    fields = ["hit", "nfev", "circuit_evaluations", "expected_cost", "cost_ratio", "energy_ratio_before",
              "approximation_ratio_after"]
    indexed = [{row["id"]: row for row in records} for records in evaluations]
    if not indexed or any(set(rows) != set(expected_ids) for rows in indexed):
        raise ValueError("Seed aggregation requires every expected graph for every model seed")
    output = []
    for graph_id in expected_ids:
        rows = [records[graph_id] for records in indexed]
        row = {key: rows[0][key] for key in ["id", "n", "type", "zero_random_hit_lower_bound"]}
        row.update({key: float(np.mean([value[key] for value in rows])) for key in fields})
        if "label_metrics" in rows[0]:
            keys = rows[0]["label_metrics"]
            row["label_metrics"] = {key: np.mean([value["label_metrics"][key] for value in rows], axis=0).tolist()
                                    for key in keys}
        output.append(row)
    seed_scores = [metric_summary(records, expected_ids)["metrics"] for records in evaluations]
    spread = {name: {"mean": float(np.mean([score[name]["value"] for score in seed_scores])),
                     "std": float(np.std([score[name]["value"] for score in seed_scores], ddof=1))
                     if len(seed_scores) > 1 else 0.0,
                     "values": [score[name]["value"] for score in seed_scores]}
              for name in seed_scores[0]}
    return {"records": output, "summary": metric_summary(output, expected_ids), "seed_spread": spread}


# Enforce the complete evaluation manifest and retain each charged ranking and polish.
def evaluate_predictions(rows, predictions, baselines, run_oracle_gate=True, summarize=True, measure_labels=True):
    import modal_app
    expected = {row["id"]: row["n"] for row in rows}
    if len(expected) != len(rows) or set(predictions) != set(expected) or set(baselines) != set(expected):
        raise ValueError("Evaluation inputs do not cover the complete expected manifest")
    if run_oracle_gate:
        modal_app.oracle_gate(expected, {row["id"]: dict(row, oracle_hit=baselines[row["id"]]["oracle"]["hit"])
                                         for row in rows})
    results = {name: [] for name in ["raw", "snap", "better", "random", "zero", "oracle"]}
    for row in rows:
        graph_id = row["id"]
        for strategy in ["raw", "snap", "better"]:
            result = model_start(row, predictions[graph_id]["phase_candidates"], baselines[graph_id], strategy)
            if measure_labels:
                chosen = result["polishes"][0]["candidate_phase"]
                result["label_metrics"] = label_metrics(row, chosen)
            results[strategy].append(result)
        for method in ["random", "zero", "oracle"]:
            results[method].append(baseline_metric(row, baselines[graph_id], method))
    summaries, label_summaries = {}, {}
    if summarize:
        summaries = {name: metric_summary(records, list(expected)) for name, records in results.items()}
        if measure_labels:
            label_summaries = {name: label_summary(results[name], list(expected)) for name in ["raw", "snap", "better"]}
        for field in ["n", "type"]:
            for value in sorted({row[field] for row in rows}):
                ids = [row["id"] for row in rows if row[field] == value]
                selected = set(ids)
                summaries[field + "=" + str(value)] = {
                    name: metric_summary([row for row in records if row["id"] in selected], ids)
                    for name, records in results.items()}
    return {"expected": expected, "results": results, "summaries": summaries, "label_summaries": label_summaries}


# Export predictions in Ian's graph traversal order while preserving the sorting permutation.
def prediction_csv(path, rows, evaluated):
    import make_dataset
    fields = ["Id", "Number of Nodes", "Type", "Edges", "Gamma Edges", "Weights", "p", "Gammas", "Betas",
              "Energy", "Max Cut", "Ratio", "Energy After", "nfev", "Hit"]
    by_id = {row["id"]: row for row in evaluated}
    if len(by_id) != len(evaluated) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Prediction CSV lacks expected evaluation evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            result = by_id[row["id"]]
            phase = np.asarray(result["polishes"][0]["candidate_phase"])
            edges, order = make_dataset.ian_order(row["edges"])
            m = len(edges)
            weights = np.asarray(row["w_sorted"])
            writer.writerow({"Id": row["id"], "Number of Nodes": row["n"], "Type": row["type"],
                             "Edges": str([tuple(edge) for edge in row["edges"]]), "p": 1,
                             "Gamma Edges": json.dumps(edges), "Weights": json.dumps(weights[order].tolist()),
                             "Gammas": json.dumps((phase[:m] / weights)[order].tolist()),
                             "Betas": json.dumps(phase[m:].tolist()), "Energy": result["energy_before"],
                             "Max Cut": row["max_cut"], "Ratio": -result["energy_before"] / row["max_cut"],
                             "Energy After": result["energy_after"], "nfev": result["nfev"], "Hit": result["hit"]})


# Recreate initialization and optimizer state exactly when continuing a saved training block.
def train_block(config, rows, feature_cache, checkpoint=None, epochs=BAKEOFF_EPOCHS, device="cuda"):
    seed = int(config.get("seed", MODEL_SEED))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    example = feature_cache[rows[0]["id"]]
    model_config = {key: value for key, value in config.items()
                    if key in ["baseline", "classification", "hypotheses", "hidden", "layers", "theta_classes"]}
    model = WarmStartModel(example["node"].shape[-1], example["edge"].shape[-1], **model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    class_weights = training_class_weights(rows, model.theta_classes.cpu().numpy(), device) if model.classification else None
    history, start = [], 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        if config.get("resume_optimizer", True):
            optimizer.load_state_dict(checkpoint["optimizer"])
        history = copy.deepcopy(checkpoint["history"])
        start = int(checkpoint["epoch"]) + 1
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.startswith("cuda") and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    for epoch in range(start, start + epochs):
        history.append(train_epoch(model, optimizer, rows, feature_cache, epoch, seed,
                                    int(config.get("maximum_epochs", BAKEOFF_EPOCHS)),
                                    label_weight=config.get("label_weight", 1.0),
                                    energy_weight=config.get("energy_weight", 0.0), device=device,
                                    class_weights=class_weights))
    state = {"config": config, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "history": history, "epoch": start + epochs - 1,
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None}
    return model, state


# Keep the debugger default limited to the numerical prerequisite for energy training.
def main():
    if MODE == "check_energy":
        check_energy()
    else:
        raise ValueError("Unsupported GNN stage: %s" % MODE)


if __name__ == "__main__":
    main()
