"""The federated gradient-boosted tree itself. Pure numpy - it never imports Flower, so the
same code drives the local simulation and the real multi-device run.

How the federation works
------------------------
A tree needs two things from the data: which feature splits are good, and what value each
leaf should hold. Both are pure *sums* of per-row quantities, and sums are the one thing
several parties can compute separately and add up:

    every client computes, over its own rows,  sum(gradient) and sum(hessian) per bin
    the server adds them, picks the split, sets the leaf value = -G / (H + lambda)

So a client transmits histograms, never rows. The server transmits the tree structure and
the leaf values. ``aggregation = "sum"`` in pyproject.toml is exactly this.

Why the split search is exact rather than approximate
-----------------------------------------------------
Both sides bin each feature into a fixed number of buckets up front (``bins.py``), so a
split is "bin <= t" and a histogram is a per-bin total. Summing per-bin totals across
clients gives the same histogram as summing the rows first, which is why the federated
model and a pooled one agree to floating-point error. ``fl/simulate.py`` asserts this.

Relation to the Phase I baseline
--------------------------------
Same family and the same hyperparameters as ``src/modeling.py`` (shallow trees, lambda_l2,
shrinkage by learning_rate, and scale_pos_weight-style imbalance weighting). The difference
is that growth is level-wise here instead of LightGBM's leaf-wise default: level-wise gives
exactly one server/client exchange per depth level, which is what makes the aggregation
tractable. Tree count and depth are config, not assumptions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

# Defaults mirror src/modeling.py PARAMS where the meaning carries over.
DEFAULT_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "lambda_l2": 1.0,
    "min_child_samples": 50,
    "min_split_gain": 1e-6,
}

EPS = 1e-12


# ------------------------------------------------------------------------------------- tree

class Tree:
    """A level-wise tree. Nodes are created root-first, so a child always has a larger id
    than its parent - which is what lets ``node_index`` settle every row in one pass."""

    def __init__(self, max_depth: int):
        self.max_depth = int(max_depth)
        self.feature = [-1]
        self.threshold = [-1]
        self.left = [-1]
        self.right = [-1]
        self.value = [float("nan")]
        self.depth = [0]

    # -- construction ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.feature)

    def is_split(self, nid: int) -> bool:
        return self.feature[nid] >= 0

    def is_leaf(self, nid: int) -> bool:
        return self.feature[nid] < 0 and not np.isnan(self.value[nid])

    def frontier(self) -> list[int]:
        return [n for n in range(self.n_nodes)
                if self.feature[n] < 0 and np.isnan(self.value[n])]

    def nodes_at_depth(self, d: int) -> list[int]:
        return [n for n in self.frontier() if self.depth[n] == d]

    def add_split(self, node: int, feature: int, threshold: int) -> tuple[int, int]:
        self.feature[node] = int(feature)
        self.threshold[node] = int(threshold)
        child_depth = self.depth[node] + 1
        for side in ("left", "right"):
            getattr(self, side)[node] = self.n_nodes
            self.feature.append(-1)
            self.threshold.append(-1)
            self.left.append(-1)
            self.right.append(-1)
            self.value.append(float("nan"))
            self.depth.append(child_depth)
        return self.left[node], self.right[node]

    def set_leaf(self, node: int, value: float) -> None:
        self.value[node] = float(value)

    # -- scoring -----------------------------------------------------------------------
    def node_index(self, binned: np.ndarray) -> np.ndarray:
        """Which node each row lands in. Both sides run this on the same binned rows and
        the same structure, so the partitions agree exactly."""
        node = np.zeros(len(binned), dtype=np.int32)
        feat = np.asarray(self.feature, dtype=np.int32)
        for nid in range(self.n_nodes):
            if feat[nid] < 0:
                continue
            rows = np.flatnonzero(node == nid)
            if rows.size == 0:
                continue
            go_left = binned[rows, feat[nid]] <= self.threshold[nid]
            node[rows[~go_left]] = self.right[nid]
            node[rows[go_left]] = self.left[nid]
        return node

    def leaf_isnan(self) -> bool:
        return any(np.isnan(v) for v in self.value)

    # -- (de)serialisation --------------------------------------------------------------
    def to_arrays(self) -> dict:
        return {k: np.asarray(getattr(self, k)) for k in
                ("feature", "threshold", "left", "right", "value", "depth")}

    @classmethod
    def from_arrays(cls, d: dict) -> "Tree":
        t = cls(max_depth=int(max(d["depth"])))
        t.feature = [int(x) for x in d["feature"]]
        t.threshold = [int(x) for x in d["threshold"]]
        t.left = [int(x) for x in d["left"]]
        t.right = [int(x) for x in d["right"]]
        t.value = [float(x) for x in d["value"]]
        t.depth = [int(x) for x in d["depth"]]
        return t

    def to_json(self) -> dict:
        return {k: list(getattr(self, k)) for k in
                ("feature", "threshold", "left", "right", "value", "depth")}

    def leaf_for_rows(self, binned: np.ndarray) -> np.ndarray:
        return np.asarray(self.value, dtype=np.float64)[self.node_index(binned)]


# ------------------------------------------------------------------------ tree transfer

TREE_FIELDS = ("feature", "threshold", "left", "right", "value", "depth")
_INT_FIELDS = ("feature", "threshold", "left", "right", "depth")


def pack_trees(trees, key: str = "tree") -> dict:
    """Pack a list of trees into flat arrays, one per field, plus a ``<key>_starts`` index.

    A round usually ships one or two trees, but the same packing carries the whole forest
    when a client has to be resynchronised, so there is only one code path to trust.
    """
    trees = list(trees)
    starts = np.zeros(1, dtype=np.int32) if not trees else np.cumsum(
        [0] + [t.n_nodes for t in trees]).astype(np.int32)
    out = {f"{key}_starts": starts}
    for name in TREE_FIELDS:
        if not trees:
            out[f"{key}_{name}"] = np.zeros(0, dtype=np.float64 if name == "value" else np.int32)
            continue
        dtype = np.float64 if name == "value" else np.int32
        out[f"{key}_{name}"] = np.concatenate(
            [np.asarray(getattr(t, name)) for t in trees]).astype(dtype)
    return out


def tree_checksum(tree) -> np.ndarray:
    """Fingerprint structure and leaf values, including trees with equal leaf sums."""
    digest = hashlib.sha256()
    for name in TREE_FIELDS:
        dtype = "<f8" if name == "value" else "<i4"
        digest.update(np.asarray(getattr(tree, name), dtype=dtype).tobytes())
    return np.frombuffer(digest.digest(), dtype=np.uint8).copy()


def unpack_trees(record, key: str) -> list[Tree]:
    """Rebuild the trees that ``pack_trees`` wrote into an ArrayRecord under ``key``."""
    starts = np.asarray(record[f"{key}_starts"])
    trees = []
    for i in range(len(starts) - 1):
        a, b = int(starts[i]), int(starts[i + 1])
        blob = {name: np.asarray(record[f"{key}_{name}"])[a:b] for name in TREE_FIELDS}
        trees.append(Tree.from_arrays(blob))
    return trees


# ----------------------------------------------------------------------------------- forest

class Forest:
    """The global model: a base score plus the trees built so far."""

    def __init__(self, schema: dict, params: dict | None = None, base_score: float = 0.0):
        self.schema = schema
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.base_score = float(base_score)
        self.trees: list[Tree] = []
        self.gain = np.zeros(schema["n_features"], dtype=np.float64)

    def add_tree(self, tree: Tree) -> None:
        self.trees.append(tree)

    def raw_score(self, binned: np.ndarray) -> np.ndarray:
        raw = np.full(len(binned), self.base_score, dtype=np.float64)
        for tree in self.trees:
            raw += tree.leaf_for_rows(binned)
        return raw

    def predict_proba(self, binned: np.ndarray) -> np.ndarray:
        return sigmoid(self.raw_score(binned))

    def feature_importance(self, top: int | None = None) -> list[tuple[str, float]]:
        order = np.argsort(-self.gain)
        rows = [(self.schema["features"][i], float(self.gain[i]))
                for i in order if self.gain[i] > 0]
        return rows if top is None else rows[:top]

    # -- persistence -------------------------------------------------------------------
    def to_json(self) -> dict:
        return {"params": self.params, "base_score": self.base_score,
                "schema": self.schema, "gain": self.gain.tolist(),
                "trees": [t.to_json() for t in self.trees]}

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path) -> "Forest":
        blob = json.loads(path.read_text())
        forest = cls(blob["schema"], blob["params"], blob["base_score"])
        forest.trees = [Tree.from_arrays(t) for t in blob["trees"]]
        forest.gain = np.asarray(blob.get("gain", []), dtype=np.float64)
        return forest


# -------------------------------------------------------------------------------- boosting

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def gradients(raw: np.ndarray, y: np.ndarray, pos_weight: float = 1.0):
    """Log-loss gradient and hessian, with the positive class up-weighted by pos_weight -
    the federated counterpart of the baseline's ``scale_pos_weight``."""
    p = sigmoid(raw)
    w = np.where(y > 0, float(pos_weight), 1.0)
    return w * (p - y), w * p * (1.0 - p)


def feature_groups(schema: dict):
    """Split the features into the binary block and the rest.

    66 of the 71 columns are one-hot or 0/1 flags. For those, the histogram on bin 1 is just
    a column sum, so the whole block is one matrix product instead of 66 separate passes -
    worth roughly an order of magnitude on a run of this size. amount and hour keep their
    own bincount.
    """
    binary = [j for j, n in enumerate(schema["bin_counts"]) if n == 2]
    multi = [j for j, n in enumerate(schema["bin_counts"]) if n != 2]
    return binary, multi


def node_histogram(binned: np.ndarray, g: np.ndarray, h: np.ndarray, rows: np.ndarray,
                   schema: dict, groups=None):
    """Per-bin totals of gradient, hessian and row count over the given rows."""
    total_bins = schema["total_bins"]
    G = np.zeros(total_bins, dtype=np.float64)
    H = np.zeros(total_bins, dtype=np.float64)
    C = np.zeros(total_bins, dtype=np.float64)
    gr = g[rows]
    hr = h[rows]
    groups = feature_groups(schema) if groups is None else groups
    binary, multi = groups

    if binary:
        block = binned[rows][:, binary].astype(np.float64)
        offset = np.asarray(schema["offsets"], dtype=np.int64)[binary]
        sums_g = block.T @ gr
        sums_h = block.T @ hr
        sums_c = block.sum(axis=0)
        tot_g, tot_h, tot_c = float(gr.sum()), float(hr.sum()), float(len(rows))
        G[offset] = tot_g - sums_g
        H[offset] = tot_h - sums_h
        C[offset] = tot_c - sums_c
        G[offset + 1] = sums_g
        H[offset + 1] = sums_h
        C[offset + 1] = sums_c

    for j in multi:
        off, nb = schema["offsets"][j], schema["bin_counts"][j]
        col = binned[rows, j]
        G[off:off + nb] = np.bincount(col, weights=gr, minlength=nb)
        H[off:off + nb] = np.bincount(col, weights=hr, minlength=nb)
        C[off:off + nb] = np.bincount(col, minlength=nb)
    return G, H, C


def pack_histograms(stats: list) -> np.ndarray:
    """One flat float64 vector - the only thing a client ever transmits."""
    return np.concatenate([np.concatenate([G, H, C]) for G, H, C in stats])


def unpack_histograms(buf, node_ids, total_bins: int) -> dict:
    per = 3 * total_bins
    out = {}
    for i, nid in enumerate(node_ids):
        chunk = buf[i * per:(i + 1) * per]
        out[nid] = (chunk[:total_bins], chunk[total_bins:2 * total_bins],
                    chunk[2 * total_bins:])
    return out


def leaf_value(G: float, H: float, params: dict) -> float:
    """Newton step, shrunk by the learning rate: the federated stand-in for a model weight."""
    return -float(G) / (float(H) + params["lambda_l2"]) * params["learning_rate"]


def node_totals(G: np.ndarray, H: np.ndarray, C: np.ndarray, schema: dict):
    """Each feature covers all rows; use one feature to count each row once."""
    off, size = schema["offsets"][0], schema["bin_counts"][0]
    return tuple(float(a[off:off + size].sum()) for a in (G, H, C))


def best_split(G: np.ndarray, H: np.ndarray, C: np.ndarray, schema: dict, params: dict):
    """Best (feature, bin threshold) for one node, or None if nothing beats the gain floor.

    The left child takes bins <= t. Every term is a sum over the aggregated histogram, so
    the choice is identical to what a pooled histogram would produce.
    """
    lam = params["lambda_l2"]
    min_child = params["min_child_samples"]
    Gt, Ht, Ct = node_totals(G, H, C, schema)
    if Ct < 2 * min_child or Ht <= EPS:
        return None
    parent = Gt * Gt / (Ht + lam)

    best = None
    for j, (off, nb) in enumerate(zip(schema["offsets"], schema["bin_counts"])):
        if nb < 2:
            continue
        gl, hl, cl = G[off:off + nb], H[off:off + nb], C[off:off + nb]
        GL, HL, CL = np.cumsum(gl)[:-1], np.cumsum(hl)[:-1], np.cumsum(cl)[:-1]
        GR, HR, CR = Gt - GL, Ht - HL, Ct - CL
        gain = GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent
        ok = (CL >= min_child) & (CR >= min_child) & (HL > EPS) & (HR > EPS)
        if not ok.any():
            continue
        gain = np.where(ok, gain, -np.inf)
        t = int(np.argmax(gain))
        if best is None or gain[t] > best[2]:
            best = (j, t, float(gain[t]))

    if best is None or best[2] <= params["min_split_gain"]:
        return None
    return best


def grow_tree(schema: dict, params: dict, tree: Tree, fetch) -> list[tuple[int, float]]:
    """Grow one tree level by level.

    ``fetch(node_ids, depth) -> {node_id: (G, H, C)}`` is a single server<->client round: the
    server names the nodes it needs statistics for, every client answers with its own
    per-bin totals, and they are summed. ``depth`` lets the round know whether the clients
    should be folding in the previous tree yet. That callback is the only thing that differs
    between the local simulation and the real Flower run.
    """
    gains: list[tuple[int, float]] = []
    frontier = [0]
    for depth in range(tree.max_depth + 1):
        if not frontier:
            break
        stats = fetch(frontier, depth)
        next_frontier: list[int] = []
        for nid in frontier:
            G, H, C = stats[nid]
            Gt, Ht, _ = node_totals(G, H, C, schema)
            if depth == tree.max_depth:
                tree.set_leaf(nid, leaf_value(Gt, Ht, params))
                continue
            best = best_split(G, H, C, schema, params)
            if best is None:
                tree.set_leaf(nid, leaf_value(Gt, Ht, params))
                continue
            j, t, gain = best
            gains.append((j, gain))
            left, right = tree.add_split(nid, j, t)
            next_frontier.extend([left, right])
        frontier = next_frontier
    return gains
