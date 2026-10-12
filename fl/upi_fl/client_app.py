"""Flower ClientApp: one simulated bank/POS client.

The client holds its own transactions and nothing else. Every value it sends out is a sum,
so the server learns aggregated statistics and never a row, a model, or a single
transaction.

Protocol (the server drives; see ``server_app`` for the other half)

    op = hello   report row count, fraud count and hostname. The server needs the global
                 positive rate to weight the two classes the way ScalePosWeight does.
    op = hist    the server names the nodes it needs statistics for; reply with the per-bin
                 gradient/hessian/count totals for each of them.

State across messages
---------------------
A process may be restarted between any two messages. The server supplies completed trees
at every level by default; the client rebuilds its score or applies only new trees when
its cached fingerprints agree. A cache is scoped to one run and one node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.client import ClientApp

from upi_fl import bins, gbdt, paths, task

app = ClientApp()

# Optional process-local cache; correctness must not depend on its survival.
_STATE: dict = {}


def _state(context: Context) -> dict:
    identity = (context.run_id, context.node_id, dict(context.node_config),
                dict(context.run_config))
    if _STATE.get("identity") == identity:
        return _STATE
    schema = bins.load_schema(
        paths.resolve_path(context.run_config.get("schema-path", ""),
                           default=Path(__file__).with_name("schema.json"),
                           what="schema"))
    partition_id = int(context.node_config.get("partition-id", 0))
    data_path = context.node_config.get("data-path", "") or f"data/fl/client_{partition_id}.csv"
    shard = task.load_shard(paths.resolve_path(data_path, what="data"), schema,
                            use_cache=False)
    _STATE.clear()
    _STATE.update(identity=identity, schema=schema, shard=shard, partition_id=partition_id,
                  num_partitions=int(context.node_config.get("num-partitions", 1)))
    return _STATE


def _unpack_arrays(ar: ArrayRecord) -> dict:
    """Read numpy arrays back from an ArrayRecord."""
    return {k: ar[k].numpy() for k in ar}


def _reply(msg: Message, arrays: dict | None = None,
           configs: dict | None = None, metrics: dict | None = None) -> Message:
    """Build a reply with optional arrays (numpy), config (str) and metrics (numeric)."""
    content = RecordDict()
    if arrays:
        ar = ArrayRecord()
        for key, val in arrays.items():
            ar[key] = Array(ndarray=np.asarray(val))
        content["arrays"] = ar
    if configs:
        content["config"] = ConfigRecord(configs)
    if metrics:
        content["metrics"] = MetricRecord(metrics)
    return msg.create_reply(content)


def _base_reply_data(shard: task.Shard, state: dict) -> tuple[dict, dict]:
    """Return (config_dict, metric_dict) with base client info.

    String fields (hostname, partition-id) go in ConfigRecord;
    numeric fields go in MetricRecord.
    """
    configs = {
        "hostname": shard.hostname,
        "partition-id": state["partition_id"],
    }
    mets = {
        "num-examples": shard.n,
        "n": shard.n,
        "n-pos": shard.n_pos,
        "n-neg": shard.n_neg,
        "tree-count": shard.trees_seen,
    }
    return configs, mets


@app.train()
def train(msg: Message, context: Context) -> Message:
    state = _state(context)
    shard, schema = state["shard"], state["schema"]
    cfg = msg.content["config"]

    if cfg["op"] == "hello":
        configs, mets = _base_reply_data(shard, state)
        return _reply(msg, configs=configs, metrics=mets)

    # --- one round of growing the current tree -----------------------------------------
    tree_index = int(cfg["tree-index"])
    base_score = float(cfg["base-score"])
    incoming = _unpack_arrays(msg.content["arrays"])

    # The forest is authoritative. Reuse scores only when both the base score and
    # completed tree fingerprints match; a restarted process rebuilds them.
    if "forest_starts" in incoming:
        forest = gbdt.unpack_trees(incoming, "forest")
        if len(forest) != tree_index:
            raise ValueError("forest length does not match tree-index")
        checksums = np.asarray(incoming["forest_tree_ck"], dtype=np.uint8)
        mine = state.get("forest_ck", np.zeros((0, 32), dtype=np.uint8))
        already = shard.trees_seen
        consistent = (state.get("base_score") == base_score
                      and already <= len(forest) and len(mine) >= already
                      and bool(np.array_equal(mine[:already], checksums[:already])))
        if not consistent:
            task.reset(shard, base_score)
            already = 0
        for t in forest[already:]:
            task.apply_tree(shard, t)
        state["forest_ck"] = checksums
        state["base_score"] = base_score

    configs, mets = _base_reply_data(shard, state)
    if shard.trees_seen != tree_index:
        # No forest in this message and the counts disagree, so the client cannot repair
        # itself. Say so instead of answering with statistics from the wrong gradients.
        mets["need-resync"] = 1
        return _reply(msg, configs=configs, metrics=mets)

    frontier = [int(x) for x in incoming["frontier"]]
    current = gbdt.unpack_trees(incoming, "tree")[0]
    buf = task.histograms(shard, current, frontier, float(cfg["pos-weight"]), schema)

    mets["payload-floats"] = int(buf.size)
    # Local loss for the model as it stands (tree_index trees in): the server averages
    # these to show progress without needing a single row of client data.
    mets["loss"] = round(task.local_loss(shard), 8)
    return _reply(msg, arrays={"hist": buf}, configs=configs, metrics=mets)
