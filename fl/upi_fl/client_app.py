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
A client keeps its shard and its running score, so a normal round only ever sends the tree
that just closed rather than the whole model. The server is authoritative about that state:
every request carries ``expected-trees`` (the number of finished trees the client must have
folded in before it answers) and one of two instructions:

    resync = 1   the whole forest ``0..expected-trees-1`` follows; reset, then apply it.
                 That is the single, explicit state reconstruction.
    resync = 0   only the tree that just closed follows (usually none); apply it and continue.

Whatever the instruction, the client then checks its own count against ``expected-trees`` and
refuses to send statistics if they differ - answering ``need-resync`` instead. The server
replies with the full forest exactly once, so a client that restarts mid-run (a SuperNode
recreates this process for every message) rejoins instead of training on a stale model, and
the two sides can never drift apart silently.
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

# One instance serves every message of a run, so the shard is read and binned once. The
# cache is module level so it also survives the instance being recreated between messages,
# and it is keyed by partition so that one process can serve several nodes - which is
# exactly what ``fl/protocol_check.py`` does to drive this file without a network.
_STATE: dict[int, dict] = {}


def _shard_path(context: Context, partition_id: int) -> Path:
    """Find this node's own shard.

    A SuperNode is usually started from the repository, but the process that runs this app
    is started by the runtime from somewhere else, so a relative ``data-path`` in
    ``--node-config`` may not be visible from here. An absolute path is always honoured
    (and always fails loudly); a relative one falls back to this node's standard shard
    name, which keeps a client from silently picking up somebody else's rows.
    """
    configured = str(context.node_config.get("data-path", "") or "")
    standard = f"data/fl/client_{partition_id}.csv"
    if configured and Path(configured).is_absolute():
        return paths.resolve_path(configured, what="data")
    try:
        return paths.resolve_path(configured or standard, what="data")
    except FileNotFoundError:
        if not configured:
            raise
        print(f"  node {partition_id}: '{configured}' not reachable from "
              f"{Path.cwd()} - trying '{standard}'", flush=True)
        return paths.resolve_path(standard, what="data")


def _state(context: Context) -> dict:
    partition_id = int(context.node_config.get("partition-id", 0))
    if partition_id in _STATE:
        return _STATE[partition_id]
    schema = bins.load_schema(
        paths.resolve_path(context.run_config.get("schema-path", ""),
                           default=Path(__file__).with_name("schema.json"),
                           what="schema"))
    path = _shard_path(context, partition_id)
    # One process can serve several nodes, so two of them could be pointed at the same file.
    # They would then share one score buffer: whichever node advanced it would drag the other
    # out of step, and the report would count one machine's rows twice. That is a data-path
    # mistake rather than a federation one, so name it instead of letting it pass as data.
    for other, cached in _STATE.items():
        if Path(cached["shard"].path).resolve() == path.resolve():
            raise RuntimeError(
                f"nodes {other} and {partition_id} were both given {path} - each node needs "
                "its own shard")
    shard = task.load_shard(path, schema)
    print(f"  node {partition_id}: {shard.n:,} rows from {shard.path}", flush=True)
    _STATE[partition_id] = dict(schema=schema, shard=shard, partition_id=partition_id,
                                num_partitions=int(context.node_config.get(
                                    "num-partitions", 1)))
    return _STATE[partition_id]


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
    return Message(content, reply_to=msg)


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
    level = int(cfg["level"])
    # The state the server says this client must be in before it may answer.
    expected = int(cfg["expected-trees"])
    base_score = float(cfg["base-score"])
    resync = int(cfg["resync"])

    # Unpack the arrays from the incoming message
    incoming = _unpack_arrays(msg.content["arrays"])

    # Apply exactly the instruction the server sent - never more. At depth 0 of a tree
    # that is either the authoritative forest (resync: reset, then apply all k finished
    # trees) or nothing; at deeper levels it is either the tree that just closed or
    # nothing. The tree being grown is in the message too, but it only serves to compute
    # the histograms and must never be folded into the score.
    if resync:
        task.reset(shard, base_score)
        for t in gbdt.unpack_trees(incoming, "forest"):
            task.apply_tree(shard, t)
    else:
        for t in gbdt.unpack_trees(incoming, "prev"):
            task.apply_tree(shard, t)

    configs, mets = _base_reply_data(shard, state)
    mets["expected-trees"] = expected
    if shard.trees_seen != expected:
        mets["need-resync"] = 1
        print(f"  node {state['partition_id']}: tree {tree_index} level {level}: "
              f"tree-count {shard.trees_seen} != expected {expected} "
              f"(resync={resync}) - asking for the authoritative forest", flush=True)
        return _reply(msg, configs=configs, metrics=mets)

    frontier = [int(x) for x in incoming["frontier"]]
    current = gbdt.unpack_trees(incoming, "tree")[0]
    buf = task.histograms(shard, current, frontier, float(cfg["pos-weight"]), schema)

    mets["payload-floats"] = int(buf.size)
    # Local loss for the model as it stands (tree_index trees in): the server averages
    # these to show progress without needing a single row of client data.
    mets["loss"] = round(task.local_loss(shard), 8)
    return _reply(msg, arrays={"hist": buf}, configs=configs, metrics=mets)
