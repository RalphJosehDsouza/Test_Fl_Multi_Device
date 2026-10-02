"""Flower ServerApp: owns the global model and performs the aggregation.

Objective 5 is the loop in ``main``: for every boosting round the server asks the clients
for per-bin statistics, adds them, picks the split, sets the leaf value, and appends the
tree to the global model. Objective 4 is what the client *sends* - only summed statistics.

What the server keeps
---------------------
The model, the client roster and the run log. It does not need client rows, and nothing
here reads a client shard. ``server_test.csv`` is optional: when it resolves, the server
scores the global model every ``eval-every-trees`` trees so the run shows a curve rather
than a single number.

Every reply carries the client's hostname, so ``reports/fl/fl_rounds.json`` records which
machines actually contributed to each round.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MessageType,
    RecordDict,
)
from flwr.server import Grid, ServerApp

from upi_fl import bins, gbdt, metrics, paths, server_core

app = ServerApp()

W = 78
RESYNC_ATTEMPTS = 3


def rule(char: str = "-") -> None:
    print(char * W, flush=True)


def _pack_arrays(arrays_dict: dict) -> ArrayRecord:
    """Wrap numpy arrays into an ArrayRecord (each value must be an Array object)."""
    ar = ArrayRecord()
    for key, val in arrays_dict.items():
        ar[key] = Array(ndarray=np.asarray(val))
    return ar


def _unpack_arrays(ar: ArrayRecord) -> dict:
    """Read numpy arrays back from an ArrayRecord."""
    return {k: ar[k].numpy() for k in ar}


def build_message(grid: Grid, dst: int, group_id: str, config: dict,
                  arrays: dict | None = None) -> Message:
    """Create a TRAIN message using the grid (which sets run_id correctly)."""
    content = RecordDict()
    content["config"] = ConfigRecord(config)
    if arrays:
        content["arrays"] = _pack_arrays(arrays)
    return grid.create_message(
        content=content, message_type=MessageType.TRAIN,
        dst_node_id=dst, group_id=group_id)


def load_server_test(cfg_value: str, schema: dict):
    """The labelled rows the server keeps for scoring. Optional: a client-only run is still
    a valid federated run, it just cannot report test metrics mid-flight."""
    try:
        path = paths.resolve_path(cfg_value, what="server-test", must_exist=True)
    except FileNotFoundError as exc:
        print(f"  [eval] disabled: {exc}", flush=True)
        return None, None
    import pandas as pd

    df = pd.read_csv(path)
    y = df.pop("fraud_flag").to_numpy(dtype=np.float64)
    df = df.drop(columns=["timestamp"], errors="ignore")
    return bins.binned_matrix(df[schema["features"]], schema), y


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    schema = bins.load_schema(paths.resolve_path(
        str(rc.get("schema-path", "")), default=Path(__file__).with_name("schema.json"),
        what="schema"))
    params = {"max_depth": int(rc["max-depth"]),
              "learning_rate": float(rc["learning-rate"]),
              "lambda_l2": float(rc["lambda-l2"]),
              "min_child_samples": int(rc["min-child-samples"]),
              "min_split_gain": float(rc.get("min-split-gain", 1e-6))}
    num_trees = int(rc["num-trees"])
    eval_every = int(rc.get("eval-every-trees", 0))
    base_score = 0.0

    out_dir = paths.resolve_out_dir(str(rc.get("out-dir", "")))
    models_dir = out_dir / "models" / "fl"
    reports_dir = out_dir / "reports" / "fl"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    node_ids = sorted(grid.get_node_ids())
    rule("=")
    print(" FEDERATED GBDT - SERVER".center(W))
    print(f" {len(node_ids)} client node(s) in the federation".center(W))
    rule("=")
    print(f"  {num_trees} trees x depth {params['max_depth']}   learning rate "
          f"{params['learning_rate']}   lambda_l2 {params['lambda_l2']}   "
          f"{schema['total_bins']} bins")
    print(f"  node ids {node_ids}")
    print(f"  output dir {out_dir}")
    t_start = time.perf_counter()

    # --- round zero: who is here, and how imbalanced is the pooled label set -----------
    roster: dict[int, dict] = {}
    hello_msgs = [build_message(grid, n, "hello", {"op": "hello"}) for n in node_ids]
    for reply in grid.send_and_receive(hello_msgs):
        src = reply.metadata.src_node_id
        if reply.has_error() or not reply.has_content():
            print(f"  node {src}: hello failed ({reply.error})", flush=True)
            continue
        cr = reply.content["config"]     # string fields: hostname, partition-id
        mr = reply.content["metrics"]    # numeric fields: n, n-pos, n-neg
        roster[src] = {"hostname": str(cr["hostname"]),
                       "partition_id": int(cr["partition-id"]),
                       "rows": int(mr["n"]), "n_pos": int(mr["n-pos"]),
                       "n_neg": int(mr["n-neg"])}
    if not roster:
        raise RuntimeError("no client answered the hello round - is any SuperNode connected?")

    n_pos = sum(v["n_pos"] for v in roster.values())
    n_neg = sum(v["n_neg"] for v in roster.values())
    pos_weight = n_neg / max(n_pos, 1)

    rule()
    print(f"  {'node':>5}  {'host':<22}{'part':>5}{'rows':>10}{'fraud':>8}{'rate':>8}")
    rule()
    for src in sorted(roster):
        v = roster[src]
        print(f"  {src:>5}  {v['hostname']:<22}{v['partition_id']:>5}{v['rows']:>10,}"
              f"{v['n_pos']:>8,}{v['n_pos'] / max(v['rows'], 1) * 100:>7.2f}%")
    rule()
    print(f"  pooled at the clients: {sum(v['rows'] for v in roster.values()):,} rows, "
          f"{n_pos:,} fraud, pos_weight {pos_weight:.4f}")
    print("  (the server sees these totals only - no client rows, no client model)")

    # --- the aggregation loop: this is Objective 5 --------------------------------------
    forest = gbdt.Forest(schema, params, base_score)
    test_binned, test_y = (load_server_test(str(rc.get("server-test-path", "")), schema)
                           if eval_every > 0 else (None, None))
    if test_binned is not None:
        print(f"  server-side scoring on {len(test_y):,} held-out rows every "
              f"{eval_every} trees")
    else:
        print("  server-side scoring off (no server-test-path given)")

    declared: dict[int, int] = {}
    rounds = 0
    total_floats = 0
    trees_log: list[dict] = []
    evaluations: list[dict] = []

    print()
    rule()
    print(f"  {'tree':>5}{'nodes':>7}{'client loss':>14}{'test AUC':>10}{'elapsed':>10}")
    rule()

    for k in range(num_trees):
        t_tree = time.perf_counter()
        tree = gbdt.Tree(params["max_depth"])
        prev = forest.trees[-1] if forest.trees else None
        tree_losses: list[tuple[int, float]] = []
        participants: set[int] = set()

        def fetch_level(frontier, depth, k=k, tree=tree, prev=prev):
            nonlocal rounds, total_floats
            per_client: list[dict] = []
            answered: set[int] = set()       # nodes that already returned valid histograms

            for _attempt in range(RESYNC_ATTEMPTS):
                # Only (re-)query nodes that haven't provided valid data yet.
                pending = sorted(nid for nid in roster if nid not in answered)
                if not pending:
                    break

                messages = []
                for nid in pending:
                    # Depth 0: always resync — send the full forest so the client
                    # resets and applies exactly the k completed trees (0..k-1).
                    # This guarantees trees_seen == k == tree_index regardless of
                    # any prior state drift.
                    # Depth > 0: the client's state is already correct from depth 0;
                    # only force a resync if its reported tree-count drifted.
                    if depth == 0:
                        resync = 1
                    else:
                        resync = 0 if declared.get(nid, -99) == k else 1
                    arrays: dict = {}
                    arrays.update(gbdt.pack_trees([tree], "tree"))
                    # prev is no longer sent — at depth 0 the full forest already
                    # contains every completed tree, and at depth > 0 the client's
                    # running score is already up to date.
                    arrays.update(gbdt.pack_trees([], "prev"))
                    arrays["frontier"] = np.asarray(frontier, dtype=np.int32)
                    if resync:
                        arrays.update(gbdt.pack_trees(forest.trees, "forest"))
                    messages.append(build_message(
                        grid, nid, f"tree-{k}-depth-{depth}",
                        {"op": "hist", "tree-index": k, "level": depth, "resync": resync,
                         "pos-weight": float(pos_weight), "base-score": float(base_score)},
                        arrays))
                replies = list(grid.send_and_receive(messages))
                rounds += 1
                resync_asked: list[int] = []
                for reply in replies:
                    src = reply.metadata.src_node_id
                    if reply.has_error() or not reply.has_content():
                        print(f"    node {src}: {reply.error}", flush=True)
                        continue
                    mr = reply.content["metrics"]
                    declared[src] = int(mr["tree-count"])
                    if int(mr.get("need-resync", 0)):
                        resync_asked.append(src)
                        continue
                    participants.add(src)
                    answered.add(src)
                    buf = _unpack_arrays(reply.content["arrays"])["hist"]
                    total_floats += int(buf.size)
                    per_client.append(gbdt.unpack_histograms(buf, frontier,
                                                             schema["total_bins"]))
                    if depth == 0 and not tree_losses:
                        tree_losses.append((int(mr["num-examples"]),
                                            float(mr["loss"])))
                if not resync_asked:
                    break
                print(f"    node(s) {resync_asked} out of sync - resending the forest",
                      flush=True)
            if not per_client:
                raise RuntimeError(f"tree {k} level {depth}: no client returned statistics"
                                   f" after {RESYNC_ATTEMPTS} resync attempt(s)")
            return server_core.sum_histograms(per_client, frontier, schema["total_bins"])

        gains = gbdt.grow_tree(schema, params, tree, fetch_level)
        for j, gain in gains:
            forest.gain[j] += gain
        forest.add_tree(tree)

        weight = sum(n for n, _ in tree_losses)
        loss = sum(n * v for n, v in tree_losses) / weight if weight else float("nan")
        evaluated = None
        if test_binned is not None and (k + 1) % eval_every == 0:
            proba = forest.predict_proba(test_binned)
            result = metrics.evaluate(test_y, proba, 0.5)
            evaluated = {"tree": k + 1, "roc_auc": result["roc_auc"],
                         "pr_auc": result["pr_auc"], "accuracy": result["accuracy"],
                         "f1": result["f1"]}
            evaluations.append(evaluated)
            forest.save(models_dir / f"global_tree_{k + 1}.json")
            print(f"  {k + 1:>5}{tree.n_nodes:>7}{loss:>14.4f}{result['roc_auc']:>10.4f}"
                  f"{time.perf_counter() - t_start:>10.1f}", flush=True)
        elif (k + 1) % 10 == 0 or k == 0:
            print(f"  {k + 1:>5}{tree.n_nodes:>7}{loss:>14.4f}{'-':>10}"
                  f"{time.perf_counter() - t_start:>10.1f}", flush=True)

        trees_log.append({
            "tree": k + 1, "nodes": tree.n_nodes,
            "participants": sorted(participants),
            "hostnames": sorted({roster[s]["hostname"] for s in participants}),
            "client_loss": None if np.isnan(loss) else round(loss, 6),
            "evaluation": evaluated,
            "seconds": round(time.perf_counter() - t_tree, 3),
        })

    # --- results ----------------------------------------------------------------------
    seconds = time.perf_counter() - t_start
    final_model = models_dir / "global_final.json"
    forest.save(final_model)

    summary = {
        "config": {"num_trees": num_trees, "max_depth": params["max_depth"],
                   "learning_rate": params["learning_rate"], "lambda_l2": params["lambda_l2"],
                   "min_child_samples": params["min_child_samples"],
                   "total_bins": schema["total_bins"],
                   "aggregation": str(rc.get("aggregation", "sum"))},
        "pos_weight": round(pos_weight, 6),
        "nodes": {str(k2): v for k2, v in roster.items()},
        "federation_rounds": rounds,
        "client_floats_total": total_floats,
        "client_megabytes_total": round(total_floats * 8 / 1e6, 4),
        "seconds": round(seconds, 2),
        "seconds_per_tree": round(seconds / max(len(forest.trees), 1), 3),
        "evaluations": evaluations,
        "trees": trees_log,
    }
    log_path = reports_dir / "fl_rounds.json"
    log_path.write_text(json.dumps(summary, indent=2))

    rule("=")
    print(" FEDERATED RUN COMPLETE".center(W))
    rule("=")
    print(f"  {len(forest.trees)} trees, {rounds} federation rounds, {seconds:.1f}s"
          f"  ({seconds / max(len(forest.trees), 1):.2f}s per tree)")
    print(f"  client -> server traffic: {total_floats * 8 / 1e6:.3f} MB total, "
          f"{total_floats * 8 / 1e6 / max(rounds, 1):.4f} MB per round across all clients")
    hosts: set[str] = set()
    for entry in trees_log:
        hosts.update(entry["hostnames"])
    print(f"  machines that contributed: {sorted(hosts)}")
    if evaluations:
        print(f"\n  {'tree':>6}{'ROC-AUC':>10}{'PR-AUC':>10}{'accuracy':>10}{'F1':>10}")
        rule()
        for e in evaluations:
            print(f"  {e['tree']:>6}{e['roc_auc']:>10.4f}{e['pr_auc']:>10.4f}"
                  f"{e['accuracy']:>10.4f}{e['f1']:>10.4f}")
    print(f"\n  model      {final_model}")
    print(f"  run log    {log_path}")
    rule("=")
