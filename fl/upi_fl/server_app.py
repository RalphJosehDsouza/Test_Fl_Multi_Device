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

Staying in step
---------------
Growing one tree is a sequence of depth-by-depth rounds, and the clients carry the model
between rounds (see ``client_app`` for why). The server is authoritative about that state:
each round states the number of finished trees every client must hold (``expected-trees``)
and either hands over the single tree that just closed or the whole forest. A reply is only
usable when the client confirms that same count; otherwise the node is reported as
``NO_REPLY`` (transport) or ``REPLY_REJECTED_DUE_TO_SYNC`` (it answered and refused), the
round is retried with the full forest, and a client that still disagrees fails the run loudly
with those numbers rather than silently averaging over fewer machines.
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
# How many exchanges one round may take. Each attempt is strictly more authoritative than
# the last (a plain resend, then a full rebuild), so the budget is exactly what the three
# possible cases need: a normal round, one retry after a client goes silent, and one rebuild
# for a client that admits it is out of step. Nothing here can repeat itself, so the run can
# never spend its time re-sending a payload that was already refused.
SYNC_ATTEMPTS = 3


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
    if len(roster) < len(node_ids):
        print(f"  WARNING: {len(node_ids) - len(roster)} of {len(node_ids)} connected node(s) "
              f"failed the hello round and will not contribute", flush=True)
        print("           check each SuperNode's --node-config data-path: it must point at "
              "that machine's own shard", flush=True)

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
    # Rounds that needed more than one exchange because a node came back short. On a real
    # federation this is where a dropped reply or a client that starts from scratch on every
    # message shows up, so the run reports the count instead of hiding it.
    recoveries = 0
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
        tree_losses: list[tuple[int, float]] = []
        participants: set[int] = set()

        def fetch_level(frontier, depth, k=k, tree=tree):
            """One server<->client round, at one depth of one tree.

            The server is authoritative: before a client may answer it must be holding
            exactly ``k`` finished trees, and the server says so in the request. The first
            exchange picks the cheapest instruction that gets a healthy client there,
            derived from the tree-count it last reported (``declared``):

                in step (k)          nothing to apply - just answer
                one behind (k - 1)   hand over the single tree that just closed
                anything else        rebuild: the whole forest 0..k-1

            A round that comes back short is retried with the one instruction that cannot
            be misapplied - the full rebuild - whatever the reason was. A client that went
            silent, restarted mid-run or drifted therefore always lands on the same state,
            and no attempt ever repeats a payload a client has already refused. The tree
            being grown travels under a separate key and is never folded in as state,
            which is what used to leave the two sides exactly one tree apart.
            """
            nonlocal rounds, total_floats, recoveries
            per_client: list[dict] = []
            answered: set[int] = set()       # nodes that already returned valid histograms
            status: dict[int, str] = {}      # why each node did (or did not) contribute
            forest_trees = forest.trees      # the authoritative state: trees 0..k-1

            for attempt in range(SYNC_ATTEMPTS):
                # Only (re-)query nodes that haven't provided valid data yet.
                pending = sorted(nid for nid in roster if nid not in answered)
                if not pending:
                    break

                messages = []
                for nid in pending:
                    last = declared.get(nid)          # tree-count this node last reported
                    if attempt or last is None or last not in (k, k - 1) \
                            or (last == k - 1 and k == 0):
                        # Never heard from it, it drifted, or the cheap send already
                        # failed: rebuild rather than guess.
                        resync, prev_trees = 1, []
                    elif last == k:
                        resync, prev_trees = 0, []                      # already in step
                    else:
                        resync, prev_trees = 0, [forest_trees[k - 1]]   # one tree behind
                    arrays: dict = {}
                    arrays.update(gbdt.pack_trees([tree], "tree"))
                    arrays.update(gbdt.pack_trees(prev_trees, "prev"))
                    arrays.update(gbdt.pack_trees(forest_trees if resync else [], "forest"))
                    arrays["frontier"] = np.asarray(frontier, dtype=np.int32)
                    messages.append(build_message(
                        grid, nid, f"tree-{k}-depth-{depth}",
                        {"op": "hist", "tree-index": k, "level": depth, "resync": resync,
                         "expected-trees": k, "pos-weight": float(pos_weight),
                         "base-score": float(base_score)},
                        arrays))
                replies = list(grid.send_and_receive(messages))
                rounds += 1
                for reply in replies:
                    src = reply.metadata.src_node_id
                    if reply.has_error() or not reply.has_content():
                        # NO_REPLY: nothing came back, or it came back broken.
                        status[src] = f"NO_REPLY ({reply.error})"
                        continue
                    mr = reply.content["metrics"]
                    declared[src] = int(mr["tree-count"])
                    if int(mr.get("need-resync", 0)):
                        # REPLY_REJECTED_DUE_TO_SYNC: the client is alive, it answered,
                        # and it refused to train on the state it was handed.
                        got, want = int(mr["tree-count"]), int(mr.get("expected-trees", k))
                        status[src] = (f"REPLY_REJECTED_DUE_TO_SYNC (client tree-count "
                                       f"{got}, server expected {want})")
                        continue
                    # VALID_HISTOGRAM: real statistics from a client that is in step.
                    status[src] = "VALID_HISTOGRAM"
                    participants.add(src)
                    answered.add(src)
                    buf = _unpack_arrays(reply.content["arrays"])["hist"]
                    total_floats += int(buf.size)
                    per_client.append(gbdt.unpack_histograms(buf, frontier,
                                                             schema["total_bins"]))
                    if depth == 0 and not tree_losses:
                        tree_losses.append((int(mr["num-examples"]),
                                            float(mr["loss"])))
                short = [nid for nid in pending if nid not in answered]
                if not short:
                    break
                detail = "\n".join(f"      node {nid}: {status.get(nid, 'NO_REPLY')}"
                                   for nid in short)
                if attempt + 1 >= SYNC_ATTEMPTS:
                    # A rebuild went out and still disagreed: report the real numbers
                    # instead of pretending the client sent nothing.
                    raise RuntimeError(
                        f"tree {k} level {depth}: client state still disagrees after a "
                        f"full rebuild - stopping rather than looping.\n"
                        f"    server state: tree {k}, depth {depth}, every client must "
                        f"hold {k} tree(s)\n{detail}")
                print(f"    tree {k} level {depth}: node(s) {short} - sending the "
                      f"authoritative forest (attempt {attempt + 2} of {SYNC_ATTEMPTS})",
                      flush=True)
                # Say *why* each node came back short. NO_REPLY means nothing arrived (a
                # dropped or unreadable reply); a rejected tree-count means the node answered
                # and disagreed about the state it held. Those need different fixes, so the
                # reason is printed rather than left to be guessed from the round count.
                print(detail, flush=True)
                recoveries += 1

            if not per_client:
                detail = "\n".join(f"      node {nid}: {status.get(nid, 'NO_REPLY')}"
                                   for nid in sorted(roster))
                raise RuntimeError(
                    f"tree {k} level {depth}: no usable client statistics.\n"
                    f"    server state: tree {k}, depth {depth}, every client must "
                    f"hold {k} tree(s)\n{detail}")
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
        "client_recoveries": recoveries,
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
    if recoveries:
        print(f"  {recoveries} round(s) needed a second exchange - a node came back short "
              f"and was rebuilt from the authoritative forest (see the reasons above)")
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
