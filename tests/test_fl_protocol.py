"""Exercise real Flower messages; transport is in memory, not a LAN test."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
from flwr.common import Context, Message, Metadata, RecordDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fl"))
from upi_fl import bins, client_app, gbdt, server_app, server_core, task


class LocalGrid:
    def __init__(self, contexts, restart=False, drop=None, duplicate=False):
        self.contexts = contexts
        self.restart = restart
        self.drop = drop
        self.duplicate = duplicate
        self.states = {}

    def get_node_ids(self):
        return self.contexts.keys()

    def create_message(self, **kwargs):
        return Message(content=kwargs["content"], metadata=Metadata(
            run_id=1, message_id="test", src_node_id=0,
            dst_node_id=kwargs["dst_node_id"], reply_to_message_id="",
            group_id=kwargs["group_id"], created_at=time.time(), ttl=60,
            message_type=kwargs["message_type"]))

    def send_and_receive(self, messages):
        replies = []
        for msg in messages:
            node = msg.metadata.dst_node_id
            if node == self.drop and msg.content["config"]["op"] == "hist":
                continue
            client_app._STATE.clear()
            if not self.restart:
                client_app._STATE.update(self.states.get(node, {}))
            reply = client_app.train(msg, self.contexts[node])
            self.states[node] = dict(client_app._STATE)
            replies.append(reply)
            if self.duplicate:
                replies.append(reply)
        return replies


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.schema = bins.build_schema(["x"], {"x": "binary"}, [2], {})
        self.schema_path = self.root / "schema.json"
        bins.save_schema(self.schema, self.schema_path)
        self.config = {"schema-path": str(self.schema_path), "num-trees": 3,
                       "max-depth": 2, "learning-rate": 0.2, "lambda-l2": 1.0,
                       "min-child-samples": 1, "eval-every-trees": 0,
                       "out-dir": str(self.root)}
        self.contexts = {}
        for node in (1, 2):
            path = self.root / f"client_{node}.csv"
            pd.DataFrame({"x": [0, 0, 1, 1] * node,
                          "fraud_flag": [0, 0, 1, 1] * node}).to_csv(path, index=False)
            self.contexts[node] = Context(1, node, {"partition-id": node,
                "data-path": str(path)}, RecordDict(), self.config)
        client_app._STATE.clear()
        self.addCleanup(client_app._STATE.clear)

    def run_server(self, **options):
        context = Context(1, 0, {}, RecordDict(), self.config)
        with contextlib.redirect_stdout(io.StringIO()):
            server_app.main(LocalGrid(self.contexts, **options), context)
        return json.loads((self.root / "models/fl/global_final.json").read_text())

    def test_restarted_clients_match_persistent_and_pooled(self):
        persistent = self.run_server()
        self.assertEqual(persistent, self.run_server(restart=True))
        self.config["forest-every-level"] = 0
        self.assertEqual(persistent, self.run_server(restart=True))
        shards = [task.load_shard(c.node_config["data-path"], self.schema,
                                 use_cache=False) for c in self.contexts.values()]
        pooled = task.Shard("pooled", np.concatenate([s.binned for s in shards]),
                            np.concatenate([s.y for s in shards]), np.zeros(12))
        params = {**gbdt.DEFAULT_PARAMS, "max_depth": 2,
                  "learning_rate": 0.2, "min_child_samples": 1}
        def fetch(k, tree, frontier, prev, depth):
            if depth == 0 and prev is not None:
                task.apply_tree(pooled, prev)
            return gbdt.unpack_histograms(task.histograms(
                pooled, tree, frontier, 1.0, self.schema), frontier, 2)
        expected = server_core.train_forest(self.schema, params, 3, fetch)
        actual = gbdt.Forest.load(self.root / "models/fl/global_final.json")
        np.testing.assert_allclose(actual.predict_proba(pooled.binned),
                                   expected.predict_proba(pooled.binned), atol=1e-12)

    def test_missing_client_aborts_without_final_model(self):
        with self.assertRaisesRegex(RuntimeError, "refusing partial aggregation"):
            self.run_server(drop=2)
        self.assertFalse((self.root / "models/fl/global_final.json").exists())

    def test_duplicate_replies_do_not_double_count(self):
        self.assertEqual(self.run_server(), self.run_server(duplicate=True))

    def test_duplicate_partitions_rejected(self):
        self.contexts[2].node_config["partition-id"] = 1
        with self.assertRaisesRegex(RuntimeError, "duplicate partition-id"):
            self.run_server()

    def test_missing_second_client_rejected_at_start(self):
        del self.contexts[2]
        with self.assertRaisesRegex(RuntimeError, "expected at least 2 clients"):
            self.run_server()

    def test_state_isolated_between_nodes_and_runs(self):
        first = client_app._state(self.contexts[1])
        first["shard"].raw[:] = 9
        self.assertEqual(client_app._state(self.contexts[2])["shard"].n, 8)
        self.contexts[1].run_id = 2
        fresh = client_app._state(self.contexts[1])
        np.testing.assert_array_equal(fresh["shard"].raw, np.zeros(4))

    def test_fingerprint_distinguishes_equal_leaf_sums(self):
        left = gbdt.Tree(1)
        a, b = left.add_split(0, 0, 0)
        left.set_leaf(a, -1)
        left.set_leaf(b, 1)
        right = gbdt.Tree.from_arrays(left.to_arrays())
        right.set_leaf(a, 1)
        right.set_leaf(b, -1)
        self.assertFalse(np.array_equal(gbdt.tree_checksum(left), gbdt.tree_checksum(right)))


if __name__ == "__main__":
    unittest.main()
