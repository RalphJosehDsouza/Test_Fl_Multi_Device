"""Check histogram training against calculations directly from individual rows."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fl"))
from upi_fl import bins, gbdt, server_core


class HistogramMathTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0, 0], [0, 1], [0, 2], [1, 0],
                           [1, 1], [1, 2], [1, 2], [0, 1]], dtype=np.int32)
        self.y = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=float)
        self.g, self.h = gbdt.gradients(np.linspace(-0.4, 0.6, 8), self.y, 1.7)
        self.schema = bins.build_schema(
            ["binary", "bucket"], {"binary": "binary", "bucket": "hour"}, [2, 3], {})
        self.params = {**gbdt.DEFAULT_PARAMS, "min_child_samples": 2,
                       "learning_rate": 0.2, "max_depth": 2}

    def histogram(self, rows):
        return gbdt.node_histogram(self.x, self.g, self.h, rows, self.schema)

    def test_each_feature_and_federated_sum_match_row_totals(self):
        pooled = self.histogram(np.arange(8))
        summed = server_core.sum_histograms(
            [{0: self.histogram(rows)} for rows in
             (np.array([0, 3]), np.array([1, 2, 4, 5, 6, 7]))], [0], 5)[0]
        np.testing.assert_allclose(summed, pooled, atol=1e-12)
        for off, size in zip(self.schema["offsets"], self.schema["bin_counts"]):
            np.testing.assert_allclose(
                [a[off:off + size].sum() for a in summed],
                [self.g.sum(), self.h.sum(), 8], atol=1e-12)

    def test_split_matches_exhaustive_row_calculation(self):
        lam = self.params["lambda_l2"]
        parent = self.g.sum() ** 2 / (self.h.sum() + lam)
        expected = None
        for j, size in enumerate(self.schema["bin_counts"]):
            for threshold in range(size - 1):
                left = self.x[:, j] <= threshold
                right = ~left
                if min(left.sum(), right.sum()) < self.params["min_child_samples"]:
                    continue
                gain = sum(self.g[mask].sum() ** 2 / (self.h[mask].sum() + lam)
                           for mask in (left, right)) - parent
                if expected is None or gain > expected[2]:
                    expected = (j, threshold, gain)
        actual = gbdt.best_split(*self.histogram(np.arange(8)), self.schema, self.params)
        self.assertEqual(actual[:2], expected[:2])
        self.assertAlmostEqual(actual[2], expected[2], places=12)

    def test_leaf_values_match_row_calculation_at_both_stop_conditions(self):
        for depth in (0, 2):
            with self.subTest(depth=depth):
                tree = gbdt.Tree(depth)
                def fetch(frontier, level):
                    nodes = tree.node_index(self.x)
                    return {nid: self.histogram(np.flatnonzero(nodes == nid))
                            for nid in frontier}
                gbdt.grow_tree(self.schema, self.params, tree, fetch)
                nodes = tree.node_index(self.x)
                for nid in np.unique(nodes):
                    rows = nodes == nid
                    expected = (-self.g[rows].sum()
                                / (self.h[rows].sum() + self.params["lambda_l2"])
                                * self.params["learning_rate"])
                    self.assertAlmostEqual(tree.value[nid], expected, places=12)

    def test_duplicate_features_cannot_create_valid_child_samples(self):
        schema = bins.build_schema(["a", "b"], {}, [2, 2], {})
        x = np.array([[0, 0], [0, 0], [0, 0], [1, 1]])
        g, h = gbdt.gradients(np.zeros(4), np.array([0, 0, 0, 1]))
        hist = gbdt.node_histogram(x, g, h, np.arange(4), schema)
        self.assertIsNone(gbdt.best_split(*hist, schema, self.params))

    def test_repeating_a_feature_does_not_change_tree_predictions(self):
        predictions = []
        for width in (1, 71):
            schema = bins.build_schema([f"x{i}" for i in range(width)], {},
                                       [2] * width, {})
            x = np.repeat(self.x[:, :1], width, axis=1)
            tree = gbdt.Tree(2)
            def fetch(frontier, level):
                nodes = tree.node_index(x)
                return {nid: gbdt.node_histogram(x, self.g, self.h,
                        np.flatnonzero(nodes == nid), schema) for nid in frontier}
            gbdt.grow_tree(schema, self.params, tree, fetch)
            predictions.append(tree.leaf_for_rows(x))
        np.testing.assert_allclose(*predictions, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
