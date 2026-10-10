"""Metriklerin tanımlara ve scikit-learn'e uygunluğu, küme bootstrap'ı, gecikme/kaynak ölçümü (sentetik küçük veri)."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support

from sdnids import metrics as M


class ConfusionMetricsTests(unittest.TestCase):
    def test_handcrafted_counts_and_rates(self):
        # TN=50, FP=10, FN=5, TP=35
        y_true = [0] * 60 + [1] * 40
        y_pred = [0] * 50 + [1] * 10 + [0] * 5 + [1] * 35
        m = M.binary_metrics(y_true, y_pred)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (50, 10, 5, 35))
        self.assertAlmostEqual(m["accuracy"], 85 / 100)
        self.assertAlmostEqual(m["attack_precision"], 35 / 45)
        self.assertAlmostEqual(m["attack_recall"], 35 / 40)
        self.assertAlmostEqual(m["fpr"], 10 / 60)
        self.assertAlmostEqual(m["fnr"], 5 / 40)
        self.assertAlmostEqual(m["fnr"], 1 - m["attack_recall"])
        self.assertAlmostEqual(m["attack_f1"], 2 * 35 / (2 * 35 + 10 + 5))

    def test_matches_sklearn_on_random_data(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2000)
        p = np.where(rng.random(2000) < 0.85, y, 1 - y)
        score = np.clip(p + rng.normal(0, 0.3, 2000), 0, 1)
        m = M.binary_metrics(y, p, score)
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
        self.assertAlmostEqual(m["attack_precision"], pr[0])
        self.assertAlmostEqual(m["attack_recall"], rc[0])
        self.assertAlmostEqual(m["attack_f1"], f1[0])
        self.assertAlmostEqual(m["macro_f1"], f1_score(y, p, average="macro"))
        self.assertAlmostEqual(m["weighted_f1"], f1_score(y, p, average="weighted"))
        self.assertAlmostEqual(m["pr_auc"], average_precision_score(y, score))

    def test_undefined_rates_are_nan_not_zero(self):
        m = M.binary_metrics([0, 0, 0], [0, 0, 0])  # hiç saldırı yok
        self.assertTrue(math.isnan(m["attack_recall"]))
        self.assertTrue(math.isnan(m["fnr"]))
        self.assertTrue(math.isnan(m["attack_precision"]))
        self.assertEqual(m["fpr"], 0.0)
        self.assertEqual(m["macro_f1"], 1.0)  # yalnız Normal sınıfı var, mükemmel

    def test_pr_auc_single_class_is_nan(self):
        self.assertTrue(math.isnan(M.pr_auc([1, 1, 1], [0.2, 0.3, 0.9])))

    def test_pr_auc_rejects_nan_scores(self):
        with self.assertRaises(ValueError):
            M.pr_auc([0, 1], [0.1, float("nan")])

    def test_invalid_labels_fail_loudly(self):
        with self.assertRaises(ValueError):
            M.counts_from_arrays([0, 2], [0, 1])
        with self.assertRaises(ValueError):
            M.counts_from_arrays([0, 1], [0])

    def test_weighted_counts_equal_duplicated_rows(self):
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0, 1, 1, 0, 1])
        w = np.array([1, 3, 2, 1, 4])
        a = M.binary_metrics(y, p, sample_weight=w)
        b = M.binary_metrics(np.repeat(y, w), np.repeat(p, w))
        for k in M.HEADLINE_METRICS:
            self.assertAlmostEqual(a[k], b[k], msg=k)

    def test_confusion_matrix_frame(self):
        cm = M.confusion_matrix_df([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(cm.to_numpy().tolist(), [[1, 1], [0, 2]])


class PerClassTests(unittest.TestCase):
    def test_rates_counts_and_reliability_flag(self):
        labels = ["Normal"] * 100 + ["DoS"] * 50 + ["U2R"] * 3
        pred = [0] * 90 + [1] * 10 + [1] * 49 + [0] + [1, 0, 0]
        t = M.per_class_detection(labels, pred, min_support=30).set_index("sınıf")
        self.assertEqual(t.loc["Normal", "n"], 100)
        self.assertAlmostEqual(t.loc["Normal", "oran"], 0.90)  # özgüllük
        self.assertAlmostEqual(t.loc["DoS", "oran"], 49 / 50)
        self.assertAlmostEqual(t.loc["U2R", "oran"], 1 / 3)
        self.assertTrue(t.loc["DoS", "güvenilir"])
        self.assertFalse(t.loc["U2R", "güvenilir"])
        self.assertEqual(list(t.index)[0], "Normal")

    def test_wilson_interval_is_sane(self):
        lo, hi = M.wilson_interval(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertTrue(math.isnan(M.wilson_interval(0, 0)[0]))
        lo0, hi0 = M.wilson_interval(0, 10)
        self.assertEqual(lo0, 0.0)
        self.assertGreater(hi0, 0.0)  # 0/10 bile üst sınırı sıfır değildir


class ClusterBootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.n = 3000
        self.y = rng.integers(0, 2, self.n)
        self.p = np.where(rng.random(self.n) < 0.9, self.y, 1 - self.y)
        self.s = np.clip(self.p + rng.normal(0, 0.3, self.n), 0, 1)

    def test_interval_contains_point_and_is_deterministic(self):
        a = M.cluster_bootstrap(self.y, self.p, self.s, n_boot=200, seed=3)
        b = M.cluster_bootstrap(self.y, self.p, self.s, n_boot=200, seed=3)
        self.assertEqual(a, b)
        for k, v in a["metrics"].items():
            self.assertLessEqual(v["low"], v["point"] + 1e-9, k)
            self.assertGreaterEqual(v["high"], v["point"] - 1e-9, k)
        self.assertEqual(a["n_clusters"], self.n)

    def test_record_level_interval_close_to_binomial_theory(self):
        r = M.cluster_bootstrap(self.y, self.p, None, n_boot=400, seed=0)["metrics"]["accuracy"]
        acc = r["point"]
        se = math.sqrt(acc * (1 - acc) / self.n)
        self.assertAlmostEqual((r["high"] - r["low"]) / 2, 1.96 * se, delta=0.35 * 1.96 * se)

    def test_correlated_clusters_widen_the_interval(self):
        # Aynı hata durumu bir kümede toplanırsa (ilişkili akışlar) belirsizlik artmalıdır.
        independent = M.cluster_bootstrap(self.y, self.p, None, n_boot=300, seed=0)["metrics"]["accuracy"]
        clusters = np.arange(self.n) // 100  # 30 büyük küme
        # her kümeyi tek tip hale getir: küme içindeki tahminler küme 'karakterini' izler
        rng = np.random.default_rng(5)
        cluster_good = rng.random(30) < 0.9
        p_corr = np.where(cluster_good[clusters], self.y, 1 - self.y)
        corr = M.cluster_bootstrap(self.y, p_corr, None, clusters, n_boot=300, seed=0)["metrics"]["accuracy"]
        self.assertGreater(corr["high"] - corr["low"], 2 * (independent["high"] - independent["low"]))

    def test_cluster_bootstrap_equals_resampling_with_duplicates(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        p = np.array([0, 1, 1, 0, 1, 0])
        clusters = np.array([0, 0, 1, 1, 2, 2])
        # tek çekiliş: kümeler [0, 0, 2] -> kayıtlar çoğaltılmış hali
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in [0, 0, 2]])
        expect = M.binary_metrics(y[idx], p[idx])
        w = np.bincount([0, 0, 2], minlength=3).astype(float)[pd.factorize(pd.Series(clusters))[0]]
        got = M.metrics_from_counts(*M.counts_from_arrays(y, p, w))
        for k in M.HEADLINE_METRICS:
            self.assertAlmostEqual(got[k], expect[k], msg=k)

    def test_cluster_ids_group_by_file_and_time_block(self):
        t0 = pd.Timestamp("2020-02-05 10:00:00")
        df = pd.DataFrame(
            {
                "source_file": ["A", "A", "A", "B"],
                "timestamp": [t0, t0 + pd.Timedelta(seconds=100), t0 + pd.Timedelta(seconds=400), t0],
            }
        )
        c = M.cluster_ids_from_frame(df, block_seconds=300)
        self.assertEqual(c[0], c[1])
        self.assertNotEqual(c[0], c[2])
        self.assertNotEqual(c[0], c[3])

    def test_missing_timestamp_gets_own_cluster(self):
        t0 = pd.Timestamp("2020-02-05 10:00:00")
        df = pd.DataFrame({"source_file": ["A", "A"], "timestamp": [t0, pd.NaT]})
        c = M.cluster_ids_from_frame(df)
        self.assertNotEqual(c[0], c[1])


class LatencyAndResourceTests(unittest.TestCase):
    def test_single_record_latency_fields(self):
        rows = [np.zeros((1, 5)) for _ in range(20)]
        r = M.measure_single_record_latency(lambda x: x.sum(), rows, n=100, warmup=10)
        self.assertEqual(r["n"], 100)
        self.assertGreater(r["p50_ms"], 0)
        self.assertGreaterEqual(r["p95_ms"], r["p50_ms"])
        self.assertGreaterEqual(r["max_ms"], r["p95_ms"])

    def test_latency_requires_rows(self):
        with self.assertRaises(ValueError):
            M.measure_single_record_latency(lambda x: x, [])

    def test_throughput_counts_all_rows(self):
        X = np.zeros((2500, 3))
        seen = []
        r = M.measure_throughput(lambda b: seen.append(len(b)), X, batch_size=1000, repeats=1)
        self.assertEqual(sum(seen), 2500)
        self.assertGreater(r["flows_per_s"], 0)

    def test_timed_resources_reports_time_and_memory(self):
        with M.timed_resources() as info:
            _ = np.ones((2000, 2000))
        self.assertGreater(info["seconds"], 0)
        self.assertIn("rss_delta_mb", info)
        self.assertGreater(info["process_peak_rss_mb"], 0)

    def test_scope_statements_exist_and_state_exclusions(self):
        for scope in M.LATENCY_SCOPE.values():
            self.assertIn("Dahil DEĞİL", scope)
        self.assertGreater(M.pickled_size_mb({"a": np.zeros(1000)}), 0)


if __name__ == "__main__":
    unittest.main()
