"""Model modülü testleri: özellik kümeleri, yalnızca eğitimde öğrenme, eşik kuralları, sütun uyumu.

Veri SENTETİKTİR ve yalnızca altyapıyı sınar.
"""

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sdnids import modeling as MD
from sdnids.schema import build_schema_from_frame


def make_frame(n=1200, seed=0, attack_share=0.4):
    """20 LLM özelliği + 3 ek özellik; saldırılarda paket hızı ve SYN sayısı yüksek (öğrenilebilir sinyal)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < attack_share).astype(int)
    cols = {}
    for name in MD.LLM_SUBSET_FEATURES + ("extra_a", "extra_b", "extra_c"):
        cols[name] = rng.lognormal(mean=3, sigma=1.5, size=n)
    cols["flow_pkts_s"] = cols["flow_pkts_s"] * np.where(y == 1, 30.0, 1.0)
    cols["syn_flag_cnt"] = rng.poisson(np.where(y == 1, 6, 1)).astype(float)
    cols["protocol"] = rng.choice([6.0, 17.0], size=n)
    df = pd.DataFrame(cols)
    df["label"] = np.where(y == 1, "DoS", "Normal")
    df["record_id"] = [f"r{i}" for i in range(n)]
    return df, y


class FeatureSetTests(unittest.TestCase):
    def setUp(self):
        self.df, _ = make_frame()
        self.schema, _ = build_schema_from_frame(self.df)

    def test_full_is_schema_order(self):
        self.assertEqual(MD.resolve_feature_set("full", self.schema), self.schema.names)

    def test_llm_subset_has_20_features_in_schema_order(self):
        feats = MD.resolve_feature_set("llm_subset", self.schema)
        self.assertEqual(sorted(feats), sorted(MD.LLM_SUBSET_FEATURES))
        positions = [self.schema.names.index(f) for f in feats]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("extra_a", feats)

    def test_missing_llm_feature_fails_loudly(self):
        schema, _ = build_schema_from_frame(self.df.drop(columns=["syn_flag_cnt"]))
        with self.assertRaises(ValueError) as ctx:
            MD.resolve_feature_set("llm_subset", schema)
        self.assertIn("syn_flag_cnt", str(ctx.exception))

    def test_unknown_feature_set(self):
        with self.assertRaises(ValueError):
            MD.resolve_feature_set("hepsi", self.schema)

    def test_no_identity_or_label_column_in_llm_subset(self):
        for bad in ("label", "src_ip", "dst_ip", "timestamp", "flow_id", "src_port", "dst_port", "source_file"):
            self.assertNotIn(bad, MD.LLM_SUBSET_FEATURES)


class RegistryTests(unittest.TestCase):
    def test_every_policy_is_documented_and_declared(self):
        for spec in MD.MODEL_REGISTRY.values():
            self.assertGreaterEqual(len(spec.threshold_policies), 1)
            for p in spec.threshold_policies:
                self.assertIn(p, MD.THRESHOLD_POLICY_DOCS, f"{spec.name}:{p}")
            self.assertGreaterEqual(len(spec.grid), 1)

    def test_required_models_exist(self):
        self.assertEqual(set(MD.MODEL_NAMES), {"dummy", "logreg", "rf", "xgboost", "iforest"})
        self.assertFalse(MD.get_model_spec("iforest").supervised)
        with self.assertRaises(ValueError):
            MD.get_model_spec("svm")

    def test_missing_xgboost_is_reported_not_hidden(self):
        spec = MD.get_model_spec("xgboost")
        ok, reason = spec.availability()
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.assertFalse(ok)
            self.assertIn("xgboost", reason)
            with self.assertRaises(MD.ModelUnavailableError):
                MD.build_pipeline(spec, spec.grid[0], 0)
        else:  # pragma: no cover - kurulu ortamda
            self.assertTrue(ok)

    def test_pipeline_steps_match_model_needs(self):
        lr = MD.build_pipeline(MD.get_model_spec("logreg"), {"C": 1.0, "class_weight": None}, 0)
        self.assertIn("log1p", lr.named_steps["prep"].named_steps)
        self.assertIn("scale", lr.named_steps["prep"].named_steps)
        rf = MD.build_pipeline(MD.get_model_spec("rf"), {"max_depth": 3, "min_samples_leaf": 1}, 0)
        self.assertNotIn("scale", rf.named_steps["prep"].named_steps)

    def test_selection_functions_never_take_test_data(self):
        for fn in (MD.tune_hyperparameters, MD.fit_model, MD.select_threshold_f1):
            params = " ".join(inspect.signature(fn).parameters)
            self.assertNotIn("test", params, fn.__name__)


class ThresholdTests(unittest.TestCase):
    def test_f1_threshold_separates_perfectly_separable_scores(self):
        y = np.array([0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        t = MD.select_threshold_f1(y, s)
        self.assertTrue(((s >= t).astype(int) == y).all())

    def test_threshold_needs_both_classes(self):
        with self.assertRaises(ValueError):
            MD.select_threshold_f1([1, 1, 1], [0.2, 0.5, 0.9])


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, y = make_frame(n=1500, seed=1)
        schema, _ = build_schema_from_frame(df)
        cls.schema = schema
        feats = MD.resolve_feature_set("llm_subset", schema)
        X = schema.align(df)[feats]
        X.iloc[::17, 3] = np.nan  # eksik değerler: medyanla doldurulmalı
        X.iloc[::23, 5] = np.inf  # sonsuz değer: eksik sayılmalı
        cls.Xtr, cls.ytr = X.iloc[:900], y[:900]
        cls.Xva, cls.yva = X.iloc[900:1200], y[900:1200]
        cls.Xte, cls.yte = X.iloc[1200:], y[1200:]

    def _fit(self, name, params=None, seed=0):
        spec = MD.get_model_spec(name)
        return MD.fit_model(
            spec, params if params is not None else spec.grid[0], seed,
            self.Xtr, self.ytr, self.Xva, self.yva,
            feature_set="llm_subset", schema_digest=self.schema.digest(), model_cfg={"n_estimators": 30},
        )

    def test_imputer_learns_only_from_training_rows(self):
        m = self._fit("logreg", {"C": 1.0, "class_weight": None})
        stats = m.pipeline.named_steps["prep"].named_steps["impute"].statistics_
        train = self.Xtr.replace([np.inf, -np.inf], np.nan)
        np.testing.assert_allclose(stats, train.median().to_numpy())
        allrows = pd.concat([self.Xtr, self.Xva]).replace([np.inf, -np.inf], np.nan)
        self.assertFalse(np.allclose(stats, allrows.median().to_numpy()))

    def test_isolation_forest_sees_only_normal_training_rows(self):
        m = self._fit("iforest", {"max_samples": 128})
        stats = m.pipeline.named_steps["prep"].named_steps["impute"].statistics_
        normal = self.Xtr.loc[self.ytr == 0].replace([np.inf, -np.inf], np.nan)
        np.testing.assert_allclose(stats, normal.median().to_numpy())
        self.assertEqual(m.pipeline.named_steps["clf"].n_features_in_, self.Xtr.shape[1])

    def test_unlabelled_threshold_flags_about_one_percent_of_training_normal(self):
        m = self._fit("iforest", {"max_samples": 128})
        s = m.score(self.Xtr.loc[self.ytr == 0])
        share = (s >= m.thresholds["train_normal_q99"]).mean()
        self.assertAlmostEqual(share, 0.01, delta=0.01)

    def test_dummy_predicts_training_majority(self):
        m = self._fit("dummy")
        majority = int(np.bincount(self.ytr).argmax())
        _, preds = m.predict_all(self.Xte)
        self.assertTrue((preds["native"] == majority).all())

    def test_supervised_models_learn_the_signal(self):
        for name in ("logreg", "rf"):
            m = self._fit(name, MD.get_model_spec(name).grid[0])
            _, preds = m.predict_all(self.Xte)
            acc = (preds["val_f1"] == self.yte).mean()
            self.assertGreater(acc, 0.85, name)
            self.assertEqual(set(preds), set(m.spec.threshold_policies))

    def test_tuning_picks_best_validation_pr_auc(self):
        spec = MD.get_model_spec("logreg")
        res = MD.tune_hyperparameters(spec, self.Xtr, self.ytr, self.Xva, self.yva, seed=0)
        best = max(r["val_pr_auc"] for r in res["grid"])
        chosen = [r for r in res["grid"] if r["params"] == res["best_params"]][0]
        self.assertEqual(chosen["val_pr_auc"], best)
        self.assertEqual(len(res["grid"]), len(spec.grid))

    def test_same_seed_same_scores_different_seed_differs(self):
        a = self._fit("rf", {"max_depth": None, "min_samples_leaf": 1}, seed=3).score(self.Xte)
        b = self._fit("rf", {"max_depth": None, "min_samples_leaf": 1}, seed=3).score(self.Xte)
        c = self._fit("rf", {"max_depth": None, "min_samples_leaf": 1}, seed=4).score(self.Xte)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_column_order_and_missing_columns_are_rejected(self):
        m = self._fit("rf", {"max_depth": 3, "min_samples_leaf": 1})
        m.check_columns(self.Xte.columns)
        with self.assertRaises(ValueError):
            m.check_columns(list(self.Xte.columns)[::-1])
        with self.assertRaises(ValueError) as ctx:
            m.check_columns(list(self.Xte.columns)[1:])
        self.assertIn("sıfırla doldurulmaz", str(ctx.exception))

    def test_bundle_roundtrip_gives_identical_scores(self):
        m = self._fit("rf", {"max_depth": 3, "min_samples_leaf": 1})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.joblib"
            MD.save_bundle(m, p)
            back = MD.load_bundle(p)
        np.testing.assert_array_equal(m.score(self.Xte), back.score(self.Xte))
        self.assertEqual(back.features, m.features)
        self.assertEqual(back.thresholds, m.thresholds)

    def test_single_thread_copy_keeps_predictions(self):
        m = self._fit("rf", {"max_depth": 3, "min_samples_leaf": 1})
        pipe = MD.single_thread_copy(m)
        self.assertEqual(pipe.named_steps["clf"].n_jobs, 1)
        np.testing.assert_array_equal(pipe.predict_proba(self.Xte)[:, 1], m.score(self.Xte))


if __name__ == "__main__":
    unittest.main()
