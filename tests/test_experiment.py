"""Deney koşucusu testleri: test bölmesinin seçimden yalıtılması, çıktılar, atlanan modeller, betik (sentetik veri)."""

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sdnids import experiment as EX
from sdnids.config import load_config
from sdnids.data.dedupe import deduplicate_vectors
from sdnids.data.split import make_split, save_split
from sdnids.data.synthetic import make_synthetic_flows, write_synthetic_insdn_zip
from sdnids.reporting import collect_results, collect_status, per_class_table
from sdnids.schema import build_schema_from_frame


def small_cfg():
    cfg = load_config()
    cfg["models"].update({
        "n_seeds": 2, "n_jobs": 1,
        "bootstrap": {"n_boot": 20, "alpha": 0.05, "time_block_seconds": 300},
        "latency": {"n_single": 20, "warmup": 2, "batch_size": 500},
        "rf": {"n_estimators": 15}, "iforest": {"n_estimators": 20, "train_normal_quantile": 0.99},
    })
    return cfg


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = make_synthetic_flows(seed=21)
        cls.schema, _ = build_schema_from_frame(raw)
        cls.df, _, _ = deduplicate_vectors(raw, cls.schema.names)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.split_dir = Path(cls.tmp.name) / "split_within_run"
        res = make_split(cls.df, cls.schema.names, seed=5)
        save_split(res, cls.split_dir, {"design": "within_run", "schema_digest": cls.schema.digest()})
        cls.test_ids = set(res.assignment[res.assignment == "test"].index)
        cls.cfg = small_cfg()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ctx(self, df=None):
        return EX.load_design_context(self.df if df is None else df, self.schema, self.split_dir)

    def run_one(self, model, out, evaluate_test, df=None):
        return EX.run_model(self.ctx(df), model, "full", self.cfg, out, evaluate_test=evaluate_test, log=lambda *a: None)

    def test_validation_run_never_builds_test_matrices(self):
        calls = []
        real = EX.split_matrices

        def spy(*a, **k):
            calls.append(k.get("only"))
            return real(*a, **k)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(EX, "split_matrices", side_effect=spy):
            st = self.run_one("logreg", d, evaluate_test=False)
            run = Path(d) / "within_run" / "full" / "logreg"
            self.assertEqual(st["status"], "ok")
            self.assertFalse((run / "test_metrics.json").exists())
            preds = pd.read_csv(run / "predictions_seed42.csv.gz")
            self.assertEqual(set(preds["split"]), {"val"})
            self.assertFalse((Path(d) / "within_run" / "test_access_log.json").exists())
        self.assertTrue(calls)
        for only in calls:
            self.assertIsNotNone(only)
            self.assertNotIn("test", only)

    def test_test_run_writes_metrics_for_exactly_the_frozen_test_records(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_one("rf", d, evaluate_test=True)
            run = Path(d) / "within_run" / "full" / "rf"
            res = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
            preds = pd.read_csv(run / "predictions_seed42.csv.gz")
            self.assertEqual(set(preds.loc[preds["split"] == "test", "record_id"]), self.test_ids)
            self.assertEqual(res["n_records"], len(self.test_ids))
            self.assertEqual(res["aggregate"]["p0.5"]["attack_f1"]["n"], 2)  # iki tohum
            det = res["primary_seed_details"]["p0.5"]
            for key in ("ci_time_block", "ci_leak_group", "per_class", "weighted_by_n_copies"):
                self.assertIn(key, det)
            self.assertIn("pr_auc", det["ci_time_block"]["metrics"])
            log = json.loads((Path(d) / "within_run" / "test_access_log.json").read_text(encoding="utf-8"))
            self.assertEqual(len(log), 1)
            self.assertEqual(log[0]["model"], "rf")
            meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["test_evaluated"])
            self.assertIn("Dahil DEĞİL", meta["speed"]["scope"]["model_only"])
            self.assertTrue((run / "model_seed42.joblib").exists())

    def test_selection_is_independent_of_test_labels(self):
        df_perm = self.df.copy()
        is_test = df_perm["record_id"].isin(self.test_ids).to_numpy()
        rng = np.random.default_rng(0)
        df_perm.loc[is_test, "label"] = rng.permutation(np.where(rng.random(is_test.sum()) < 0.5, "Normal", "DoS"))
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for model in ("logreg", "iforest"):
                self.run_one(model, a, evaluate_test=True)
                self.run_one(model, b, evaluate_test=True, df=df_perm)
                ra, rb = (Path(x) / "within_run" / "full" / model for x in (a, b))
                sa = json.loads((ra / "selection.json").read_text(encoding="utf-8"))
                sb = json.loads((rb / "selection.json").read_text(encoding="utf-8"))
                self.assertEqual(sa["best_params"], sb["best_params"], model)
                self.assertEqual([g["val_pr_auc"] for g in sa["grid"]], [g["val_pr_auc"] for g in sb["grid"]], model)
                self.assertEqual({s: v["thresholds"] for s, v in sa["per_seed"].items()},
                                 {s: v["thresholds"] for s, v in sb["per_seed"].items()}, model)
                va = json.loads((ra / "val_metrics.json").read_text(encoding="utf-8"))["per_seed"]
                vb = json.loads((rb / "val_metrics.json").read_text(encoding="utf-8"))["per_seed"]
                self.assertEqual(va, vb, model)
                ta = json.loads((ra / "test_metrics.json").read_text(encoding="utf-8"))["per_seed"]
                tb = json.loads((rb / "test_metrics.json").read_text(encoding="utf-8"))["per_seed"]
                self.assertNotEqual(ta, tb, "test etiketleri değişince test sonucu değişmeli (negatif kontrol)")

    def test_stale_test_results_are_removed_by_a_validation_run(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_one("dummy", d, evaluate_test=True)
            run = Path(d) / "within_run" / "full" / "dummy"
            self.assertTrue((run / "test_metrics.json").exists())
            self.run_one("dummy", d, evaluate_test=False)
            self.assertFalse((run / "test_metrics.json").exists())

    def test_deterministic_models_run_once_stochastic_models_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_one("dummy", d, evaluate_test=False)
            self.run_one("iforest", d, evaluate_test=False)
            v_d = json.loads((Path(d) / "within_run/full/dummy/val_metrics.json").read_text(encoding="utf-8"))
            v_i = json.loads((Path(d) / "within_run/full/iforest/val_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(v_d["per_seed"]), 1)
            self.assertEqual(len(v_i["per_seed"]), 2)
            meta = json.loads((Path(d) / "within_run/full/iforest/run_meta.json").read_text(encoding="utf-8"))
            self.assertLess(meta["n_train_used_for_fit"], meta["n_train"])  # yalnızca Normal ile
            self.assertIn("YALNIZCA", meta["training_information"])

    def test_missing_xgboost_is_recorded_as_skipped(self):
        try:
            import xgboost  # noqa: F401
            self.skipTest("xgboost kurulu")
        except ImportError:
            pass
        with tempfile.TemporaryDirectory() as d:
            st = self.run_one("xgboost", d, evaluate_test=False)
            self.assertEqual(st["status"], "skipped")
            self.assertIn("xgboost", st["reason"])
            s = collect_status(d)
            self.assertEqual(list(s["status"]), ["skipped"])

    def test_data_and_split_must_match(self):
        with self.assertRaises(ValueError):
            self.ctx(self.df.iloc[1:])
        other, _ = build_schema_from_frame(self.df.drop(columns=["syn_flag_cnt"]))
        with self.assertRaises(ValueError):
            EX.load_design_context(self.df, other, self.split_dir)

    def test_reporting_collects_results(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_one("logreg", d, evaluate_test=True)
            t = collect_results(d, "test")
            self.assertEqual(set(t["eşik_kuralı"]), {"p0.5", "val_f1"})
            self.assertEqual(int(t["birincil_kural"].sum()), 1)
            for c in ("attack_f1", "fpr", "attack_f1_ga_alt", "gecikme_p95_ms", "akış_per_sn"):
                self.assertIn(c, t.columns)
            pc = per_class_table(d, "within_run", "full", "logreg")
            self.assertIn("Normal", set(pc["sınıf"]))


class TrainEvalScriptTests(unittest.TestCase):
    def test_script_end_to_end_quick_and_refuses_synthetic_without_flag(self):
        cfg = {"cross_session": {"target_files": ["SYNTHETIC_Meta.csv"], "shared_files": ["SYNTHETIC_Normal_data.csv"]}}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            zp = d / "s.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            (d / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
            out = d / "out"

            def run(script, *args, check=True):
                return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                                      capture_output=True, text=True, cwd=ROOT, check=check)

            run("01_audit_data.py", "--zip", str(zp), "--out", str(out / "audit"), "--allow-synthetic")
            run("02_make_split.py", "--zip", str(zp), "--out", str(out), "--config", str(d / "cfg.json"), "--allow-synthetic")
            refused = run("03_train_eval.py", "--data", str(zp), "--out", str(out), check=False)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("SyntheticDataError", refused.stderr)

            r = run("03_train_eval.py", "--data", str(zp), "--out", str(out), "--config", str(d / "cfg.json"),
                    "--allow-synthetic", "--quick", "--feature-sets", "full", "--models", "dummy", "logreg", "rf",
                    "iforest", "--evaluate-test", check=False)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            self.assertIn("Özet (TEST)", r.stdout)
            for split in ("val", "test"):
                self.assertTrue((out / "models" / f"summary_{split}.md").exists())
            for design in ("within_run", "cross_session"):
                self.assertTrue((out / "models" / design / "test_access_log.json").exists())

            # Sentetik veride LLM kümesinin sütunları yok: sessizce geçilmez, 'failed' kaydı ve hata kodu.
            r2 = run("03_train_eval.py", "--data", str(zp), "--out", str(out), "--config", str(d / "cfg.json"),
                     "--allow-synthetic", "--quick", "--feature-sets", "llm_subset", "--models", "dummy",
                     "--designs", "within_run", check=False)
            self.assertNotEqual(r2.returncode, 0)
            st = json.loads((out / "models/within_run/llm_subset/dummy/status.json").read_text(encoding="utf-8"))
            self.assertEqual(st["status"], "failed")
            self.assertIn("şemada yok", st["error"])


if __name__ == "__main__":
    unittest.main()
