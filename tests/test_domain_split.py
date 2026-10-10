"""Koşular arası (senaryo bazlı) bölmenin sızıntı ve kronoloji testleri (sentetik veri)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sdnids.data.dedupe import deduplicate_vectors
from sdnids.data.split import make_domain_holdout_split, verify_split_integrity
from sdnids.data.synthetic import make_synthetic_flows
from sdnids.schema import build_schema_from_frame

TARGET = ["SYNTHETIC_Meta.csv"]
SHARED = ["SYNTHETIC_Normal_data.csv"]


class DomainSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = make_synthetic_flows(seed=11)
        cls.features = build_schema_from_frame(raw)[0].names
        cls.df, _, _ = deduplicate_vectors(raw, cls.features)
        cls.res = make_domain_holdout_split(cls.df, cls.features, target_files=TARGET, shared_files=SHARED, seed=3)
        cls.split = cls.res.assignment.reindex(cls.df["record_id"].to_numpy()).to_numpy()

    def test_target_file_is_entirely_test(self):
        is_target = self.df["source_file"].isin(TARGET).to_numpy()
        self.assertGreater(is_target.sum(), 0)
        self.assertTrue((self.split[is_target] == "test").all())

    def test_source_attack_files_never_reach_test(self):
        ovs = (self.df["source_file"] == "SYNTHETIC_OVS.csv").to_numpy()
        self.assertGreater(ovs.sum(), 0)
        self.assertFalse((self.split[ovs] == "test").any())

    def test_shared_normal_traffic_is_split_chronologically(self):
        normal = (self.df["source_file"] == "SYNTHETIC_Normal_data.csv").to_numpy()
        ts = self.df["timestamp"].to_numpy()
        test_t = ts[normal & (self.split == "test")]
        early_t = ts[normal & (self.split != "test")]
        self.assertGreater(len(test_t), 0)
        self.assertGreater(len(early_t), 0)
        self.assertLess(early_t.max(), test_t.min(), "Test'teki Normal kayıtlar eğitimdekilerden sonra olmalı")
        share = len(test_t) / normal.sum()
        self.assertAlmostEqual(share, 0.2, delta=0.1)

    def test_no_group_crosses_splits(self):
        self.assertEqual(
            verify_split_integrity(self.df, self.res.group_ids, self.res.assignment, self.features), []
        )

    def test_train_val_ratio_in_source_domain(self):
        src = self.split != "test"
        val_share = (self.split[src] == "val").mean()
        self.assertAlmostEqual(val_share, 0.25, delta=0.1)
        self.assertTrue({"train", "val"} <= set(self.split))

    def test_source_only_class_is_reported_as_unmeasurable_in_test(self):
        text = " ".join(self.res.report["warnings"])
        self.assertIn("BOTNET", text)  # BOTNET yalnızca kaynak dosyada
        self.assertIn("ölçüm yapılamaz", text)

    def test_classes_in_both_domains_are_present_in_train_and_test(self):
        table = self.res.report["class_counts_by_split"]
        for cls in ("DoS", "Probe"):
            self.assertGreater(table[cls]["train"], 0)
            self.assertGreater(table[cls]["test"], 0)

    def test_deterministic_for_same_seed(self):
        again = make_domain_holdout_split(self.df, self.features, target_files=TARGET, shared_files=SHARED, seed=3)
        pd.testing.assert_series_equal(self.res.assignment, again.assignment)

    def test_invalid_arguments_fail_loudly(self):
        with self.assertRaises(ValueError):
            make_domain_holdout_split(self.df, self.features, target_files=["yok.csv"], shared_files=SHARED)
        with self.assertRaises(ValueError):
            make_domain_holdout_split(self.df, self.features, target_files=SHARED, shared_files=SHARED)

    def test_same_flow_id_in_different_files_is_not_merged(self):
        # Farklı yakalama dosyalarında aynı flow_id rastlantıyla çıkabilir; bu aynı bağlantı değildir.
        df = self.df.copy()
        a = df.index[df["source_file"] == "SYNTHETIC_Normal_data.csv"][0]
        b = df.index[df["source_file"] == "SYNTHETIC_Meta.csv"][0]
        df.loc[[a, b], "flow_id"] = "AYNI-ID"
        res = make_domain_holdout_split(df, self.features, target_files=TARGET, shared_files=SHARED, seed=3)
        codes = pd.Series(res.group_ids, index=df.index)
        self.assertNotEqual(codes[a], codes[b])
        self.assertEqual(
            verify_split_integrity(df, res.group_ids, res.assignment, self.features), []
        )


if __name__ == "__main__":
    unittest.main()
