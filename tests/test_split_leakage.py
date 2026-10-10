"""Veri sızıntısı testleri: grup bütünlüğü, tekrar kayıtlar, sabit test kümesi.

Bu testler SENTETİK veriyle yalnızca altyapıyı sınar; gerçek sonuç üretmez.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sdnids.data.split import (
    build_groups,
    load_split,
    make_split,
    save_split,
    verify_split_integrity,
)
from sdnids.data.synthetic import make_synthetic_flows
from sdnids.schema import build_schema_from_frame


class SplitLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_flows(seed=1)
        schema, _ = build_schema_from_frame(cls.df)
        cls.features = schema.names
        cls.result = make_split(cls.df, cls.features, seed=7, min_test_support=30)

    def test_same_flow_id_shares_group(self):
        codes, _ = build_groups(self.df, self.features)
        fid = self.df["flow_id"]
        repeated = fid[fid.duplicated(keep=False)].unique()
        self.assertGreater(len(repeated), 0)
        for f in repeated:
            self.assertEqual(len(set(codes[(fid == f).to_numpy()])), 1)

    def test_exact_duplicate_vectors_share_group(self):
        codes, _ = build_groups(self.df, self.features)
        h = pd.util.hash_pandas_object(self.df[self.features], index=False).to_numpy()
        s = pd.DataFrame({"h": h, "g": codes}).groupby("h")["g"].nunique()
        self.assertTrue((s == 1).all())

    def test_no_integrity_violations(self):
        problems = verify_split_integrity(self.df, self.result.group_ids, self.result.assignment, self.features)
        self.assertEqual(problems, [])

    def test_every_record_assigned_exactly_once(self):
        a = self.result.assignment
        self.assertEqual(len(a), len(self.df))
        self.assertFalse(a.index.duplicated().any())
        self.assertEqual(set(a.index), set(self.df["record_id"]))
        self.assertEqual(set(a.unique()), {"train", "val", "test"})

    def test_ratios_close_to_target_when_groups_are_small(self):
        for name, target in self.result.report["ratios_target"].items():
            self.assertAlmostEqual(self.result.report["ratios_achieved"][name], target, delta=0.08)

    def test_deterministic_for_same_seed_and_different_for_other_seed(self):
        again = make_split(self.df, self.features, seed=7)
        pd.testing.assert_series_equal(self.result.assignment, again.assignment)
        other = make_split(self.df, self.features, seed=8)
        self.assertFalse(self.result.assignment.equals(other.assignment))

    def test_negative_control_record_level_random_split_is_detected(self):
        rng = np.random.default_rng(0)
        bad = pd.Series(
            rng.choice(["train", "val", "test"], size=len(self.df), p=[0.6, 0.2, 0.2]),
            index=self.df["record_id"].to_numpy(),
        )
        problems = verify_split_integrity(self.df, self.result.group_ids, bad, self.features)
        self.assertTrue(problems, "Kayıt düzeyinde rastgele bölme sızıntı olarak yakalanmalıydı.")

    def test_rare_class_is_flagged_not_silently_trusted(self):
        text = " ".join(self.result.report["warnings"])
        self.assertIn("BOTNET", text)
        self.assertIn("güvenilir ölçüm", text)

    def test_chronological_strategy_is_monotone_within_class(self):
        res = make_split(self.df, self.features, strategy="chronological", seed=1)
        self.assertEqual(
            verify_split_integrity(self.df, res.group_ids, res.assignment, self.features), []
        )
        order = {"train": 0, "val": 1, "test": 2}
        codes = res.group_ids
        gtime = self.df["timestamp"].groupby(codes).min()
        gsplit = pd.Series(res.assignment.to_numpy(), index=codes).groupby(level=0).first().map(order)
        dominant = pd.crosstab(codes, self.df["label"].to_numpy()).idxmax(axis=1)
        normal_groups = dominant[dominant == "Normal"].index
        seq = pd.DataFrame({"t": gtime.loc[normal_groups], "s": gsplit.loc[normal_groups]}).sort_values("t")
        self.assertTrue((np.diff(seq["s"].to_numpy()) >= 0).all())

    def test_missing_grouping_information_is_reported_as_limitation(self):
        bare = self.df.drop(columns=["flow_id", "timestamp"])
        _, info = build_groups(bare, self.features, use_duplicates=False, time_block_seconds=None)
        self.assertIn("gruplama", " ".join(info["warnings"]))

    def test_saved_split_is_frozen_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            save_split(self.result, d, {"seed": 7})
            assignment, _, meta = load_split(d)
            pd.testing.assert_series_equal(assignment.sort_index(), self.result.assignment.sort_index())
            self.assertEqual(set(meta["record_id_sha256_by_split"]), {"train", "val", "test"})
            path = Path(d) / "split_assignment.csv"
            table = pd.read_csv(path)
            i = table.index[table["split"] == "test"][0]
            table.loc[i, "split"] = "train"
            table.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_split(d)


if __name__ == "__main__":
    unittest.main()
