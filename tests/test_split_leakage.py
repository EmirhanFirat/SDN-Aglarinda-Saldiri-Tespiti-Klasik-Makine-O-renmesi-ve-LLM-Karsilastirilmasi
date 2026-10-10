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


class FragmentRuleTests(unittest.TestCase):
    """flow_id birleştirmesi: aynı dosya + aynı flow_id + en çok 300 sn arayla (elle kurulmuş küçük tablo)."""

    @staticmethod
    def _frame():
        t0 = pd.Timestamp("2020-02-05 10:00:00")
        rows = [
            # r0 ve r1: aynı bağlantının iki parçası (60 sn arayla)  -> aynı grup olmalı
            ("r0", "A.csv", "10.0.0.1-10.0.0.2-1000-80-6", t0, 1.0),
            ("r1", "A.csv", "10.0.0.1-10.0.0.2-1000-80-6", t0 + pd.Timedelta(seconds=60), 2.0),
            # r2: aynı flow_id ama 3 hafta sonra -> YENİ bağlantı, ayrı grup olmalı
            ("r2", "A.csv", "10.0.0.1-10.0.0.2-1000-80-6", t0 + pd.Timedelta(days=21), 3.0),
            # r3: aynı flow_id, FARKLI dosya -> asla birleşmez
            ("r3", "B.csv", "10.0.0.1-10.0.0.2-1000-80-6", t0 + pd.Timedelta(seconds=30), 4.0),
            # r4: başka bir bağlantı
            ("r4", "A.csv", "10.0.0.9-10.0.0.2-2000-80-6", t0, 5.0),
        ]
        return pd.DataFrame(rows, columns=["record_id", "source_file", "flow_id", "timestamp", "f1"])

    def test_close_fragments_share_group_far_apart_do_not(self):
        df = self._frame()
        codes, info = build_groups(df, ["f1"])
        self.assertEqual(codes[0], codes[1])
        self.assertNotEqual(codes[0], codes[2])
        self.assertNotEqual(codes[0], codes[3])
        self.assertNotEqual(codes[0], codes[4])
        self.assertIn("flow_id_fragments_gap_300s", info["keys_used"])

    def test_chain_of_close_fragments_is_one_group(self):
        # Her parça bir öncekinden 200 sn sonra: zincir boyunca tek bağlantı sayılır.
        t0 = pd.Timestamp("2020-02-05 10:00:00")
        df = pd.DataFrame(
            {
                "record_id": [f"r{i}" for i in range(6)],
                "source_file": "A.csv",
                "flow_id": "X",
                "timestamp": [t0 + pd.Timedelta(seconds=200 * i) for i in range(6)],
                "f1": np.arange(6.0),
            }
        )
        codes, _ = build_groups(df, ["f1"])
        self.assertEqual(len(set(codes)), 1)

    def test_without_time_gap_same_flow_id_always_merges(self):
        df = self._frame()
        codes, _ = build_groups(df, ["f1"], fragment_gap_seconds=None)
        self.assertEqual(codes[0], codes[2])  # eski davranış: zaman sınırı yok
        self.assertNotEqual(codes[0], codes[3])  # farklı dosya yine birleşmez

    def test_verify_flags_split_of_adjacent_fragments_but_not_far_apart_rows(self):
        df = self._frame()
        feats = ["f1"]
        codes, _ = build_groups(df, feats)
        ids = df["record_id"].tolist()

        # r2 (3 hafta sonraki aynı flow_id) farklı bölmede olabilir -> ihlal DEĞİL
        ok = pd.Series(["train", "train", "test", "train", "train"], index=ids)
        self.assertEqual(verify_split_integrity(df, codes, ok, feats), [])

        # r0 ve r1 (60 sn arayla) farklı bölmelerde -> ihlal
        bad = pd.Series(["train", "test", "train", "train", "train"], index=ids)
        problems = verify_split_integrity(df, np.arange(len(df)), bad, feats)  # grup bilgisi bilerek yetersiz
        self.assertTrue(any("bağlantı parçası" in p for p in problems), problems)

    def test_missing_timestamp_is_handled_conservatively(self):
        df = self._frame()
        df.loc[2, "timestamp"] = pd.NaT  # zamanı bilinmeyen parça: aralık bilinemez -> bağla
        codes, _ = build_groups(df, ["f1"])
        self.assertEqual(codes[0], codes[2])


if __name__ == "__main__":
    unittest.main()
