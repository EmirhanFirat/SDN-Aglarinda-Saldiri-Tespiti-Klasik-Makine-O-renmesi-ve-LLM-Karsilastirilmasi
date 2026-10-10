import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdnids.data.audit import audit_quality, single_feature_leak_check
from sdnids.data.labels import to_binary
from sdnids.data.synthetic import make_synthetic_flows
from sdnids.schema import build_schema_from_frame


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_flows(seed=2)
        schema, _ = build_schema_from_frame(cls.df)
        cls.features = schema.names
        cls.report = audit_quality(cls.df, cls.features)

    def test_missing_and_infinite_values_are_counted(self):
        self.assertIn("flow_byts_s", self.report["missing_by_feature"])
        self.assertIn("flow_pkts_s", self.report["inf_by_feature"])

    def test_duplicates_and_conflicting_labels_are_found(self):
        d = self.report["duplicates"]
        self.assertGreater(d["extra_copies"], 0)
        self.assertGreater(d["conflicting_label_vectors"], 0)

    def test_file_name_leak_risk_is_flagged(self):
        pure = self.report["source_file_binary_purity"]["files_with_single_binary_label"]
        self.assertIn("SYNTHETIC_Normal_data.csv", pure)
        self.assertIn("source_file", " ".join(self.report["flags"]))

    def test_ip_purity_matches_an_independent_computation(self):
        # Kusurlu veride çelişkili etiketli kayıtlar bazı IP'lere iki etiket karıştırır; oran 1'in altında kalır.
        binary = to_binary(self.df["label"])
        per_ip = (
            self.df.assign(b=binary).groupby("src_ip")["b"].nunique()
        )
        expected = float((self.df["src_ip"].map(per_ip) == 1).mean())
        got = self.report["ip_purity"]["src_ip"]["row_share_with_single_label_ip"]
        self.assertAlmostEqual(got, expected, places=12)
        self.assertLess(got, 1.0)

    def test_ip_based_labeling_is_flagged_on_clean_data_but_shared_server_ip_is_not(self):
        clean = make_synthetic_flows(seed=2, add_defects=False)
        rep = audit_quality(clean, self.features)
        self.assertEqual(rep["ip_purity"]["src_ip"]["row_share_with_single_label_ip"], 1.0)
        self.assertEqual(rep["ip_purity"]["dst_ip"]["row_share_with_single_label_ip"], 0.0)
        text = " ".join(rep["flags"])
        self.assertIn("src_ip", text)
        self.assertNotIn("dst_ip", text)

    def test_duplicates_by_class_are_consistent(self):
        by_class = self.report["duplicates_by_class"]
        self.assertEqual(sum(d["rows"] for d in by_class.values()), len(self.df))
        for d in by_class.values():
            self.assertLessEqual(d["unique_vectors"], d["rows"])
            self.assertAlmostEqual(d["duplicate_share"], 1 - d["unique_vectors"] / d["rows"], places=3)

    def test_class_counts_match_the_frame_and_include_the_rare_class(self):
        # Sentetik kusurlar arasında rastgele tekrar kayıtlar da var; bu yüzden sayı 40'tan biraz fazla olabilir.
        expected = int((self.df["label"] == "BOTNET").sum())
        self.assertEqual(self.report["class_counts"]["BOTNET"], expected)
        self.assertGreaterEqual(expected, 40)
        self.assertEqual(sum(self.report["class_counts"].values()), len(self.df))

    def test_single_feature_leak_check_finds_planted_leak_only(self):
        leaky = make_synthetic_flows(seed=2, leaky_feature=True)
        feats = self.features + ["leaky_feature"]
        res = single_feature_leak_check(leaky[feats], to_binary(leaky["label"]), threshold=0.99)
        self.assertEqual(list(res["flagged"]), ["leaky_feature"])
        clean = single_feature_leak_check(self.df[self.features], to_binary(self.df["label"]), threshold=0.99)
        self.assertEqual(clean["flagged"], {})

    def test_leak_check_needs_two_classes(self):
        one = self.df[self.df["label"] == "Normal"]
        with self.assertRaises(ValueError):
            single_feature_leak_check(one[self.features], to_binary(one["label"]))


if __name__ == "__main__":
    unittest.main()
