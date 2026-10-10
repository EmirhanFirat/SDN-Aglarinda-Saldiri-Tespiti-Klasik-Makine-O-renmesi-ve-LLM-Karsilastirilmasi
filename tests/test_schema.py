import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sdnids.schema import (
    FeatureSchema,
    FeatureSpec,
    ForbiddenColumnError,
    SchemaMismatchError,
    assert_no_forbidden_columns,
    build_schema_from_frame,
    guess_unit,
    normalize_column_name,
)


def _frame():
    df = pd.DataFrame(
        {
            "Flow ID": ["a", "b"],
            "Src IP": ["1.1.1.1", "2.2.2.2"],
            "Timestamp": ["t1", "t2"],
            "Flow Duration": [1.0, 2.0],
            "Tot Fwd Pkts": [3.0, 4.0],
            "Label": ["Normal", "DoS"],
            "source_file": ["f", "f"],
            "Some Text": ["x", "y"],
        }
    )
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


class NormalizeTests(unittest.TestCase):
    def test_known_examples(self):
        self.assertEqual(normalize_column_name("Flow Byts/s"), "flow_byts_s")
        self.assertEqual(normalize_column_name(" Src IP "), "src_ip")
        self.assertEqual(normalize_column_name("TotLen Fwd Pkts"), "totlen_fwd_pkts")
        self.assertEqual(normalize_column_name("SYN Flag Cnt"), "syn_flag_cnt")

    def test_idempotent(self):
        for n in ["Flow Byts/s", "Fwd IAT Mean", "Label"]:
            once = normalize_column_name(n)
            self.assertEqual(once, normalize_column_name(once))


class SchemaTests(unittest.TestCase):
    def test_forbidden_and_text_columns_never_become_features(self):
        schema, skipped = build_schema_from_frame(_frame())
        self.assertEqual(schema.names, ["flow_duration", "tot_fwd_pkts"])
        self.assertIn("some_text", skipped)

    def test_digest_depends_on_order(self):
        a = FeatureSchema([FeatureSpec("x"), FeatureSpec("y")])
        b = FeatureSchema([FeatureSpec("y"), FeatureSpec("x")])
        self.assertNotEqual(a.digest(), b.digest())

    def test_constructor_rejects_forbidden_feature(self):
        with self.assertRaises(ForbiddenColumnError):
            FeatureSchema([FeatureSpec("flow_duration"), FeatureSpec("label")])

    def test_assert_no_forbidden_columns_normalizes_names(self):
        with self.assertRaises(ForbiddenColumnError):
            assert_no_forbidden_columns(["Flow Duration", "Label"])
        with self.assertRaises(ForbiddenColumnError):
            assert_no_forbidden_columns(["Src IP"])
        assert_no_forbidden_columns(["Flow Duration", "Tot Fwd Pkts"])

    def test_validate_columns_detects_missing_extra_and_order(self):
        schema = FeatureSchema([FeatureSpec("a"), FeatureSpec("b")])
        schema.validate_columns(["a", "b"])
        for bad in (["a"], ["a", "b", "c"], ["b", "a"]):
            with self.assertRaises(SchemaMismatchError):
                schema.validate_columns(bad)

    def test_align_reorders_but_never_zero_fills(self):
        schema = FeatureSchema([FeatureSpec("a"), FeatureSpec("b")])
        shuffled = pd.DataFrame({"b": [2.0], "z": [9.0], "a": [1.0]})
        self.assertEqual(list(schema.align(shuffled).columns), ["a", "b"])
        with self.assertRaises(SchemaMismatchError):
            schema.align(pd.DataFrame({"a": [1.0]}))

    def test_save_load_roundtrip_and_tamper_detection(self):
        schema = FeatureSchema([FeatureSpec("a", unit_guess="bayt"), FeatureSpec("b")], source="test")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "schema.json"
            schema.save(p)
            loaded = FeatureSchema.load(p)
            self.assertEqual(loaded.names, schema.names)
            self.assertEqual(loaded.digest(), schema.digest())
            p.write_text(p.read_text(encoding="utf-8").replace('"name": "a"', '"name": "c"'), encoding="utf-8")
            with self.assertRaises(SchemaMismatchError):
                FeatureSchema.load(p)

    def test_unit_guess_is_a_hint_not_a_verified_unit(self):
        self.assertEqual(guess_unit("flow_byts_s"), "1/saniye")
        self.assertEqual(guess_unit("flow_duration"), "mikrosaniye")
        self.assertEqual(guess_unit("totlen_fwd_pkts"), "bayt")
        self.assertEqual(guess_unit("tot_fwd_pkts"), "paket")
        schema, _ = build_schema_from_frame(_frame())
        self.assertTrue(all(f.unit is None for f in schema.features))


if __name__ == "__main__":
    unittest.main()
