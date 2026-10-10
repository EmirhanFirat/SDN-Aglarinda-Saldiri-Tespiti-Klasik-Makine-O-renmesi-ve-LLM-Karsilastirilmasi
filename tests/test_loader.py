import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sdnids.data.loader import load_insdn_zip
from sdnids.data.synthetic import (
    SyntheticDataError,
    is_synthetic,
    require_real_data,
    write_synthetic_insdn_zip,
)


def _zip_with(members: dict[str, str], path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)


class LoaderTests(unittest.TestCase):
    def test_synthetic_zip_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "x.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            df = load_insdn_zip(zp)
        self.assertTrue(df["record_id"].is_unique)
        self.assertEqual(len(df.attrs["source_zip_sha256"]), 64)
        self.assertEqual(df.attrs["unknown_labels"], [])
        self.assertEqual(set(df["label"]), {"Normal", "DoS", "Probe", "BOTNET"})
        self.assertIn("Botnet", set(df["label_raw"]))  # ham etiket korunur, kanonik etiket ayrı
        self.assertTrue(np.isinf(df["flow_pkts_s"]).any())  # "Infinity" dizgisi sayıya çevrildi
        self.assertTrue(df["flow_byts_s"].isna().any())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        self.assertEqual(df.attrs["coerced_to_nan"], {})

    def test_synthetic_data_is_recognised_and_refused_for_results(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "x.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            df = load_insdn_zip(zp)
        self.assertTrue(is_synthetic(df))
        with self.assertRaises(SyntheticDataError):
            require_real_data(df)

    def test_missing_label_column_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "x.zip"
            _zip_with({"a.csv": "Flow ID,Flow Duration\nf1,1\n"}, zp)
            with self.assertRaises(ValueError) as ctx:
                load_insdn_zip(zp)
        self.assertIn("flow_duration", str(ctx.exception))

    def test_unknown_labels_are_reported_not_silently_mapped(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "x.zip"
            _zip_with({"a.csv": "Flow ID,Flow Duration,Label\nf1,1,Normal\nf2,2,Weird\n"}, zp)
            df = load_insdn_zip(zp)
        self.assertEqual(df.attrs["unknown_labels"], ["Weird"])
        self.assertEqual(df.loc[1, "label"], "Weird")

    def test_unsafe_member_names_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "x.zip"
            csv = "Flow ID,Flow Duration,Label\nf1,1,Normal\n"
            _zip_with({"../evil.csv": csv, "/abs.csv": csv, "ok.csv": csv}, zp)
            df = load_insdn_zip(zp)
        self.assertEqual(set(df["source_file"]), {"ok.csv"})

    def test_unparseable_numeric_values_are_counted_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "x.zip"
            _zip_with({"a.csv": "Flow ID,Flow Duration,Label\nf1,1,Normal\nf2,abc,Normal\nf3,3,Normal\n"}, zp)
            df = load_insdn_zip(zp)
        self.assertEqual(df.attrs["coerced_to_nan"], {"a.csv": {"flow_duration": 1}})


if __name__ == "__main__":
    unittest.main()
