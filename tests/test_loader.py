import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sdnids.data.audit import audit_quality
from sdnids.data.loader import load_insdn, load_insdn_dir, load_insdn_zip, parse_timestamps
from sdnids.schema import build_schema_from_frame
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


class TimestampTests(unittest.TestCase):
    def test_real_insdn_formats_are_read_day_first(self):
        # Örnekler gerçek InSDN CSV'lerinden alınmıştır.
        raw = pd.Series(
            ["5/2/2020 13:58", "12/1/2020 1:14", "25/12/2019 05:20:00 PM", "05/02/2020 07:19:05 PM"]
        )
        got = parse_timestamps(raw, dayfirst=True)
        expected = pd.to_datetime(
            ["2020-02-05 13:58:00", "2020-01-12 01:14:00", "2019-12-25 17:20:00", "2020-02-05 19:19:05"]
        )
        self.assertEqual(list(got), list(expected))

    def test_iso_is_never_misread_when_dayfirst_is_true(self):
        # pandas'ın 'mixed' ayrıştırması burada 2026-01-02'yi 1 Şubat sanıyordu.
        got = parse_timestamps(pd.Series(["2026-01-02 10:00:00"]), dayfirst=True)
        self.assertEqual(got.iloc[0], pd.Timestamp("2026-01-02 10:00:00"))

    def test_dayfirst_false_reads_month_first(self):
        got = parse_timestamps(pd.Series(["5/2/2020 13:58"]), dayfirst=False)
        self.assertEqual(got.iloc[0], pd.Timestamp("2020-05-02 13:58:00"))

    def test_unknown_format_becomes_nat_not_a_wrong_date(self):
        got = parse_timestamps(pd.Series(["garbage", "2020/02/05"]), dayfirst=True)
        self.assertTrue(got.isna().all())

    def test_mixed_formats_in_one_file_are_recorded_and_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "x.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            df = load_insdn_zip(zp, dayfirst=True)
        self.assertTrue(all(len(s) >= 2 for s in df.attrs["timestamp_shapes"].values()))
        self.assertEqual(int(df["timestamp"].isna().sum()), 0)
        schema, _ = build_schema_from_frame(df)
        report = audit_quality(df, schema.names)
        self.assertIn("birden fazla zaman damgası biçimi", " ".join(report["flags"]))

    def test_loader_roundtrip_preserves_time_to_the_minute(self):
        from sdnids.data.synthetic import make_synthetic_flows

        original = make_synthetic_flows(seed=0)
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "x.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            df = load_insdn_zip(zp, dayfirst=True)
        a = original["timestamp"].dt.floor("min").value_counts().sort_index()
        b = df["timestamp"].dt.floor("min").value_counts().sort_index()
        pd.testing.assert_series_equal(a, b, check_names=False)


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

    def test_directory_loader_matches_zip_loader_and_records_file_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            zp = d / "x.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            extracted = d / "csvs"
            with zipfile.ZipFile(zp) as zf:
                for info in zf.infolist():
                    if info.filename.lower().endswith(".csv"):
                        (extracted / Path(info.filename).name).parent.mkdir(exist_ok=True)
                        (extracted / Path(info.filename).name).write_bytes(zf.read(info))
            from_zip = load_insdn_zip(zp)
            from_dir = load_insdn_dir(extracted)
            from_auto = load_insdn(extracted)
        key = ["label", "flow_duration", "flow_pkts_s", "src_ip"]
        a = from_zip.sort_values(key).reset_index(drop=True)[key]
        b = from_dir.sort_values(key).reset_index(drop=True)[key]
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(from_dir), len(from_auto))
        self.assertEqual(len(from_dir.attrs["source_sha256"]), 64)
        self.assertEqual(len(from_dir.attrs["source_file_sha256"]), 3)
        self.assertTrue(from_dir["record_id"].is_unique)

    def test_empty_directory_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_insdn_dir(d)

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
