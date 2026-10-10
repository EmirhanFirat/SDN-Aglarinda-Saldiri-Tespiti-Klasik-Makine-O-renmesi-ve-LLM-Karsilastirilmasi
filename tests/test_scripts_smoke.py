"""Betiklerin veri -> şema -> denetim -> tekrar indirgeme -> bölme yolunu uçtan uca (sentetik zip ile) sınar."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from sdnids.data.split import load_split
from sdnids.data.synthetic import write_synthetic_insdn_zip
from sdnids.schema import FeatureSchema

CONFIG = {
    "cross_session": {
        "target_files": ["SYNTHETIC_Meta.csv"],
        "shared_files": ["SYNTHETIC_Normal_data.csv"],
    }
}


def run(script: str, *args: str, check: bool = True):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=check,
    )


class ScriptSmokeTests(unittest.TestCase):
    def test_audit_then_both_splits(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            zp = d / "synthetic.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            cfg = d / "cfg.json"
            cfg.write_text(json.dumps(CONFIG), encoding="utf-8")
            out = d / "out"

            r1 = run("01_audit_data.py", "--zip", str(zp), "--out", str(out / "audit"), "--allow-synthetic")
            self.assertIn("kayıt", r1.stdout)
            report = json.loads((out / "audit" / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["data_origin"], "synthetic")
            self.assertIn("duplicates_by_class", report)
            schema = FeatureSchema.load(out / "audit" / "schema.json")

            r2 = run(
                "02_make_split.py", "--zip", str(zp), "--out", str(out), "--config", str(cfg), "--allow-synthetic"
            )
            self.assertIn("within_run", r2.stdout)
            self.assertIn("cross_session", r2.stdout)
            self.assertIn("Tekrar indirgeme", r2.stdout)
            self.assertTrue((out / "dedupe" / "excluded_conflicting_label.csv").exists())

            for design in ("within_run", "cross_session"):
                split_dir = out / f"split_{design}"
                assignment, _, meta = load_split(split_dir)
                self.assertEqual(meta["params"]["schema_digest"], schema.digest())
                self.assertEqual(meta["params"]["design"], design)
                self.assertTrue({"train", "test"} <= set(assignment.unique()))
                weights = pd.read_csv(split_dir / "record_weights.csv")
                self.assertTrue((weights["n_copies"] >= 1).all())
                self.assertEqual(len(weights), len(assignment))
                leak = json.loads((split_dir / "leak_check_train.json").read_text(encoding="utf-8"))
                self.assertEqual(leak["flagged"], {})

    def test_scripts_refuse_synthetic_data_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "synthetic.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            for script in ("01_audit_data.py", "02_make_split.py"):
                r = run(script, "--zip", str(zp), "--out", str(Path(d) / "o"), check=False)
                self.assertNotEqual(r.returncode, 0, script)
                self.assertIn("SyntheticDataError", r.stderr, script)


if __name__ == "__main__":
    unittest.main()
