"""Betiklerin zip -> şema -> denetim -> bölme yolunu uçtan uca (sentetik zip ile) sınar."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdnids.data.split import load_split
from sdnids.data.synthetic import write_synthetic_insdn_zip
from sdnids.schema import FeatureSchema


def run(script: str, *args: str, check: bool = True):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=check,
    )


class ScriptSmokeTests(unittest.TestCase):
    def test_audit_then_split_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            zp = d / "synthetic.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            audit_dir, split_dir = d / "audit", d / "split"

            r1 = run("01_audit_data.py", "--zip", str(zp), "--out", str(audit_dir), "--allow-synthetic")
            self.assertIn("kayıt", r1.stdout)
            report = json.loads((audit_dir / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["data_origin"], "synthetic")
            self.assertTrue((audit_dir / "audit.md").exists())
            schema = FeatureSchema.load(audit_dir / "schema.json")

            r2 = run("02_make_split.py", "--zip", str(zp), "--out", str(split_dir), "--allow-synthetic")
            self.assertIn("Bölme yazıldı", r2.stdout)
            assignment, _, meta = load_split(split_dir)
            self.assertEqual(meta["params"]["schema_digest"], schema.digest())
            self.assertEqual(set(assignment.unique()), {"train", "val", "test"})
            leak = json.loads((split_dir / "leak_check_train.json").read_text(encoding="utf-8"))
            self.assertEqual(leak["flagged"], {})

    def test_scripts_refuse_synthetic_data_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "synthetic.zip"
            write_synthetic_insdn_zip(zp, seed=0)
            r = run("01_audit_data.py", "--zip", str(zp), "--out", str(Path(d) / "a"), check=False)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("SyntheticDataError", r.stderr)


if __name__ == "__main__":
    unittest.main()
