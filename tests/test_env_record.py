import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sdnids.env_record import collect_environment, set_global_seed, write_environment


class EnvRecordTests(unittest.TestCase):
    def test_required_fields_present(self):
        info = collect_environment(seed=42)
        for key in ("python", "platform", "cpu_count", "ram_bytes", "gpus", "packages", "seed", "recorded_at_utc"):
            self.assertIn(key, info)
        self.assertEqual(info["seed"], 42)
        self.assertIn("scikit-learn", info["packages"])

    def test_write_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "env.json"
            write_environment(p, seed=1)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["seed"], 1)

    def test_seed_makes_numpy_deterministic(self):
        set_global_seed(123)
        a = np.random.rand(3)
        set_global_seed(123)
        b = np.random.rand(3)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
