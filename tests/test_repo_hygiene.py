"""Depoya giren üretilmiş dosyalarda (raporlar, notebook'lar, belgeler) makineye özgü mutlak yol olmamalı.

Mutlak yol hem başka makinede anlamsızdır hem de çalışılan ortamın kullanıcı adını/dizinini sızdırır.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABS_PATH = re.compile(r"(/home/\w+|/mnt/\w|/tmp/\w|/root/\w|[A-Za-z]:\\\\Users)")
SCAN = ("outputs/**/*.json", "outputs/**/*.md", "notebooks/*.ipynb", "docs/*.md", "README.md", "configs/*.json")


class RepoHygieneTests(unittest.TestCase):
    def test_no_absolute_machine_paths_in_tracked_text_outputs(self):
        offenders = {}
        for pattern in SCAN:
            for path in ROOT.glob(pattern):
                hits = ABS_PATH.findall(path.read_text(encoding="utf-8", errors="replace"))
                if hits:
                    offenders[str(path.relative_to(ROOT))] = sorted(set(hits))[:3]
        self.assertEqual(offenders, {})


if __name__ == "__main__":
    unittest.main()
