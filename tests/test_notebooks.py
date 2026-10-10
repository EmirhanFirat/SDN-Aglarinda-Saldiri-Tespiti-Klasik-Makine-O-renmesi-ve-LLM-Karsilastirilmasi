"""Notebook'ların biçimi ve kodu: çıktısız, makineye özgü yol içermiyor, kod hücreleri hatasız çalışıyor.

Kod hücreleri SENTETİK veri üzerinde (betiklerle üretilmiş küçük bir çıktı klasörüyle) sırayla çalıştırılır.
Jupyter gerektirmez; .ipynb JSON'u okunur ve hücreler aynı ad alanında exec edilir.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
ABS_PATH = re.compile(r"(/home/\w+|/mnt/\w|/tmp/\w|/root/\w|[A-Za-z]:\\\\Users)")  # makineye özgü mutlak yol


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class NotebookFormatTests(unittest.TestCase):
    def test_notebooks_exist(self):
        self.assertTrue(NOTEBOOKS, "notebooks/ altında .ipynb yok")

    def test_valid_structure_and_no_outputs(self):
        for nb_path in NOTEBOOKS:
            nb = load(nb_path)
            self.assertEqual(nb["nbformat"], 4, nb_path.name)
            ids = [c.get("id") for c in nb["cells"]]
            self.assertEqual(len(ids), len(set(ids)), f"{nb_path.name}: yinelenen hücre kimliği")
            for c in nb["cells"]:
                self.assertIn(c["cell_type"], ("markdown", "code"))
                self.assertIsInstance(c["source"], list)
                if c["cell_type"] == "code":
                    self.assertEqual(c["outputs"], [], f"{nb_path.name}: çıktılı hücre depoya girmemeli")
                    self.assertIsNone(c["execution_count"])

    def test_every_code_cell_is_preceded_by_an_explanation(self):
        for nb_path in NOTEBOOKS:
            cells = load(nb_path)["cells"]
            for i, c in enumerate(cells):
                if c["cell_type"] == "code" and i > 0:
                    prev = [x for x in cells[:i] if x["cell_type"] == "markdown"]
                    self.assertTrue(prev, f"{nb_path.name}: hücre {i} öncesinde açıklama yok")

    def test_no_machine_specific_paths(self):
        for nb_path in NOTEBOOKS:
            found = ABS_PATH.findall(nb_path.read_text(encoding="utf-8"))
            self.assertEqual(found, [], nb_path.name)


class NotebookExecutionTests(unittest.TestCase):
    """02 notebook'unun kodunu sentetik ortamda baştan sona çalıştırır."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "src"))
        from sdnids.data.synthetic import write_synthetic_insdn_zip

        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.zip = d / "s.zip"
        write_synthetic_insdn_zip(cls.zip, seed=0)
        cfg = {"cross_session": {"target_files": ["SYNTHETIC_Meta.csv"], "shared_files": ["SYNTHETIC_Normal_data.csv"]}}
        (d / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
        cls.out = d / "out"
        cls.fig = d / "fig"

        def run(script, *args):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True,
                           capture_output=True, text=True, cwd=ROOT)

        run("01_audit_data.py", "--zip", str(cls.zip), "--out", str(cls.out / "audit"), "--allow-synthetic")
        run("02_make_split.py", "--zip", str(cls.zip), "--out", str(cls.out), "--config", str(d / "cfg.json"),
            "--allow-synthetic")
        run("03_train_eval.py", "--data", str(cls.zip), "--out", str(cls.out), "--config", str(d / "cfg.json"),
            "--allow-synthetic", "--quick", "--feature-sets", "full", "--evaluate-test")
        run("04_make_figures.py", "--out", str(cls.out), "--figures", str(cls.fig))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_figures_script_produced_pngs(self):
        names = {p.name for p in self.fig.glob("*.png")}
        for expected in ("sinif_dagilimi_within_run.png", "confusion_within_run_full.png", "f1_ga_within_run.png",
                         "fpr_ga_within_run.png", "basari_gecikme_cross_session.png"):
            self.assertIn(expected, names)

    def test_notebook_02_code_runs_top_to_bottom(self):
        nb = load(ROOT / "notebooks" / "02_klasik_modeller.ipynb")
        env_backup = dict(os.environ)
        cwd = os.getcwd()
        os.environ.update({
            "INSDN_DIR": str(self.zip), "SDNIDS_OUT": str(self.out), "SDNIDS_FIG_DIR": str(self.fig),
            "SDNIDS_DEMO_FEATURE_SET": "full", "SDNIDS_ALLOW_SYNTHETIC": "1", "MPLBACKEND": "Agg",
        })
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.chdir(ROOT / "notebooks")
            ns = {"__name__": "__notebook__"}
            for i, c in enumerate(nb["cells"]):
                if c["cell_type"] != "code":
                    continue
                src = "".join(c["source"])
                try:
                    exec(compile(src, f"<hücre {i}>", "exec"), ns)
                except Exception as e:  # hangi hücrede kırıldığını göster
                    self.fail(f"Hücre {i} ({c.get('id')}) hata verdi: {e!r}\n---\n{src[:400]}")
                finally:
                    plt.close("all")
            self.assertIn("test_tab", ns)
            self.assertFalse(ns["test_tab"].empty)
            self.assertTrue((self.fig / "nb02_pr_egrisi_logreg_val.png").exists())
        finally:
            os.chdir(cwd)
            os.environ.clear()
            os.environ.update(env_backup)


if __name__ == "__main__":
    unittest.main()
