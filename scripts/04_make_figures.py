"""Rapor grafiklerini outputs/ altındaki sonuçlardan üretir (veri dosyası gerekmez).

Üretilenler (figures/):
  sinif_dagilimi_<tasarım>.png        bölme başına sınıf sayıları (tekrarlar indirgenmiş)
  confusion_<tasarım>_<özellik>.png   birincil eşik kuralıyla test confusion matrix'leri (birincil tohum)
  f1_ga_, macro_f1_ga_, fpr_ga_<tasarım>.png  saldırı F1, macro-F1, yanlış alarm oranı; %95 küme bootstrap GA
  basari_gecikme_<tasarım>.png        macro-F1 - tek kayıt gecikmesi (p95)

Kullanım: python scripts/04_make_figures.py [--out outputs] [--figures figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sdnids.plots import (  # noqa: E402
    FEATURE_SET_LABELS,
    MODEL_LABELS,
    plot_class_distribution,
    plot_confusion,
    plot_metric_ci,
    plot_success_latency,
    SURFACE,
)
from sdnids.reporting import collect_results  # noqa: E402

DESIGN_TITLES = {"within_run": "within_run (aynı yakalamalar)", "cross_session": "cross_session (görülmemiş oturum)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    args = ap.parse_args()
    out, fig_dir = Path(args.out), Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    for design in ("within_run", "cross_session"):
        meta_path = out / f"split_{design}" / "split_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            table = pd.DataFrame(meta["report"]["class_counts_by_split"]).T
            p = fig_dir / f"sinif_dagilimi_{design}.png"
            plot_class_distribution(table, title=f"Sınıf dağılımı: {DESIGN_TITLES[design]}", path=p)
            made.append(p)
            plt.close("all")

    tab = collect_results(out / "models", "test")
    if tab.empty:
        print("Test sonucu yok; yalnızca sınıf dağılımı grafikleri üretildi.")
    for design in sorted(set(tab["tasarım"])) if not tab.empty else []:
        t = tab[tab["tasarım"] == design]
        for fs in sorted(set(t["özellik_kümesi"])):
            rows = t[(t["özellik_kümesi"] == fs) & t["birincil_kural"]]
            models = [m for m in MODEL_LABELS if m in set(rows["model"])]
            fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2), facecolor=SURFACE)
            axes = [axes] if len(models) == 1 else list(axes)
            for ax, m in zip(axes, models):
                res = json.loads((out / "models" / design / fs / m / "test_metrics.json").read_text(encoding="utf-8"))
                rmeta = json.loads((out / "models" / design / fs / m / "run_meta.json").read_text(encoding="utf-8"))
                pol = rmeta["primary_policy"]
                counts = res["per_seed"][str(res["primary_seed"])][pol]
                plot_confusion(counts, title=f"{MODEL_LABELS[m]} ({pol})", ax=ax)
            fig.suptitle(f"Test confusion matrix: {DESIGN_TITLES[design]}, {FEATURE_SET_LABELS[fs]}", x=0.01, ha="left",
                         fontsize=10)
            fig.tight_layout()
            p = fig_dir / f"confusion_{design}_{fs}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURFACE)
            made.append(p)
            plt.close("all")
        short = {"attack_f1": "f1", "macro_f1": "macro_f1", "fpr": "fpr"}
        for metric, label in (("attack_f1", "saldırı F1 (test)"), ("macro_f1", "macro-F1 (test)"),
                              ("fpr", "yanlış alarm oranı FPR (test)")):
            p = fig_dir / f"{short[metric]}_ga_{design}.png"
            plot_metric_ci(t, metric, title=f"{label}: {DESIGN_TITLES[design]}", xlabel=f"{label}; birincil tohum, çubuk: %95 GA",
                           path=p, log=(metric == "fpr"))
            made.append(p)
            plt.close("all")
        p = fig_dir / f"basari_gecikme_{design}.png"
        plot_success_latency(t, title=f"Başarı ve gecikme: {DESIGN_TITLES[design]}", path=p)
        made.append(p)
        plt.close("all")

    for p in made:
        print("yazıldı:", p)


if __name__ == "__main__":
    main()
