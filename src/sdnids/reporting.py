"""Deney çıktılarını (outputs/models/...) okuyup karşılaştırma tablosuna çevirir."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_METRICS = ("accuracy", "attack_precision", "attack_recall", "attack_f1", "macro_f1", "weighted_f1", "fpr", "fnr", "pr_auc")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect_status(root: str | Path) -> pd.DataFrame:
    rows = []
    for st in sorted(Path(root).glob("*/*/*/status.json")):
        rows.append(_read(st))
    return pd.DataFrame(rows)


def collect_results(root: str | Path, split: str = "test") -> pd.DataFrame:
    """Her (tasarım, özellik kümesi, model, eşik kuralı) için tohumlar arası ortalama/std ve birincil tohum GA'ları."""
    rows = []
    for run_dir in sorted(p.parent for p in Path(root).glob("*/*/*/status.json")):
        status = _read(run_dir / "status.json")
        if not status or status.get("status") != "ok":
            continue
        res = _read(run_dir / f"{split}_metrics.json")
        if res is None:
            continue
        meta = _read(run_dir / "run_meta.json") or {}
        speed = meta.get("speed", {})
        sel = _read(run_dir / "selection.json") or {}
        train_secs = [v["train_resources"]["seconds"] for v in sel.get("per_seed", {}).values()]
        train_ram = [v["train_resources"].get("rss_delta_mb", np.nan) for v in sel.get("per_seed", {}).values()]
        details = res.get("primary_seed_details", {})
        for policy, agg in res["aggregate"].items():
            row = {
                "tasarım": meta.get("design", status.get("design")),
                "özellik_kümesi": meta.get("feature_set", status.get("feature_set")),
                "model": meta.get("model", status.get("model")),
                "eşik_kuralı": policy,
                "birincil_kural": policy == meta.get("primary_policy"),
                "tohum_sayısı": agg["attack_f1"]["n"],
                "n_kayıt": res.get("n_records"),
                "n_saldırı": res.get("n_attack"),
            }
            for m in TABLE_METRICS:
                row[m] = agg[m]["mean"]
                row[f"{m}_std"] = agg[m]["std"]
            ci = details.get(policy, {}).get("ci_time_block", {}).get("metrics", {})
            for m in ("attack_f1", "macro_f1", "fpr", "fnr", "pr_auc"):
                if m in ci:
                    row[f"{m}_ga_nokta"] = ci[m]["point"]  # birincil tohumun değeri (GA bu değerin etrafında)
                    row[f"{m}_ga_alt"] = ci[m]["low"]
                    row[f"{m}_ga_üst"] = ci[m]["high"]
            row["gecikme_p50_ms"] = speed.get("model_only", {}).get("p50_ms")
            row["gecikme_p95_ms"] = speed.get("model_only", {}).get("p95_ms")
            row["gecikme_hizalamalı_p95_ms"] = speed.get("with_schema_align", {}).get("p95_ms")
            row["akış_per_sn"] = speed.get("throughput_batch", {}).get("flows_per_s")
            row["eğitim_sn_ort"] = float(np.mean(train_secs)) if train_secs else np.nan
            row["eğitim_ram_artışı_mb"] = float(np.nanmean(train_ram)) if train_ram else np.nan
            row["model_mb"] = meta.get("model_size_mb_pickled")
            rows.append(row)
    return pd.DataFrame(rows)


def per_class_table(root: str | Path, design: str, feature_set: str, model: str, policy: str | None = None) -> pd.DataFrame:
    run_dir = Path(root) / design / feature_set / model
    res = _read(run_dir / "test_metrics.json")
    meta = _read(run_dir / "run_meta.json") or {}
    if res is None:
        raise FileNotFoundError(f"{run_dir}/test_metrics.json yok (test henüz değerlendirilmedi).")
    policy = policy or meta.get("primary_policy")
    return pd.DataFrame(res["primary_seed_details"][policy]["per_class"])


def fmt_mean_std(df: pd.DataFrame, metric: str, digits: int = 4) -> pd.Series:
    return df.apply(lambda r: f"{r[metric]:.{digits}f} ± {r[metric + '_std']:.{digits}f}", axis=1)


def to_markdown(df: pd.DataFrame, columns, digits: int = 4) -> str:
    """tabulate gerektirmeyen basit Markdown tablosu."""
    cols = list(columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if np.isnan(v) else f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
