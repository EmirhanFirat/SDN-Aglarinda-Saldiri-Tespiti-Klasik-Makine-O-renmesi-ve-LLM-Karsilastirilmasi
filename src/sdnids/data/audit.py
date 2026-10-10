"""Veri inceleme: eksik/sonsuz değer, tekrar kayıt, şüpheli etiket, sızıntı işaretleri.

İki ayrı işlev vardır ve bilerek ayrılmıştır:

* audit_quality: tüm veri üzerinde çalışır; model eğitmez, karar vermez, yalnızca
  veri kalitesini ve sızıntı risklerini raporlar.
* single_feature_leak_check: tek özellikli karar kütüğü (depth=1) ile "tek başına
  etiketi neredeyse tamamen ele veren özellik" arar. Hangi özelliğin atılacağına
  karar vermek için kullanıldığından YALNIZCA EĞİTİM bölümünde çalıştırılmalıdır
  (test bilgisi karara karışmasın).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .labels import to_binary


def audit_quality(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    label_col: str = "label",
    source_col: str = "source_file",
    ip_cols: Sequence[str] = ("src_ip", "dst_ip"),
    time_col: str = "timestamp",
) -> dict:
    feature_cols = list(feature_cols)
    X = df[feature_cols]
    arr = X.to_numpy(dtype="float64")  # sayısal değilse burada açık hata verir
    binary = to_binary(df[label_col])
    flags: list[str] = []

    report: dict = {"n_rows": int(len(df)), "n_features": len(feature_cols)}
    report["class_counts"] = {str(k): int(v) for k, v in df[label_col].value_counts().items()}
    report["binary_counts"] = {"normal": int((binary == 0).sum()), "attack": int((binary == 1).sum())}

    if source_col in df.columns:
        ct = pd.crosstab(df[source_col], df[label_col])
        report["class_counts_by_source"] = {
            str(f): {str(c): int(v) for c, v in row.items() if v > 0} for f, row in ct.iterrows()
        }

    missing = X.isna().sum()
    report["missing_by_feature"] = {c: int(v) for c, v in missing.items() if v > 0}
    inf_counts = np.isinf(arr).sum(axis=0)
    report["inf_by_feature"] = {c: int(v) for c, v in zip(feature_cols, inf_counts) if v > 0}
    if report["missing_by_feature"] or report["inf_by_feature"]:
        flags.append(
            "Eksik veya sonsuz değer var; yalnızca eğitim bölümünden öğrenilen medyanla doldurulacak."
        )

    report["constant_features"] = [c for c in feature_cols if X[c].nunique(dropna=True) <= 1]
    if report["constant_features"]:
        flags.append(f"Sabit özellik(ler): {report['constant_features']}")

    row_hash = pd.util.hash_pandas_object(X, index=False).to_numpy()
    hs = pd.Series(row_hash)
    conflict = pd.DataFrame({"h": row_hash, "y": df[label_col].to_numpy()}).groupby("h")["y"].nunique()
    conflict_hashes = conflict[conflict > 1].index
    report["duplicates"] = {
        "extra_copies": int(hs.duplicated(keep="first").sum()),
        "rows_in_duplicate_sets": int(hs.duplicated(keep=False).sum()),
        "conflicting_label_vectors": int(len(conflict_hashes)),
        "rows_in_conflicting_vectors": int(hs.isin(conflict_hashes).sum()),
    }
    if report["duplicates"]["extra_copies"]:
        flags.append("Tekrar eden özellik vektörleri var; aynı bölmeye konacak (grup bütünlüğü).")
    if report["duplicates"]["conflicting_label_vectors"]:
        flags.append("Aynı özellik vektörü farklı etiketlerle görülüyor (şüpheli etiket / ayırt edilemez kayıt).")

    per_class = pd.DataFrame({"h": row_hash, "y": df[label_col].to_numpy()}).groupby("y")["h"].agg(["size", "nunique"])
    report["duplicates_by_class"] = {
        str(c): {
            "rows": int(r["size"]),
            "unique_vectors": int(r["nunique"]),
            "duplicate_share": round(1.0 - float(r["nunique"]) / float(r["size"]), 4),
        }
        for c, r in per_class.iterrows()
    }

    if source_col in df.columns:
        per_file = pd.DataFrame({"f": df[source_col].to_numpy(), "b": binary}).groupby("f")["b"].nunique()
        pure = per_file[per_file == 1].index
        share = float(df[source_col].isin(pure).mean())
        report["source_file_binary_purity"] = {
            "files_with_single_binary_label": [str(x) for x in pure],
            "row_share": share,
        }
        if share > 0.5:
            flags.append(
                "Dosya adı ikili etiketi büyük ölçüde ele veriyor; source_file model girdisine verilmemeli."
            )

    report["ip_purity"] = {}
    for col in ip_cols:
        if col not in df.columns:
            continue
        tmp = pd.DataFrame({"ip": df[col].to_numpy(), "b": binary})
        per_ip = tmp.groupby("ip")["b"].nunique()
        pure_share = float(tmp["ip"].map(per_ip).eq(1).mean())
        report["ip_purity"][col] = {"unique": int(len(per_ip)), "row_share_with_single_label_ip": pure_share}
        if pure_share > 0.9:
            flags.append(
                f"{col}: kayıtların %{pure_share * 100:.1f}'i tek etiketli IP'lerden geliyor; "
                "IP'ye dayalı etiketleme olabilir, IP modele verilmemeli."
            )

    if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        t = df[time_col]
        by_source = {}
        if source_col in df.columns:
            g = t.groupby(df[source_col])
            for f in g.groups:
                s = g.get_group(f)
                by_source[str(f)] = {
                    "min": None if s.isna().all() else s.min().isoformat(),
                    "max": None if s.isna().all() else s.max().isoformat(),
                    "nat": int(s.isna().sum()),
                }
        report["timestamp"] = {"nat_count": int(t.isna().sum()), "by_source": by_source}
        if report["timestamp"]["nat_count"]:
            flags.append("Zaman damgası ayrıştırılamayan kayıtlar var (dayfirst/biçim kontrol edilmeli).")

    shapes = df.attrs.get("timestamp_shapes")
    if shapes:
        report["timestamp_shapes"] = shapes
        mixed = [f for f, s in shapes.items() if len(s) > 1]
        if mixed:
            flags.append(
                f"Aynı dosyada birden fazla zaman damgası biçimi var ({mixed}); dosyalar bir elektronik "
                "tabloda yeniden kaydedilmiş olabilir. Saniye hassasiyeti ve gün/ay sırası güvenilir değil."
            )

    report["flags"] = flags
    return report


def single_feature_leak_check(
    X: pd.DataFrame,
    y_binary,
    *,
    threshold: float = 0.99,
    max_rows: int = 50_000,
    seed: int = 0,
) -> dict:
    """Tek başına etiketi ele veren özellikleri bulur (dengeli doğruluk >= eşik).

    YALNIZCA eğitim bölümünde çalıştırın.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    y = np.asarray(y_binary).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Sızıntı denetimi için iki sınıf da gerekli.")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    if len(idx) > max_rows:
        idx = np.sort(rng.choice(idx, size=max_rows, replace=False))
    arr = np.nan_to_num(X.to_numpy(dtype="float64")[idx], nan=-1.0, posinf=1e18, neginf=-1e18)
    yy = y[idx]
    scores: dict[str, float] = {}
    for j, name in enumerate(X.columns):
        col = arr[:, [j]]
        stump = DecisionTreeClassifier(max_depth=1, random_state=seed).fit(col, yy)
        scores[name] = float(balanced_accuracy_score(yy, stump.predict(col)))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "threshold": threshold,
        "n_rows_used": int(len(idx)),
        "flagged": {k: v for k, v in ranked if v >= threshold},
        "top5": dict(ranked[:5]),
    }


def write_audit_report(report: dict, json_path: str | Path, md_path: str | Path, *, title: str) -> None:
    jp, mp = Path(json_path), Path(md_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [f"# {title}", ""]
    lines += [f"- Kayıt sayısı: {report.get('n_rows')}", f"- Özellik sayısı: {report.get('n_features')}", ""]
    lines += ["## Sınıf dağılımı", "", "| Sınıf | Kayıt |", "|---|---:|"]
    for k, v in report.get("class_counts", {}).items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Uyarılar", ""]
    flags = report.get("flags", [])
    lines += [f"- {f}" for f in flags] if flags else ["- Uyarı yok."]
    dup = report.get("duplicates", {})
    lines += ["", "## Tekrar kayıtlar", ""]
    lines += [f"- {k}: {v}" for k, v in dup.items()]
    by_class = report.get("duplicates_by_class")
    if by_class:
        lines += ["", "| Sınıf | Kayıt | Benzersiz vektör | Tekrar payı |", "|---|---:|---:|---:|"]
        for c, d in by_class.items():
            lines.append(f"| {c} | {d['rows']} | {d['unique_vectors']} | {d['duplicate_share']:.1%} |")
    leak = report.get("single_feature_leak_check")
    if leak:
        lines += ["", "## Tek özellikli sızıntı denetimi (yalnızca eğitim bölümü)", ""]
        lines += [f"- Eşik: {leak['threshold']}", f"- İşaretlenen: {leak['flagged'] or 'yok'}"]
        lines += ["- En yüksek 5: " + ", ".join(f"{k}={v:.3f}" for k, v in leak["top5"].items())]
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
