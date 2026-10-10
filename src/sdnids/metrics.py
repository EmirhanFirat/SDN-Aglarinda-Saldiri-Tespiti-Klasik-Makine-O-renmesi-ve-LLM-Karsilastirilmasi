"""Değerlendirme metrikleri, küme (oturum) düzeyinde bootstrap güven aralığı, gecikme ve kaynak ölçümü.

İkili görev: 0 = Normal, 1 = Saldırı. "Pozitif sınıf" her yerde saldırıdır.

Tanımlar (confusion matrix: TN, FP, FN, TP):
    accuracy          = (TP + TN) / toplam
    attack_precision  = TP / (TP + FP)          saldırı dediklerimizin kaçı gerçekten saldırı
    attack_recall     = TP / (TP + FN)          gerçek saldırıların kaçını yakaladık
    attack_f1         = 2TP / (2TP + FP + FN)
    FPR (yanlış alarm oranı)    = FP / (FP + TN)  normal trafiğin kaçına yanlışlıkla alarm verdik
    FNR (kaçırma oranı)         = FN / (FN + TP)  = 1 - recall
    macro_f1          = iki sınıfın (Normal, Saldırı) F1 değerinin ortalaması (destek > 0 olan sınıflar)
    weighted_f1       = aynı F1 değerlerinin sınıf örnek sayılarıyla ağırlıklı ortalaması
    pr_auc            = precision-recall eğrisi altındaki alan, sklearn average_precision_score ile
                        (basamaklı toplam, interpolasyonsuz; saldırı = pozitif sınıf; sürekli skor gerekir)

Tanımsız oranlar (payda 0) NaN döner; sessizce 0 yapılmaz.
"""

from __future__ import annotations

import math
import pickle
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:  # Windows'ta yok
    import resource
except ImportError:  # pragma: no cover
    resource = None

PR_AUC_METHOD = (
    "sklearn.metrics.average_precision_score; saldırı = pozitif sınıf; "
    "AP = sum_n (R_n - R_(n-1)) * P_n (basamaklı, interpolasyonsuz)"
)

LATENCY_SCOPE = {
    "model_only": (
        "Tek kayıt (1 x F sayısal dizi): sonsuz->eksik, medyan doldurma, (varsa) ölçekleme ve model tahmini. "
        "Dahil DEĞİL: özellik çıkarma, serileştirme, ağ isteği, DataFrame oluşturma."
    ),
    "with_schema_align": (
        "Tek satırlık DataFrame -> şema denetimi/sütun sıralama (FeatureSchema.align) -> model_only adımları. "
        "Dahil DEĞİL: özellik çıkarma (trafikten), serileştirme, ağ isteği."
    ),
}

HEADLINE_METRICS = (
    "accuracy",
    "attack_precision",
    "attack_recall",
    "attack_f1",
    "macro_f1",
    "weighted_f1",
    "fpr",
    "fnr",
)


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else float("nan")


def counts_from_arrays(y_true, y_pred, sample_weight=None) -> tuple[float, float, float, float]:
    """(TN, FP, FN, TP). sample_weight verilirse ağırlıklı sayımlar."""
    yt = np.asarray(y_true).astype(np.int64)
    yp = np.asarray(y_pred).astype(np.int64)
    if yt.shape != yp.shape:
        raise ValueError("y_true ve y_pred aynı uzunlukta olmalı.")
    if not (np.isin(yt, (0, 1)).all() and np.isin(yp, (0, 1)).all()):
        raise ValueError("Etiketler ve tahminler yalnızca 0/1 olmalı.")
    c = np.bincount(yt * 2 + yp, weights=sample_weight, minlength=4).astype(np.float64)
    return float(c[0]), float(c[1]), float(c[2]), float(c[3])


def metrics_from_counts(tn: float, fp: float, fn: float, tp: float) -> dict:
    n_normal, n_attack = tn + fp, tp + fn
    f1_attack = _div(2 * tp, 2 * tp + fp + fn)
    f1_normal = _div(2 * tn, 2 * tn + fn + fp)
    f1s, supports = [], []
    for f1, support in ((f1_normal, n_normal), (f1_attack, n_attack)):
        if support > 0:
            f1s.append(f1)
            supports.append(support)
    macro = float(np.mean(f1s)) if f1s else float("nan")
    weighted = float(np.average(f1s, weights=supports)) if f1s else float("nan")
    return {
        "accuracy": _div(tn + tp, tn + fp + fn + tp),
        "attack_precision": _div(tp, tp + fp),
        "attack_recall": _div(tp, tp + fn),
        "attack_f1": f1_attack,
        "normal_precision": _div(tn, tn + fn),
        "normal_recall": _div(tn, tn + fp),
        "normal_f1": f1_normal,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "fpr": _div(fp, fp + tn),
        "fnr": _div(fn, fn + tp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n_normal": n_normal,
        "n_attack": n_attack,
    }


def pr_auc(y_true, score, sample_weight=None) -> float:
    """Saldırı sınıfı için PR-AUC (bkz. PR_AUC_METHOD). Tek sınıf varsa NaN."""
    yt = np.asarray(y_true).astype(np.int64)
    s = np.asarray(score, dtype=np.float64)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if w is not None:
        keep = w > 0
        yt, s, w = yt[keep], s[keep], w[keep]
    if len(yt) == 0 or yt.min() == yt.max():
        return float("nan")
    if np.isnan(s).any():
        raise ValueError("Skorlarda NaN var.")
    return float(average_precision_score(yt, s, sample_weight=w))


def binary_metrics(y_true, y_pred, score=None, sample_weight=None) -> dict:
    out = metrics_from_counts(*counts_from_arrays(y_true, y_pred, sample_weight))
    out["pr_auc"] = pr_auc(y_true, score, sample_weight) if score is not None else float("nan")
    return out


def confusion_matrix_df(y_true, y_pred) -> pd.DataFrame:
    tn, fp, fn, tp = counts_from_arrays(y_true, y_pred)
    return pd.DataFrame(
        [[int(tn), int(fp)], [int(fn), int(tp)]],
        index=["gerçek Normal", "gerçek Saldırı"],
        columns=["tahmin Normal", "tahmin Saldırı"],
    )


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Oran için Wilson aralığı. Kayıtların bağımsız olduğunu varsayar (ilişkili akışlarda iyimser)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def per_class_detection(
    labels, y_pred, *, normal_label: str = "Normal", min_support: int = 30
) -> pd.DataFrame:
    """Özgün sınıf başına (ikili modelin) sonucu.

    Normal için: doğru 'Normal' denen oran (özgüllük = 1 - FPR).
    Saldırı sınıfları için: 'saldırı' diye yakalanma oranı (recall).
    reliable=False: örnek sayısı min_support altında; bu sınıf için güvenilir ölçüm iddia edilemez.
    """
    lab = np.asarray(labels).astype(str)
    yp = np.asarray(y_pred).astype(np.int64)
    rows = []
    for cls in sorted(pd.unique(lab), key=lambda c: (c != normal_label, c)):
        m = lab == cls
        n = int(m.sum())
        if cls == normal_label:
            k = int((yp[m] == 0).sum())
            rate_name = "özgüllük (doğru Normal)"
        else:
            k = int((yp[m] == 1).sum())
            rate_name = "yakalama (recall)"
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "sınıf": cls,
                "n": n,
                "doğru_sayısı": k,
                "oran": _div(k, n),
                "oran_adı": rate_name,
                "wilson_alt": lo,
                "wilson_üst": hi,
                "güvenilir": n >= min_support,
            }
        )
    return pd.DataFrame(rows)


def cluster_ids_from_frame(
    df: pd.DataFrame,
    *,
    block_seconds: float = 300.0,
    source_col: str = "source_file",
    time_col: str = "timestamp",
) -> np.ndarray:
    """Bağımsız birim olarak kullanılacak küme kimliği: aynı dosya + aynı zaman bloğu (varsayılan 5 dk).

    Zamanda yakın akışlar (aynı koşu/oturum) ilişkilidir; bu yüzden güven aralığı kayıt değil bu
    kümeler yeniden örneklenerek hesaplanır. Zaman damgası eksik kayıtlar kendi kümesini oluşturur.
    """
    t = df[time_col]
    secs = (t - t.min()).dt.total_seconds()
    missing = t.isna().to_numpy()
    block = np.floor(secs.fillna(0.0).to_numpy() / float(block_seconds)).astype(np.int64)
    key = np.array([f"{s}|{b}" for s, b in zip(df[source_col].astype(str).to_numpy(), block)], dtype=object)
    if missing.any():
        key[missing] = [f"nat#{i}" for i in np.flatnonzero(missing)]
    return key


def cluster_bootstrap(
    y_true,
    y_pred,
    score=None,
    clusters=None,
    *,
    n_boot: int = 300,
    seed: int = 0,
    alpha: float = 0.05,
    include_pr_auc: bool = True,
    sample_weight=None,
) -> dict:
    """Küme düzeyinde bootstrap: kümeler (grup/oturum/zaman bloğu) yerine koyarak yeniden örneklenir.

    Her çekilişte küme k kez seçildiyse o kümedeki tüm kayıtlar k ağırlığıyla sayılır; bu, kümeler
    yerine koyarak çekilip kayıtların kopyalanmasıyla birebir aynıdır. Yüzdelik (percentile) aralığı döner.
    """
    yt = np.asarray(y_true).astype(np.int64)
    yp = np.asarray(y_pred).astype(np.int64)
    n = len(yt)
    if clusters is None:
        clusters = np.arange(n)
    codes, uniques = pd.factorize(pd.Series(np.asarray(clusters)))
    n_clusters = len(uniques)
    base_w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    sc = None if score is None else np.asarray(score, dtype=np.float64)

    point = binary_metrics(yt, yp, sc if include_pr_auc else None, sample_weight)
    keys = list(HEADLINE_METRICS) + (["pr_auc"] if include_pr_auc and sc is not None else [])
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, len(keys)), np.nan)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters).astype(np.float64)
        w = counts[codes] * base_w
        m = metrics_from_counts(*counts_from_arrays(yt, yp, w))
        if "pr_auc" in keys:
            m["pr_auc"] = pr_auc(yt, sc, w)
        draws[b] = [m[k] for k in keys]
    with warnings.catch_warnings():  # tümüyle tanımsız metrik (ör. hiç saldırı denmemiş) NaN kalır
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0) if n_boot else np.full(len(keys), np.nan)
        hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0) if n_boot else np.full(len(keys), np.nan)
    return {
        "metrics": {k: {"point": point[k], "low": float(lo[i]), "high": float(hi[i])} for i, k in enumerate(keys)},
        "n_clusters": int(n_clusters),
        "n_records": int(n),
        "n_boot": int(n_boot),
        "alpha": alpha,
        "seed": seed,
    }


# --------------------------------------------------------------------------------------
# Gecikme, verim, bellek
# --------------------------------------------------------------------------------------


def measure_single_record_latency(
    predict_fn: Callable, rows: Sequence, *, n: int = 500, warmup: int = 50, seed: int = 0
) -> dict:
    """predict_fn(rows[i]) çağrısını tek tek zamanlar. rows: her eleman predict_fn'in girdisi."""
    if len(rows) == 0:
        raise ValueError("Gecikme ölçümü için en az bir kayıt gerekir.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=n + warmup)
    times = np.empty(n)
    for j, i in enumerate(idx):
        t0 = time.perf_counter_ns()
        predict_fn(rows[i])
        dt = time.perf_counter_ns() - t0
        if j >= warmup:
            times[j - warmup] = dt / 1e6
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "max_ms": float(times.max()),
        "n": int(n),
        "warmup": int(warmup),
    }


def measure_throughput(predict_batch_fn: Callable, X, *, batch_size: int = 1000, repeats: int = 3) -> dict:
    """Toplu tahmin verimi (akış/sn): X'in tamamı batch_size'lık parçalarla, en iyi tekrar."""
    n = len(X)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for s in range(0, n, batch_size):
            predict_batch_fn(X[s : s + batch_size])
        best = min(best, time.perf_counter() - t0)
    return {"flows_per_s": float(n / best) if best > 0 else float("inf"), "n": int(n), "batch_size": int(batch_size)}


def current_rss_mb() -> float:
    import psutil

    return float(psutil.Process().memory_info().rss) / 1e6


def peak_rss_mb() -> float:
    """Sürecin şimdiye kadarki tepe bellek kullanımı (MB). Süreç geneli üst sınırdır, modele özgü değildir."""
    if resource is not None:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # macOS bayt, Linux KB
    try:  # Windows
        import psutil

        return float(psutil.Process().memory_info().peak_wset) / 1e6
    except Exception:  # pragma: no cover
        return float("nan")


@contextmanager
def timed_resources() -> Iterator[dict]:
    """with bloğunun süresini ve bellek değişimini yazar. Sonuç sözlüğü blok bitince dolar."""
    info: dict = {}
    rss0 = current_rss_mb()
    t0 = time.perf_counter()
    try:
        yield info
    finally:
        info["seconds"] = time.perf_counter() - t0
        info["rss_before_mb"] = rss0
        info["rss_after_mb"] = current_rss_mb()
        info["rss_delta_mb"] = info["rss_after_mb"] - rss0
        info["process_peak_rss_mb"] = peak_rss_mb()


def pickled_size_mb(obj) -> float:
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) / 1e6
