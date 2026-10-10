"""Rapor ve notebook grafikleri (matplotlib, açık zemin).

Renkler: kategorik ilk üç yuva (mavi, turuncu, su yeşili) renk körlüğü denetiminden geçti (tüm çiftler).
Su yeşili açık zeminde 3:1 kontrastın altında; bu yüzden her grafiğin yanında sayı tablosu verilir.
Tek eksen kuralı: farklı ölçekteki iki metrik aynı grafikte iki y-ekseniyle gösterilmez.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # mavi, turuncu, su yeşili
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_RAMP = ("#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

MODEL_LABELS = {
    "dummy": "Dummy",
    "logreg": "Logistic Reg.",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "iforest": "Isolation Forest",
}
FEATURE_SET_LABELS = {"full": "77 özellik", "llm_subset": "LLM kümesi (20)"}


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _style(ax, *, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def _figure(w: float, h: float):
    plt = _plt()
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    return fig


def _save(fig, path) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)


def plot_class_distribution(table: pd.DataFrame, *, title: str = "", path=None):
    """table: index=sınıf, sütunlar=bölmeler (train/val/test). Logaritmik eksen: 17 ile 67.000 arası sınıflar."""
    cols = [c for c in ("train", "val", "test") if c in table.columns]
    t = table[cols].copy()
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig = _figure(7.5, 0.55 * len(t) + 1.2)
    ax = fig.add_subplot(111)
    h = 0.8 / len(cols)
    y = np.arange(len(t))
    zeros = []
    for i, c in enumerate(cols):
        vals = t[c].to_numpy(dtype=float)
        pos = y + 0.4 - h * (i + 0.5)
        ax.barh(pos[vals > 0], vals[vals > 0], height=h * 0.9, color=SERIES[i], label=c, edgecolor=SURFACE, linewidth=1)
        zeros += [(p, c) for p, v in zip(pos, vals) if v == 0]
    ax.set_xscale("log")
    left = ax.get_xlim()[0]
    for p, c in zeros:
        ax.text(left * 1.15, p, f"{c}: 0", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, t.index)
    ax.set_xlabel("kayıt sayısı (log ölçek, tekrarlar indirgenmiş)")
    ax.set_title(title, loc="left", fontsize=11)
    _style(ax, grid_axis="x")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    _save(fig, path)
    return fig


def plot_confusion(counts: dict, *, title: str = "", ax=None, path=None):
    """counts: {'tn','fp','fn','tp'}. Hücrede sayı ve satır yüzdesi (gerçek sınıfa göre) yazar."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    m = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=float)
    row = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    own = ax is None
    if own:
        fig = _figure(3.4, 3.0)
        ax = fig.add_subplot(111)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    ax.imshow(row, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j]):,}\n%{100 * row[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if row[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], ["Normal", "Saldırı"])
    ax.set_yticks([0, 1], ["Normal", "Saldırı"])
    ax.set_xlabel("tahmin")
    ax.set_ylabel("gerçek")
    ax.set_title(title, fontsize=9, loc="left")
    ax.tick_params(colors=INK_2, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    if own:
        _save(ax.figure, path)
    return ax.figure


def plot_metric_ci(table: pd.DataFrame, metric: str, *, title: str = "", xlabel: str = "", path=None, xlim=None,
                   log: bool = False, log_floor: float = 1e-5):
    """Birincil eşik kuralı satırları: model başına birincil tohumun değeri ve küme bootstrap %95 GA'sı.

    Renk = özellik kümesi. Nokta ve çubuk AYNI tohumdan (birincil) gelir; tohumlar arası ortalama ve sapma tablodadır.
    """
    t = table[table["birincil_kural"]].copy()
    models = [m for m in MODEL_LABELS if m in set(t["model"])]
    fsets = [f for f in FEATURE_SET_LABELS if f in set(t["özellik_kümesi"])]
    fig = _figure(6.8, 0.6 * len(models) + 1.3)
    ax = fig.add_subplot(111)
    clipped: list[str] = []
    for i, fs in enumerate(fsets):
        sub = t[t["özellik_kümesi"] == fs].set_index("model")
        for k, m in enumerate(models):
            if m not in sub.index:
                continue
            r = sub.loc[m]
            yk = k + (i - (len(fsets) - 1) / 2) * 0.22
            lo, hi = r.get(f"{metric}_ga_alt", np.nan), r.get(f"{metric}_ga_üst", np.nan)
            if log:  # log eksende 0 çizilemez: taban değere kırpılır ve not düşülür
                lo, hi = max(lo, log_floor) if np.isfinite(lo) else lo, max(hi, log_floor) if np.isfinite(hi) else hi
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [yk, yk], color=SERIES[i], linewidth=2, solid_capstyle="round")
            point = r.get(f"{metric}_ga_nokta", np.nan)
            point = r[metric] if not np.isfinite(point) else point
            if log and point <= log_floor:
                clipped.append(m)
                point = log_floor
            ax.plot([point], [yk], "o", color=SERIES[i], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=FEATURE_SET_LABELS[fs] if k == 0 or m == models[0] else None)
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    ax.invert_yaxis()
    if log:
        from matplotlib.ticker import FuncFormatter

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel((xlabel or metric) + (" (log ölçek)" if log else ""))
    if clipped:
        ax.text(0.0, -0.22, f"0 olan değerler {log_floor:g} olarak çizildi: {', '.join(sorted(set(clipped)))}",
                transform=ax.transAxes, fontsize=7, color=MUTED)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontsize=11)
    _style(ax, grid_axis="x")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              labelcolor=INK_2)
    _save(fig, path)
    return fig


def _place_labels(ax, points, labels) -> None:
    """Etiketleri birbirine ve noktalara çarpmayacak ilk konuma yerleştirir (basit açgözlü yerleşim)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken = []
    for x, y in points:
        cx, cy = ax.transData.transform((x, y))
        from matplotlib.transforms import Bbox

        taken.append(Bbox.from_bounds(cx - 6, cy - 6, 12, 12))
    offsets = [(8, 3), (8, -11), (-8, 3), (-8, -11), (8, 13), (8, -21), (-8, 13), (-8, -21), (8, 23), (-8, 23)]
    for (x, y), lab in zip(points, labels):
        chosen = None
        for dx, dy in offsets:
            t = ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7.5, color=INK_2,
                            ha="left" if dx > 0 else "right")
            bb = t.get_window_extent(renderer).expanded(1.04, 1.1)
            if not any(bb.overlaps(o) for o in taken):
                chosen = t
                taken.append(bb)
                break
            t.remove()
        if chosen is None:  # yer bulunamazsa ilk konuma yaz
            t = ax.annotate(lab, (x, y), textcoords="offset points", xytext=offsets[0], fontsize=7.5, color=INK_2)
            taken.append(t.get_window_extent(renderer))


def plot_success_latency(table: pd.DataFrame, *, metric: str = "macro_f1", title: str = "", path=None):
    """x: tek kayıt gecikmesi p95 (ms, log ölçek), y: başarı metriği (varsayılan macro-F1). Renk = özellik kümesi.

    Varsayılan macro-F1'dir: saldırı F1'i saldırı oranına çok bağlıdır (saldırının %80 olduğu testte
    her şeye 'saldırı' diyen Dummy bile yüksek saldırı F1'i alır); macro-F1 iki sınıfı eşit tartar.
    """
    t = table[table["birincil_kural"]].copy()
    fsets = [f for f in FEATURE_SET_LABELS if f in set(t["özellik_kümesi"])]
    fig = _figure(7.4, 4.2)
    ax = fig.add_subplot(111)
    points, labels = [], []
    for i, fs in enumerate(fsets):
        sub = t[t["özellik_kümesi"] == fs]
        ax.scatter(sub["gecikme_p95_ms"], sub[metric], s=60, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5,
                   label=FEATURE_SET_LABELS[fs], zorder=3)
        for _, r in sub.iterrows():
            points.append((r["gecikme_p95_ms"], r[metric]))
            labels.append(MODEL_LABELS.get(r["model"], r["model"]))
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    lo, hi = ax.get_xlim()
    ticks = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo <= v <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("tek kayıt tahmin gecikmesi p95 (ms, log ölçek; yalnızca model, 1 iş parçacığı)")
    names = {"macro_f1": "macro-F1", "attack_f1": "saldırı F1"}
    ax.set_ylabel(f"{names.get(metric, metric)} (test, tohum ortalaması)")
    ax.set_title(title, loc="left", fontsize=11)
    _style(ax, grid_axis="both")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), labelcolor=INK_2)
    _place_labels(ax, points, labels)
    _save(fig, path)
    return fig


def plot_pr_curve(y_true, score, threshold=None, *, title: str = "", path=None):
    from sklearn.metrics import precision_recall_curve

    p, r, thr = precision_recall_curve(np.asarray(y_true), np.asarray(score))
    fig = _figure(4.6, 3.8)
    ax = fig.add_subplot(111)
    ax.plot(r, p, color=SERIES[0], linewidth=2)
    if threshold is not None and len(thr):
        k = int(np.argmin(np.abs(thr - threshold)))
        ax.plot([r[k]], [p[k]], "o", color=SERIES[1], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"seçilen eşik = {threshold:.3g}")
        ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK_2)
    ax.set_xlabel("recall (yakalama)")
    ax.set_ylabel("precision (kesinlik)")
    ax.set_title(title, loc="left", fontsize=10)
    _style(ax, grid_axis="both")
    _save(fig, path)
    return fig


def plot_score_hist(score, y_true, thresholds: dict | None = None, *, title: str = "", path=None):
    """Normal ve saldırı skor dağılımları (yoğunluk) ve eşik çizgileri."""
    s, y = np.asarray(score), np.asarray(y_true)
    fig = _figure(6.4, 3.6)
    ax = fig.add_subplot(111)
    bins = np.linspace(np.nanmin(s), np.nanmax(s), 60)
    for i, (cls, name) in enumerate(((0, "Normal"), (1, "Saldırı"))):
        ax.hist(s[y == cls], bins=bins, density=True, histtype="step", linewidth=2, color=SERIES[i], label=name)
    for k, (name, thr) in enumerate((thresholds or {}).items()):
        ax.axvline(thr, color=INK_2 if k == 0 else MUTED, linestyle="--" if k == 0 else ":", linewidth=1.2)
        ax.text(thr, ax.get_ylim()[1] * (0.95 - 0.1 * k), f" {name}", fontsize=7.5, color=INK_2, va="top")
    ax.set_xlabel("anomali skoru (büyük = daha anormal)")
    ax.set_ylabel("yoğunluk")
    ax.set_title(title, loc="left", fontsize=10)
    _style(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    _save(fig, path)
    return fig
