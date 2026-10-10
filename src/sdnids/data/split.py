"""Grup bütünlüğünü koruyan eğitim/doğrulama/test bölmesi.

Neden grup? Aynı bağlantının parçaları, aynı trafik koşusundaki akışlar ve birebir
tekrar eden kayıtlar birbirine çok benzer. Bunlar eğitim ve teste dağılırsa model
"ezberlediği" kayıtları testte tekrar görür ve başarı olduğundan yüksek çıkar.

Gruplama kuralları (birleştirme: union-find):
  1. Aynı flow_id (CICFlowMeter uzun bağlantıyı zaman aşımıyla parçalar).
  2. Birebir aynı özellik vektörü.
  3. Aynı kaynak dosyada, aynı zaman bloğundaki kayıtlar (senaryo/koşu yaklaşımı).

Güvenilir senaryo kimliği yoksa (3) yalnızca bir yaklaşımdır; bu sınırlılık raporda
belirtilmelidir. Grup bütünlüğü oranlardan önceliklidir: oranlar tutmazsa uyarı
verilir, grup bölünmez.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _union_by_key(uf: _UnionFind, keys) -> None:
    """Aynı anahtara sahip kayıtları birleştirir; boş (NaN) anahtarlar atlanır."""
    codes, _ = pd.factorize(pd.Series(keys))
    first: dict[int, int] = {}
    for i, c in enumerate(codes.tolist()):
        if c < 0:
            continue
        j = first.setdefault(c, i)
        if j != i:
            uf.union(j, i)


def _row_hash(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(df[list(feature_cols)], index=False).to_numpy()


def build_groups(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    flow_id_col: str = "flow_id",
    source_col: str = "source_file",
    time_col: str = "timestamp",
    time_block_seconds: float | None = 300.0,
    use_flow_id: bool = True,
    use_duplicates: bool = True,
) -> tuple[np.ndarray, dict]:
    """Her kayıt için grup kodu (0..G-1) ve bilgi sözlüğü döndürür."""
    n = len(df)
    uf = _UnionFind(n)
    info: dict = {"keys_used": [], "warnings": []}

    if use_flow_id:
        if flow_id_col in df.columns:
            _union_by_key(uf, df[flow_id_col])
            info["keys_used"].append("flow_id")
        else:
            info["warnings"].append(f"'{flow_id_col}' sütunu yok; bağlantı parçaları gruplanamadı.")
    if use_duplicates:
        _union_by_key(uf, _row_hash(df, feature_cols))
        info["keys_used"].append("duplicate_feature_vector")
    if time_block_seconds:
        if time_col in df.columns and source_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            t = df[time_col]
            valid = t.notna()
            if valid.any():
                secs = (t - t.min()).dt.total_seconds()
                block = np.floor(secs / float(time_block_seconds)).astype("Int64").astype(str)
                key = (df[source_col].astype(str) + "|" + block).where(valid)
                _union_by_key(uf, key)
                info["keys_used"].append(f"time_block_{int(time_block_seconds)}s")
            if (~valid).any():
                info["warnings"].append(
                    f"{int((~valid).sum())} kayıtta zaman damgası yok; zaman bloğuyla gruplanamadı."
                )
        else:
            info["warnings"].append("Zaman damgası/kaynak dosya yok; zaman bloğu gruplaması yapılamadı.")

    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
    codes, _ = pd.factorize(roots)
    sizes = np.bincount(codes)
    info["n_groups"] = int(len(sizes))
    info["n_singleton_groups"] = int((sizes == 1).sum())
    info["largest_group_share"] = float(sizes.max() / n) if n else 0.0
    info["group_size_p50"] = float(np.percentile(sizes, 50))
    info["group_size_p95"] = float(np.percentile(sizes, 95))
    if info["n_groups"] == n:
        info["warnings"].append(
            "Hiçbir kayıt birleştirilemedi: güvenilir gruplama bilgisi yok. Bu bir sınırlılıktır; "
            "aynı oturumun parçaları eğitim ve teste dağılmış olabilir."
        )
    return codes.astype(np.int64), info


def assign_groups(
    C: np.ndarray,
    ratios: Sequence[float],
    *,
    seed: int = 42,
    strategy: str = "random",
    group_times: np.ndarray | None = None,
) -> np.ndarray:
    """Grupları bölmelere atar. C: (grup x sınıf) kayıt sayısı matrisi."""
    ratios_arr = np.asarray(ratios, dtype=float)
    if not np.isclose(ratios_arr.sum(), 1.0):
        raise ValueError("Oranların toplamı 1 olmalı.")
    G, K = C.shape
    S = len(ratios_arr)
    sizes = C.sum(axis=1)
    assign = np.full(G, -1, dtype=np.int64)

    if strategy == "random":
        rng = np.random.default_rng(seed)
        class_tot = C.sum(axis=0).astype(float)
        targets = np.outer(ratios_arr, class_tot)
        cur = np.zeros((S, K))
        perm = rng.permutation(G)
        order = perm[np.argsort(-sizes[perm], kind="stable")]  # büyük gruplar önce, eşitlikte rastgele
        weight = 1.0 / np.maximum(class_tot, 1.0)
        for g in order:
            c = C[g].astype(float)
            deficit = ((targets - cur) * (c * weight)).sum(axis=1)
            s = int(np.argmax(deficit))
            assign[g] = s
            cur[s] += c
        return assign

    if strategy == "chronological":
        if group_times is None or np.isnan(group_times).any():
            raise ValueError("chronological strateji için tüm gruplarda zaman damgası gerekli.")
        dominant = C.argmax(axis=1)
        bounds = np.cumsum(ratios_arr)
        for k in range(K):
            gs = np.where(dominant == k)[0]
            if len(gs) == 0:
                continue
            gs = gs[np.argsort(group_times[gs], kind="stable")]
            total = float(sizes[gs].sum())
            mid = (np.cumsum(sizes[gs]) - sizes[gs] / 2.0) / total
            s = np.minimum(np.searchsorted(bounds, mid, side="right"), S - 1)
            assign[gs] = s
        return assign

    raise ValueError(f"Bilinmeyen strateji: {strategy}")


@dataclass
class SplitResult:
    record_ids: list
    group_ids: np.ndarray
    assignment: pd.Series  # indeks: record_id, değer: bölme adı
    report: dict


def make_split(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    label_col: str = "label",
    record_col: str = "record_id",
    time_col: str = "timestamp",
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
    names: Sequence[str] = ("train", "val", "test"),
    seed: int = 42,
    strategy: str = "random",
    time_block_seconds: float | None = 300.0,
    min_test_support: int = 30,
    group_kwargs: dict | None = None,
) -> SplitResult:
    names = list(names)
    codes, ginfo = build_groups(
        df, feature_cols, time_col=time_col, time_block_seconds=time_block_seconds, **(group_kwargs or {})
    )
    G = int(codes.max()) + 1
    ct = pd.crosstab(codes, df[label_col].to_numpy()).reindex(range(G), fill_value=0)
    C = ct.to_numpy()
    classes = [str(c) for c in ct.columns]

    group_times = None
    if strategy == "chronological":
        t = df[time_col]
        secs = (t - t.min()).dt.total_seconds().to_numpy()
        group_times = pd.Series(secs).groupby(codes).min().reindex(range(G)).to_numpy()

    gassign = assign_groups(C, ratios, seed=seed, strategy=strategy, group_times=group_times)
    split_arr = np.asarray(names, dtype=object)[gassign[codes]]
    assignment = pd.Series(split_arr, index=df[record_col].to_numpy(), name="split")

    counts = pd.crosstab(df[label_col].to_numpy(), split_arr).reindex(columns=names, fill_value=0)
    rows_by_split = {s: int((split_arr == s).sum()) for s in names}
    n = len(df)
    achieved = {s: rows_by_split[s] / n for s in names}
    warnings = list(ginfo["warnings"])
    for s, target in zip(names, ratios):
        if abs(achieved[s] - target) > 0.05:
            warnings.append(
                f"'{s}' oranı hedeften sapıyor: hedef {target:.2f}, gerçekleşen {achieved[s]:.3f} "
                "(grup bütünlüğü oranlardan önceliklidir)."
            )
    if ginfo["largest_group_share"] > min(ratios):
        warnings.append(
            f"En büyük grup kayıtların %{ginfo['largest_group_share'] * 100:.1f}'ini içeriyor; "
            "bölme oranları ve nadir sınıflar etkilenebilir."
        )
    groups_per_class = (C > 0).sum(axis=0)
    test_name = names[-1]
    for cls, gcount in zip(classes, groups_per_class):
        if gcount < 3:
            warnings.append(f"Sınıf '{cls}' yalnızca {int(gcount)} grupta var; üç bölmeye ayrı grup konamaz.")
        test_n = int(counts.loc[cls, test_name]) if cls in counts.index else 0
        if test_n < min_test_support:
            warnings.append(
                f"Sınıf '{cls}' {test_name} kümesinde {test_n} kayıt (< {min_test_support}): "
                "bu sınıf için güvenilir ölçüm iddia edilemez."
            )
        for s in names:
            if int(counts.loc[cls, s]) == 0:
                warnings.append(f"Sınıf '{cls}' '{s}' bölmesinde hiç yok.")

    report = {
        "strategy": strategy,
        "seed": seed,
        "ratios_target": dict(zip(names, map(float, ratios))),
        "ratios_achieved": achieved,
        "rows_by_split": rows_by_split,
        "groups_by_split": {s: int((gassign == i).sum()) for i, s in enumerate(names)},
        "class_counts_by_split": {str(c): {s: int(counts.loc[c, s]) for s in names} for c in counts.index},
        "group_info": ginfo,
        "warnings": warnings,
    }
    return SplitResult(
        record_ids=df[record_col].tolist(), group_ids=codes, assignment=assignment, report=report
    )


def verify_split_integrity(
    df: pd.DataFrame,
    group_ids: np.ndarray,
    assignment: pd.Series,
    feature_cols: Sequence[str],
    *,
    record_col: str = "record_id",
    flow_id_col: str = "flow_id",
) -> list[str]:
    """Sızıntı ihlallerini metin listesi olarak döndürür; boş liste = ihlal yok."""
    problems: list[str] = []
    split = assignment.reindex(df[record_col].to_numpy()).to_numpy()
    if pd.isna(split).any():
        problems.append("Bölmeye atanmamış kayıt var.")
        return problems
    if assignment.index.duplicated().any():
        problems.append("Bir kayıt birden fazla bölmeye atanmış (yinelenen record_id).")

    per_group = pd.DataFrame({"g": group_ids, "s": split}).groupby("g")["s"].nunique()
    bad = int((per_group > 1).sum())
    if bad:
        problems.append(f"{bad} grup birden fazla bölmeye dağılmış.")

    per_hash = pd.DataFrame({"h": _row_hash(df, feature_cols), "s": split}).groupby("h")["s"].nunique()
    bad = int((per_hash > 1).sum())
    if bad:
        problems.append(f"{bad} birebir aynı özellik vektörü birden fazla bölmede.")

    if flow_id_col in df.columns:
        fid = pd.DataFrame({"f": df[flow_id_col].to_numpy(), "s": split}).dropna(subset=["f"])
        bad = int((fid.groupby("f")["s"].nunique() > 1).sum())
        if bad:
            problems.append(f"{bad} flow_id birden fazla bölmede.")
    return problems


def _ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def save_split(result: SplitResult, out_dir: str | Path, params: dict) -> None:
    """Bölmeyi diske yazar. Test kayıtları SHA-256 ile sabitlenir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {"record_id": result.record_ids, "group_id": result.group_ids, "split": result.assignment.to_numpy()}
    )
    table.to_csv(out / "split_assignment.csv", index=False)
    digests = {s: _ids_digest(g["record_id"]) for s, g in table.groupby("split")}
    meta = {"params": params, "report": result.report, "record_id_sha256_by_split": digests}
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_split(out_dir: str | Path) -> tuple[pd.Series, np.ndarray, dict]:
    """Kaydedilmiş bölmeyi okur ve özetleri doğrular; değiştirilmişse hata verir."""
    out = Path(out_dir)
    table = pd.read_csv(out / "split_assignment.csv", dtype={"record_id": str})
    meta = json.loads((out / "split_meta.json").read_text(encoding="utf-8"))
    for s, expected in meta["record_id_sha256_by_split"].items():
        got = _ids_digest(table.loc[table["split"] == s, "record_id"])
        if got != expected:
            raise ValueError(f"'{s}' bölmesinin kayıt kimlikleri değiştirilmiş (SHA-256 uyuşmuyor).")
    assignment = pd.Series(table["split"].to_numpy(), index=table["record_id"].to_numpy(), name="split")
    return assignment, table["group_id"].to_numpy(), meta
