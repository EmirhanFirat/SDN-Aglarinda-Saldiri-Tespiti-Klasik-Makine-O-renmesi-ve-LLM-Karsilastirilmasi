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


def _basename(name: str) -> str:
    return str(name).replace("\\", "/").rsplit("/", 1)[-1]


def _fragment_links(
    df: pd.DataFrame, flow_id_col: str, source_col: str, time_col: str, gap_seconds: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Aynı bağlantının ardışık parçalarını bağlayan (önceki_satır, sonraki_satır) çiftlerini döndürür.

    Parça tanımı: aynı kaynak dosya + aynı flow_id + başlangıçları en çok `gap_seconds` arayla.
    Neden zaman sınırı? CICFlowMeter uzun bağlantıyı zaman aşımıyla parçalar; parçalar birbirine
    yakındır. Haftalar sonra aynı beşliyle (ör. periyodik DNS) görülen akış YENİ bir bağlantıdır;
    onları birleştirmek tüm bir trafik sınıfını tek dev gruba zincirler (gerçek InSDN'de görüldü).
    Zaman damgası yoksa (veya gap_seconds=None) aynı flow_id her zaman aynı parça sayılır.
    """
    n = len(df)
    ids = df[flow_id_col]
    key = (df[source_col].astype(str) + "|" + ids.astype(str)) if source_col in df.columns else ids.astype(str)
    key = key.where(ids.notna())
    use_time = gap_seconds is not None and time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col])
    secs = (df[time_col] - df[time_col].min()).dt.total_seconds() if use_time else pd.Series(np.zeros(n))
    tmp = pd.DataFrame({"k": key.to_numpy(), "t": secs.to_numpy(), "i": np.arange(n)}).dropna(subset=["k"])
    tmp = tmp.sort_values(["k", "t", "i"], kind="stable")
    same_key = tmp["k"].eq(tmp["k"].shift())
    if use_time:
        gap = tmp["t"] - tmp["t"].shift()
        # Zaman damgası eksikse aralık bilinemez: muhafazakâr davran, bağla.
        close = (gap <= float(gap_seconds)) | gap.isna() | tmp["t"].isna() | tmp["t"].shift().isna()
        link = same_key & close
    else:
        link = same_key
    prev_i = tmp["i"].shift().to_numpy()[link.to_numpy()].astype(np.int64)
    cur_i = tmp["i"].to_numpy()[link.to_numpy()]
    return prev_i, cur_i


def build_groups(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    flow_id_col: str = "flow_id",
    source_col: str = "source_file",
    time_col: str = "timestamp",
    time_block_seconds: float | None = None,
    time_block_files: Sequence[str] | None = None,
    fragment_gap_seconds: float | None = 300.0,
    use_flow_id: bool = True,
    use_duplicates: bool = True,
) -> tuple[np.ndarray, dict]:
    """Her kayıt için grup kodu (0..G-1) ve bilgi sözlüğü döndürür.

    flow_id kuralı: aynı kaynak dosya + aynı flow_id + en çok `fragment_gap_seconds` arayla
    başlayan akışlar aynı bağlantının parçası sayılır (bkz. _fragment_links).
    time_block_seconds verilirse zaman bloğu gruplaması yapılır; time_block_files ile yalnızca
    belirtilen dosyalara uygulanabilir (dosya adının son bileşeniyle eşleşir).
    NOT: Gerçek InSDN'de saldırılar çok kısa kayıtlardır; zaman bloğu + tekrar kuralı birlikte
    dev gruplar üretir (bkz. docs/veri_sozlugu.md). Varsayılan bu yüzden kapalıdır.
    """
    n = len(df)
    uf = _UnionFind(n)
    info: dict = {"keys_used": [], "warnings": []}

    if use_flow_id:
        if flow_id_col in df.columns:
            prev_i, cur_i = _fragment_links(df, flow_id_col, source_col, time_col, fragment_gap_seconds)
            for a, b in zip(prev_i.tolist(), cur_i.tolist()):
                uf.union(a, b)
            info["keys_used"].append(
                f"flow_id_fragments_gap_{int(fragment_gap_seconds)}s" if fragment_gap_seconds else "flow_id"
            )
        else:
            info["warnings"].append(f"'{flow_id_col}' sütunu yok; bağlantı parçaları gruplanamadı.")
    if use_duplicates:
        _union_by_key(uf, _row_hash(df, feature_cols))
        info["keys_used"].append("duplicate_feature_vector")
    if time_block_seconds:
        if time_col in df.columns and source_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            t = df[time_col]
            valid = t.notna()
            if time_block_files is not None:
                wanted = {_basename(f) for f in time_block_files}
                valid = valid & df[source_col].map(_basename).isin(wanted)
            if valid.any():
                secs = (t - t.min()).dt.total_seconds()
                block = np.floor(secs / float(time_block_seconds)).astype("Int64").astype(str)
                key = (df[source_col].astype(str) + "|" + block).where(valid)
                _union_by_key(uf, key)
                info["keys_used"].append(f"time_block_{int(time_block_seconds)}s")
            if time_block_files is None and (~valid).any():
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
    time_block_seconds: float | None = None,
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


def _report_for_assignment(
    df: pd.DataFrame,
    codes: np.ndarray,
    split_arr: np.ndarray,
    names: Sequence[str],
    ginfo: dict,
    *,
    label_col: str,
    min_test_support: int,
    extra: dict,
) -> dict:
    """Sınıf x bölme tablosunu ve uyarıları üretir (oran hedefi olmayan tasarımlar için)."""
    names = list(names)
    counts = pd.crosstab(df[label_col].to_numpy(), split_arr).reindex(columns=names, fill_value=0)
    rows = {s: int((split_arr == s).sum()) for s in names}
    groups_by_split = pd.Series(codes).groupby(split_arr).nunique().to_dict()
    groups_per_class = (pd.crosstab(codes, df[label_col].to_numpy()) > 0).sum(axis=0)
    test_name, train_name = names[-1], names[0]
    warnings = list(ginfo["warnings"])
    if ginfo["largest_group_share"] > 0.10:
        warnings.append(
            f"En büyük grup kayıtların %{ginfo['largest_group_share'] * 100:.1f}'ini içeriyor; "
            "bölme beklenen dağılımı vermeyebilir."
        )
    for cls in counts.index:
        n_test, n_train = int(counts.loc[cls, test_name]), int(counts.loc[cls, train_name])
        if n_train == 0:
            warnings.append(f"Sınıf '{cls}' eğitim bölmesinde yok; model bu sınıfı hiç görmeyecek.")
        if n_test == 0:
            warnings.append(f"Sınıf '{cls}' {test_name} bölmesinde yok; bu sınıf için bu tasarımda ölçüm yapılamaz.")
        elif n_test < min_test_support:
            warnings.append(
                f"Sınıf '{cls}' {test_name} bölmesinde {n_test} kayıt (< {min_test_support}): "
                "güvenilir ölçüm iddia edilemez."
            )
    return {
        "rows_by_split": rows,
        "groups_by_split": {s: int(groups_by_split.get(s, 0)) for s in names},
        "groups_per_class": {str(c): int(v) for c, v in groups_per_class.items()},
        "class_counts_by_split": {str(c): {s: int(counts.loc[c, s]) for s in names} for c in counts.index},
        "group_info": ginfo,
        "warnings": warnings,
        **extra,
    }


def make_domain_holdout_split(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_files: Sequence[str],
    shared_files: Sequence[str],
    shared_target_fraction: float = 0.2,
    trainval_ratios: Sequence[float] = (0.75, 0.25),
    names: Sequence[str] = ("train", "val", "test"),
    seed: int = 42,
    label_col: str = "label",
    record_col: str = "record_id",
    source_col: str = "source_file",
    time_col: str = "timestamp",
    time_block_seconds: float = 300.0,
    min_test_support: int = 30,
) -> SplitResult:
    """Koşular arası (senaryo bazlı) bölme.

    * target_files: tamamı TEST olur (ör. farklı kurban makinede alınmış yakalama).
    * shared_files: tek dosyada bulunan sınıflar için (ör. Normal_data.csv). Zaman bloklarına
      ayrılır; kronolojik olarak son `shared_target_fraction` kısmı TEST olur, öncesi kaynak alana gider.
    * Kalan her şey kaynak alandır; grup bütünlüğüyle train/val'e `trainval_ratios` oranında bölünür.
    Hiçbir grup birden fazla bölmeye dağılmaz. Test'te bulunmayan sınıflar raporda uyarı olarak çıkar.
    """
    names = list(names)
    if len(names) != 3 or len(trainval_ratios) != 2:
        raise ValueError("names üç, trainval_ratios iki elemanlı olmalı.")
    base = df[source_col].map(_basename).to_numpy()
    tgt = {_basename(f) for f in target_files}
    shr = {_basename(f) for f in shared_files}
    if tgt & shr:
        raise ValueError("Bir dosya hem target hem shared olamaz.")
    missing = (tgt | shr) - set(base)
    if missing:
        raise ValueError(f"Veride bulunmayan dosya(lar): {sorted(missing)}")

    codes, ginfo = build_groups(
        df,
        feature_cols,
        source_col=source_col,
        time_col=time_col,
        use_duplicates=False,
        time_block_seconds=time_block_seconds,
        time_block_files=sorted(shr),
    )
    G = int(codes.max()) + 1
    if (pd.Series(base).groupby(codes).nunique() > 1).any():
        raise RuntimeError("Bir grup birden fazla kaynak dosyadan kayıt içeriyor; gruplama kuralı hatalı.")
    group_file = pd.Series(base).groupby(codes).first().reindex(range(G)).to_numpy()
    t = df[time_col]
    secs = (t - t.min()).dt.total_seconds().to_numpy()
    gtime = pd.Series(secs).groupby(codes).min().reindex(range(G)).to_numpy()
    gsize = np.bincount(codes, minlength=G)

    TRAIN, VAL, TEST = 0, 1, 2
    gsplit = np.full(G, -1, dtype=np.int64)
    gsplit[np.isin(group_file, sorted(tgt))] = TEST
    for f in sorted(shr):
        gs = np.where(group_file == f)[0]
        if np.isnan(gtime[gs]).any():
            raise ValueError(f"{f}: zaman damgası olmayan grup var; kronolojik ayırma yapılamaz.")
        gs = gs[np.argsort(gtime[gs], kind="stable")]
        mid = (np.cumsum(gsize[gs]) - gsize[gs] / 2.0) / gsize[gs].sum()
        gsplit[gs[mid > 1.0 - shared_target_fraction]] = TEST

    src_groups = np.where(gsplit == -1)[0]
    ct = pd.crosstab(codes, df[label_col].to_numpy()).reindex(range(G), fill_value=0)
    gsplit[src_groups] = assign_groups(ct.to_numpy()[src_groups], trainval_ratios, seed=seed, strategy="random")

    split_arr = np.asarray(names, dtype=object)[gsplit[codes]]
    assignment = pd.Series(split_arr, index=df[record_col].to_numpy(), name="split")
    report = _report_for_assignment(
        df,
        codes,
        split_arr,
        names,
        ginfo,
        label_col=label_col,
        min_test_support=min_test_support,
        extra={
            "design": "cross_session",
            "seed": seed,
            "target_files": sorted(tgt),
            "shared_files": sorted(shr),
            "shared_target_fraction": shared_target_fraction,
            "trainval_ratios": list(map(float, trainval_ratios)),
            "time_block_seconds": time_block_seconds,
        },
    )
    return SplitResult(record_ids=df[record_col].tolist(), group_ids=codes, assignment=assignment, report=report)


def verify_split_integrity(
    df: pd.DataFrame,
    group_ids: np.ndarray,
    assignment: pd.Series,
    feature_cols: Sequence[str],
    *,
    record_col: str = "record_id",
    flow_id_col: str = "flow_id",
    fragment_gap_seconds: float | None = 300.0,
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
        # Bağımsız yeniden türetme: aynı bağlantının ardışık parçaları aynı bölmede olmalı.
        prev_i, cur_i = _fragment_links(df, flow_id_col, "source_file", "timestamp", fragment_gap_seconds)
        bad = int((split[prev_i] != split[cur_i]).sum())
        if bad:
            problems.append(f"{bad} bağlantı parçası çifti farklı bölmelere dağılmış.")
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
