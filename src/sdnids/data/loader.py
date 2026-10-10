"""InSDN CSV dosyalarını okur ve tek bir tablo biçimine getirir.

ÖNEMLİ: InSDN dosyası bu oturumda indirilemedi (çıkış politikası). Sütun adları
CICFlowMeter biçimi varsayılarak yazıldı ('Flow ID', 'Src IP', 'Label' ...).
Gerçek dosya gelince sütun listesi docs/veri_sozlugu.md içine kaydedilip
doğrulanmalıdır. Etiket sütunu bulunamazsa yükleyici açık bir hata verir.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from ..schema import LABEL_COLUMN, normalize_column_name
from .labels import canonicalize_labels

# Sayısal dönüştürmeden muaf tutulan metin sütunları.
TEXT_COLUMNS = frozenset(
    {"flow_id", "src_ip", "dst_ip", "timestamp", "label", "label_raw", "source_file", "record_id"}
)

_NUMERIC_TOKENS = {
    "Infinity": "inf",
    "-Infinity": "-inf",
    "infinity": "inf",
    "-infinity": "-inf",
    "NaN": "nan",
    "": "nan",
}


def _coerce_numeric(df: pd.DataFrame) -> dict[str, int]:
    """Metin olarak okunmuş sayısal sütunları sayıya çevirir.

    Çevrilemeyen (ve NaN/Infinity simgesi olmayan) değerlerin sayısını döndürür;
    böylece sessiz veri kaybı denetlenebilir.
    """
    lost: dict[str, int] = {}
    for col in df.columns:
        if col in TEXT_COLUMNS or pd.api.types.is_numeric_dtype(df[col]):
            continue
        original = df[col]
        cleaned = original.astype(str).str.strip().replace(_NUMERIC_TOKENS)
        converted = pd.to_numeric(cleaned, errors="coerce")
        bad = int((converted.isna() & ~cleaned.isin(["nan", "None"]) & original.notna()).sum())
        if bad:
            lost[col] = bad
        df[col] = converted
    return lost


_TS_FORMATS_DAYFIRST = (
    "%Y-%m-%d %H:%M:%S",  # ISO (belirsiz değil)
    "%d/%m/%Y %H:%M",  # InSDN: 24 saat, dakika hassasiyeti (ör. 5/2/2020 13:58)
    "%d/%m/%Y %I:%M:%S %p",  # InSDN: 12 saat AM/PM, saniyeli (ör. 25/12/2019 05:20:00 PM)
    "%d/%m/%Y %H:%M:%S",
)
_TS_FORMATS_MONTHFIRST = tuple(f.replace("%d/%m", "%m/%d") for f in _TS_FORMATS_DAYFIRST)


def parse_timestamps(values: pd.Series, *, dayfirst: bool = True) -> pd.Series:
    """Zaman damgalarını BİLİNEN biçimlerle tek tek, açıkça ayrıştırır.

    pandas'ın tahmine dayalı ayrıştırması (format='mixed') belirsiz tarihte sessizce yanlış
    okuyabilir: dayfirst=True iken '2026-01-02' ISO tarihi 1 Şubat sanılır. Burada her biçim
    tam eşleşme (exact) ile denenir; hiçbir biçime uymayan değer NaT olur ve veri
    incelemesinde uyarı üretir. Sessizce yanlış tarih üretilmez.
    """
    formats = _TS_FORMATS_DAYFIRST if dayfirst else _TS_FORMATS_MONTHFIRST
    text = values.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in formats:
        todo = out.isna()
        if not todo.any():
            break
        out.loc[todo] = pd.to_datetime(text[todo], format=fmt, errors="coerce", exact=True)
    return out


def normalize_frame(
    raw: pd.DataFrame,
    source_file: str,
    *,
    dayfirst: bool = False,
    aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Ham tabloyu standart biçime getirir. (tablo, bilgi sözlüğü) döndürür."""
    df = raw.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        raise ValueError(f"{source_file}: normalleştirme sonrası yinelenen sütun adları: {dup}")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"{source_file}: '{LABEL_COLUMN}' sütunu bulunamadı. Bulunan sütunlar: {list(df.columns)}"
        )
    df["label_raw"] = df[LABEL_COLUMN].astype(str).str.strip()
    canon, unknown = canonicalize_labels(df["label_raw"], aliases)
    df[LABEL_COLUMN] = canon
    df.insert(0, "source_file", source_file)
    df.insert(0, "record_id", [f"{source_file}#{i}" for i in range(len(df))])
    ts_shapes: dict[str, int] = {}
    if "timestamp" in df.columns:
        # Ayrıştırmadan önce ham biçim kalıplarını say (rakamlar 9 ile değiştirilir).
        # Aynı dosyada birden fazla kalıp varsa dosya yeniden kaydedilmiş olabilir.
        shapes = Counter(re.sub(r"\d", "9", s) for s in df["timestamp"].astype(str))
        ts_shapes = dict(shapes.most_common(6))
        df["timestamp"] = parse_timestamps(df["timestamp"], dayfirst=dayfirst)
    lost = _coerce_numeric(df)
    return df, {"unknown_labels": unknown, "coerced_to_nan": lost, "timestamp_shapes": ts_shapes}


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_insdn_zip(
    zip_path: str | Path, *, dayfirst: bool = False, aliases: dict[str, str] | None = None
) -> pd.DataFrame:
    """Zip içindeki tüm CSV dosyalarını okur ve birleştirir.

    Dosyalar diske açılmaz; bellekte okunur. Güvensiz yol adları atlanır.
    Sonuç tablosunun .attrs alanı: kaynak zip SHA-256'sı, bilinmeyen etiketler,
    sayıya çevrilemeyen değer sayıları.
    """
    frames: list[pd.DataFrame] = []
    unknown_all: set[str] = set()
    lost_all: dict[str, dict[str, int]] = {}
    shapes_all: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if info.is_dir() or not name.lower().endswith(".csv"):
                continue
            if name.startswith(("/", "\\")) or ".." in parts or "__MACOSX" in parts:
                continue
            with zf.open(info) as fh:
                raw = pd.read_csv(fh, low_memory=False)
            df, meta = normalize_frame(raw, source_file=name, dayfirst=dayfirst, aliases=aliases)
            frames.append(df)
            unknown_all.update(meta["unknown_labels"])
            shapes_all[name] = meta["timestamp_shapes"]
            if meta["coerced_to_nan"]:
                lost_all[name] = meta["coerced_to_nan"]
    if not frames:
        raise FileNotFoundError(f"{zip_path}: zip içinde CSV dosyası bulunamadı.")
    out = pd.concat(frames, ignore_index=True)
    if out["record_id"].duplicated().any():
        raise ValueError("record_id değerleri benzersiz değil.")
    zip_sha = _file_sha256(zip_path)
    out.attrs["source_zip_sha256"] = zip_sha
    out.attrs["source_sha256"] = zip_sha
    out.attrs["unknown_labels"] = sorted(unknown_all)
    out.attrs["coerced_to_nan"] = lost_all
    out.attrs["timestamp_shapes"] = shapes_all
    return out


def load_insdn_dir(
    dir_path: str | Path, *, dayfirst: bool = False, aliases: dict[str, str] | None = None
) -> pd.DataFrame:
    """Klasördeki (alt klasörler dahil) tüm CSV dosyalarını okur ve birleştirir.

    Sembolik bağlantılar atlanır. source_file, klasöre göreli yoldur. Sonuç tablosunun
    .attrs alanında her dosyanın SHA-256 özeti ve bunlardan türetilen toplu özet bulunur;
    böylece deneyin tam olarak hangi dosyalarla yapıldığı sonradan kanıtlanabilir.
    """
    root = Path(dir_path)
    files = sorted(f for f in root.rglob("*.csv") if f.is_file() and not f.is_symlink())
    if not files:
        raise FileNotFoundError(f"{root}: klasörde CSV dosyası bulunamadı.")
    frames: list[pd.DataFrame] = []
    unknown_all: set[str] = set()
    lost_all: dict[str, dict[str, int]] = {}
    file_hashes: dict[str, str] = {}
    shapes_all: dict[str, dict[str, int]] = {}
    for f in files:
        name = f.relative_to(root).as_posix()
        raw = pd.read_csv(f, low_memory=False)
        df, meta = normalize_frame(raw, source_file=name, dayfirst=dayfirst, aliases=aliases)
        frames.append(df)
        unknown_all.update(meta["unknown_labels"])
        shapes_all[name] = meta["timestamp_shapes"]
        if meta["coerced_to_nan"]:
            lost_all[name] = meta["coerced_to_nan"]
        file_hashes[name] = _file_sha256(f)
    out = pd.concat(frames, ignore_index=True)
    if out["record_id"].duplicated().any():
        raise ValueError("record_id değerleri benzersiz değil.")
    combined = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(file_hashes.items())).encode()).hexdigest()
    out.attrs["source_sha256"] = combined
    out.attrs["source_file_sha256"] = file_hashes
    out.attrs["unknown_labels"] = sorted(unknown_all)
    out.attrs["coerced_to_nan"] = lost_all
    out.attrs["timestamp_shapes"] = shapes_all
    return out


def load_insdn(
    path: str | Path, *, dayfirst: bool = False, aliases: dict[str, str] | None = None
) -> pd.DataFrame:
    """Yol bir klasörse klasörü, değilse zip dosyasını okur."""
    if Path(path).is_dir():
        return load_insdn_dir(path, dayfirst=dayfirst, aliases=aliases)
    return load_insdn_zip(path, dayfirst=dayfirst, aliases=aliases)
