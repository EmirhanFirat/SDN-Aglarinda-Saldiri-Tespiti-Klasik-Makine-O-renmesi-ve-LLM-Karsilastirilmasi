"""InSDN CSV dosyalarını okur ve tek bir tablo biçimine getirir.

ÖNEMLİ: InSDN dosyası bu oturumda indirilemedi (çıkış politikası). Sütun adları
CICFlowMeter biçimi varsayılarak yazıldı ('Flow ID', 'Src IP', 'Label' ...).
Gerçek dosya gelince sütun listesi docs/veri_sozlugu.md içine kaydedilip
doğrulanmalıdır. Etiket sütunu bulunamazsa yükleyici açık bir hata verir.
"""

from __future__ import annotations

import hashlib
import zipfile
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
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", dayfirst=dayfirst, format="mixed"
        )
    lost = _coerce_numeric(df)
    return df, {"unknown_labels": unknown, "coerced_to_nan": lost}


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
            if meta["coerced_to_nan"]:
                lost_all[name] = meta["coerced_to_nan"]
    if not frames:
        raise FileNotFoundError(f"{zip_path}: zip içinde CSV dosyası bulunamadı.")
    out = pd.concat(frames, ignore_index=True)
    if out["record_id"].duplicated().any():
        raise ValueError("record_id değerleri benzersiz değil.")
    out.attrs["source_zip_sha256"] = _file_sha256(zip_path)
    out.attrs["unknown_labels"] = sorted(unknown_all)
    out.attrs["coerced_to_nan"] = lost_all
    return out
