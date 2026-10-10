"""Özellik şeması: eğitimde ve canlı kullanımda aynı sütunlar, aynı sıra.

Üç ayrı sütun grubu vardır:
  1. Özellikler: modele girer (ör. flow_duration, tot_fwd_pkts).
  2. Kimlik sütunları (IP, port, zaman damgası, flow_id): denetim ve veri bölme
     için saklanır, modele VERİLMEZ. Gerekçe docs/ozellik_ve_sizinti_kararlari.md.
  3. Etiket ve defter tutma sütunları (label, source_file, record_id ...):
     cevabı doğrudan ele verebilir; modele asla girmez.

Şema bir SHA-256 özeti taşır. Eğitim ve canlı hat aynı şema dosyasını yükler;
sütun eksik, fazla veya farklı sırada ise sessizce devam edilmez, hata verilir.
Eksik özellikler asla sıfırla doldurulmaz.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

LABEL_COLUMN = "label"
BINARY_LABEL_COLUMN = "binary_label"
BENIGN_LABEL = "Normal"

# Denetim ve bölme için saklanan, modele verilmeyen kimlik alanları.
IDENTITY_COLUMNS = (
    "flow_id",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "timestamp",
)

# Deney altyapısının tuttuğu yardımcı sütunlar.
BOOKKEEPING_COLUMNS = ("record_id", "source_file", "group_id", "split", "scenario_id")

# Model girdisinde bulunması yasak sütunlar.
FORBIDDEN_INPUT_COLUMNS = frozenset(
    (
        LABEL_COLUMN,
        "label_raw",
        BINARY_LABEL_COLUMN,
        "attack_category",
        "attack_type",
        "file_name",
        "filename",
        "capture_file",
        *IDENTITY_COLUMNS,
        *BOOKKEEPING_COLUMNS,
    )
)


class SchemaMismatchError(ValueError):
    """Veri çerçevesinin sütunları şemayla uyuşmuyor."""


class ForbiddenColumnError(ValueError):
    """Model girdisine yasak bir sütun (etiket, kimlik vb.) girmiş."""


def normalize_column_name(name: str) -> str:
    """'Flow Byts/s' -> 'flow_byts_s'. Aynı ad her zaman aynı sonucu verir."""
    s = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower())
    return s.strip("_")


def assert_no_forbidden_columns(columns: Iterable[str]) -> None:
    bad = sorted({normalize_column_name(c) for c in columns} & FORBIDDEN_INPUT_COLUMNS)
    if bad:
        raise ForbiddenColumnError(f"Model girdisinde yasak sütun(lar) var: {bad}")


# Birim tahmini YALNIZCA ipucudur ve doğrulanmış değildir.
# Birimler, verinin üretildiği araca (ör. Java CICFlowMeter) göre belgelenmeli ve
# canlı özellik çıkaran araçla birebir karşılaştırılmalıdır.
_UNIT_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"(byts_s|bytes_s|pkts_s|packets_s)$"), "1/saniye"),
    (re.compile(r"(duration|iat|idle|active)"), "mikrosaniye"),
    (re.compile(r"(flag_cnt|flags)"), "adet"),
    (re.compile(r"(len|byts|bytes|seg_size|win)"), "bayt"),
    (re.compile(r"pkts?$"), "paket"),
)


def guess_unit(name: str) -> str | None:
    for pattern, unit in _UNIT_RULES:
        if pattern.search(name):
            return unit
    return None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str = "float64"
    unit: str | None = None  # doğrulanmış birim (elle doldurulur)
    unit_guess: str | None = None  # otomatik tahmin; doğrulanmamıştır
    description: str = ""


@dataclass
class FeatureSchema:
    features: list[FeatureSpec]
    version: str = "1"
    source: str = ""
    _names: list[str] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self._names = [f.name for f in self.features]
        if len(set(self._names)) != len(self._names):
            raise SchemaMismatchError("Şemada yinelenen özellik adı var.")
        assert_no_forbidden_columns(self._names)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def digest(self) -> str:
        """Ad ve tür sırasından türetilen SHA-256 özeti."""
        payload = json.dumps([[f.name, f.dtype] for f in self.features], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_columns(self, columns: Sequence[str]) -> None:
        """Sütun listesi şemayla BİREBİR (aynı adlar, aynı sıra) eşleşmeli."""
        cols = list(columns)
        if cols == self._names:
            return
        missing = [n for n in self._names if n not in cols]
        extra = [c for c in cols if c not in self._names]
        detail = []
        if missing:
            detail.append(f"eksik: {missing}")
        if extra:
            detail.append(f"fazla: {extra}")
        if not missing and not extra:
            detail.append("adlar aynı ama sıra farklı")
        raise SchemaMismatchError("Sütunlar şemayla uyuşmuyor (" + "; ".join(detail) + ").")

    def align(self, df):
        """Şemadaki özellikleri, şemadaki sırayla seçer. Eksik sütun varsa HATA verir."""
        missing = [n for n in self._names if n not in df.columns]
        if missing:
            raise SchemaMismatchError(
                f"Eksik özellik(ler): {missing}. Sıfırla doldurma yapılmaz; "
                "özelliği çıkaracak bileşeni düzeltin."
            )
        out = df.loc[:, self._names]
        self.validate_columns(list(out.columns))
        return out

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "digest": self.digest(),
            "features": [asdict(f) for f in self.features],
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSchema":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = cls(
            features=[FeatureSpec(**f) for f in data["features"]],
            version=data.get("version", "1"),
            source=data.get("source", ""),
        )
        if schema.digest() != data.get("digest"):
            raise SchemaMismatchError("Şema dosyasının özeti uyuşmuyor; dosya değiştirilmiş olabilir.")
        return schema


def build_schema_from_frame(df, *, source: str = "", extra_excluded: Sequence[str] = ()):
    """Veri çerçevesinden şema üretir.

    Yalnızca sayısal ve yasak olmayan sütunlar alınır; sütun sırası korunur.
    (şema, atlanan_sayısal_olmayan_sütunlar) döndürür.
    """
    import pandas as pd

    specs: list[FeatureSpec] = []
    skipped: list[str] = []
    excluded = set(extra_excluded)
    for col in df.columns:
        name = normalize_column_name(col)
        if name in FORBIDDEN_INPUT_COLUMNS or name in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            skipped.append(name)
            continue
        specs.append(FeatureSpec(name=name, unit_guess=guess_unit(name)))
    return FeatureSchema(features=specs, source=source), skipped
