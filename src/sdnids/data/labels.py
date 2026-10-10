"""Etiket adlarını tek biçime getirir ve ikili (normal/saldırı) etikete çevirir.

Alias tablosu, InSDN README'sinde ve literatürde görülen yazım farklarını kapsar
(ör. 'brute-force-attack' ve 'BFA'). Tablo gerçek CSV açılınca DOĞRULANMALIDIR.
Tabloda olmayan etiketler sessizce eşlenmez; 'bilinmeyen' olarak raporlanır.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..schema import BENIGN_LABEL


def _key(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(value).strip().lower()).strip("_")


DEFAULT_LABEL_ALIASES: dict[str, str] = {
    "normal": "Normal",
    "benign": "Normal",
    "dos": "DoS",
    "ddos": "DDoS",
    "probe": "Probe",
    "bfa": "BFA",
    "brute_force": "BFA",
    "brute_force_attack": "BFA",
    "web_attack": "Web-Attack",
    "webattack": "Web-Attack",
    "botnet": "BOTNET",
    # Gerçek CSV'de (metasploitable-2.csv) bu sınıfın adı 'U2R'; README ise 'Exploitation (R2L)'
    # diyor. İki ad birbirine eşlenmedi: README ile CSV uyuşmuyor, bu docs/veri_sozlugu.md'de belgeli.
    "u2r": "U2R",
}


def canonicalize_labels(
    labels: pd.Series, aliases: dict[str, str] | None = None, *, strict: bool = False
) -> tuple[pd.Series, list[str]]:
    """(kanonik etiket serisi, bilinmeyen etiketler) döndürür.

    Bilinmeyen etiket olduğu gibi (boşlukları kırpılmış) bırakılır; strict=True ise
    ValueError verilir. Hiçbir etiket sessizce başka bir sınıfa eşlenmez.
    """
    table = {**DEFAULT_LABEL_ALIASES, **{_key(k): v for k, v in (aliases or {}).items()}}
    unique = pd.unique(labels.astype(str))
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    for raw in unique:
        canon = table.get(_key(raw))
        if canon is None:
            unknown.append(str(raw).strip())
            canon = str(raw).strip()
        mapping[raw] = canon
    if unknown and strict:
        raise ValueError(f"Bilinmeyen etiket(ler): {sorted(unknown)}")
    return labels.astype(str).map(mapping), sorted(unknown)


def to_binary(labels, benign: str = BENIGN_LABEL) -> np.ndarray:
    """0 = normal, 1 = saldırı."""
    arr = np.asarray(labels).astype(str)
    return (arr != benign).astype(np.int64)
