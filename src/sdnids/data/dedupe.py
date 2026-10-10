"""Birebir tekrar eden akış vektörlerini tek kayda indirir.

Neden? Gerçek InSDN'de kayıtların yarısından fazlası, bir başka kaydın özellik vektörüyle
birebir aynıdır (DDoS sınıfında 121.942 satırın yalnızca 1.520'si farklıdır). Deterministik bir
model aynı vektöre her zaman aynı cevabı verir; tekrarlar bir akışı "tekrar tekrar" saydırıp
birkaç vektörün sonucu baskın kılar ve başarıyı olduğundan farklı gösterir.

Yöntem:
  * Aynı vektör birden fazla etiketle görülüyorsa model onları ayıramaz: TÜM kopyalar
    değerlendirme dışı bırakılır, kimlikleri ve etiketleri ayrıca kaydedilir.
  * Kalan her vektör için ilk kayıt temsilci olur; `n_copies` kopya sayısını tutar.
    `n_copies` bir özellik DEĞİLDİR (sızıntı olur); yalnızca ağırlıklı raporlama için saklanır.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def deduplicate_vectors(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    label_col: str = "label",
    record_col: str = "record_id",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(benzersiz_tablo, dışlanan_kayıtlar, rapor) döndürür.

    Not: vektör eşleşmesi 64 bitlik satır özetiyle yapılır; 185 bin vektörde çakışma olasılığı
    yaklaşık 1e-9 mertebesindedir ve kabul edilmiştir.
    """
    cols = list(feature_cols)
    h = pd.util.hash_pandas_object(df[cols], index=False).to_numpy()
    tmp = pd.DataFrame({"h": h, "y": df[label_col].to_numpy()})
    grp = tmp.groupby("h")
    conflict = (grp["y"].transform("nunique") > 1).to_numpy()
    copies = grp["h"].transform("size").to_numpy()
    first = ~tmp.duplicated(subset="h", keep="first").to_numpy()
    keep = first & ~conflict

    unique = df.loc[keep].copy()
    unique["n_copies"] = copies[keep]
    excluded = df.loc[conflict, [record_col, label_col]].copy()

    before = df[label_col].value_counts()
    after = unique[label_col].value_counts()
    report = {
        "rows_in": int(len(df)),
        "rows_unique_kept": int(len(unique)),
        "rows_removed_as_copies": int((~first & ~conflict).sum()),
        "rows_excluded_conflicting_label": int(conflict.sum()),
        "conflicting_vectors": int(tmp.loc[conflict, "h"].nunique()),
        "by_class": {
            str(c): {
                "rows_before": int(before[c]),
                "unique_after": int(after.get(c, 0)),
                "kept_share": round(float(after.get(c, 0)) / float(before[c]), 4),
            }
            for c in before.index
        },
    }
    # Tutarlılık: kopya sayıları toplamı, çelişkisiz kayıtların sayısına eşit olmalı.
    assert int(unique["n_copies"].sum()) == int((~conflict).sum()), "n_copies toplamı tutmuyor"
    assert np.unique(h[keep]).size == int(keep.sum()), "benzersiz tabloda hâlâ tekrar var"
    return unique, excluded, report
