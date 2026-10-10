"""Ön işleme: yalnızca EĞİTİM verisinde öğrenilir, diğer bölmelere uygulanır.

Adımlar: sonsuz değeri eksik say -> eğitim medyanı ile doldur -> (isteğe bağlı) işaretli log1p
-> (isteğe bağlı) ölçekle. Ağaç tabanlı modeller ölçeklemeye ve log dönüşümüne ihtiyaç duymaz.

log1p, veriden hiçbir şey öğrenmez (sabit bir fonksiyon); bu yüzden sızıntı riski yoktur. Bayt/süre gibi
çok ağır kuyruklu özelliklerde doğrusal modelin çalışabilmesi içindir.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ..schema import FeatureSchema, assert_no_forbidden_columns
from .labels import to_binary


def _inf_to_nan(X):
    arr = np.asarray(X, dtype="float64")
    return np.where(np.isinf(arr), np.nan, arr)


def _signed_log1p(X):
    arr = np.asarray(X, dtype="float64")
    return np.sign(arr) * np.log1p(np.abs(arr))


def make_preprocessor(*, scale: bool = True, log1p: bool = False) -> Pipeline:
    steps = [
        ("inf_to_nan", FunctionTransformer(_inf_to_nan, validate=False, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if log1p:
        steps.append(("log1p", FunctionTransformer(_signed_log1p, validate=False, feature_names_out="one-to-one")))
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def split_matrices(
    df: pd.DataFrame,
    assignment: pd.Series,
    schema: FeatureSchema,
    *,
    label_col: str = "label",
    record_col: str = "record_id",
    only: Sequence[str] | None = None,
) -> dict[str, dict]:
    """Bölme başına {X, y_binary, y_multi, record_ids} üretir.

    X yalnızca şemadaki özellikleri, şemadaki sırayla içerir; etiket ve kimlik
    sütunlarının girmediği ayrıca doğrulanır.
    only: verilirse yalnızca bu bölmeler üretilir. Ayar/eşik seçimi aşamasında only=("train","val")
    kullanılır; böylece test matrisleri hiç oluşturulmaz.
    """
    split = assignment.reindex(df[record_col].to_numpy()).to_numpy()
    if pd.isna(split).any():
        raise ValueError("Bölmeye atanmamış kayıt var.")
    out: dict[str, dict] = {}
    wanted = None if only is None else {str(o) for o in only}
    for name in pd.unique(split):
        if wanted is not None and str(name) not in wanted:
            continue
        mask = split == name
        sub = df.loc[mask]
        X = schema.align(sub)
        assert_no_forbidden_columns(X.columns)
        out[str(name)] = {
            "X": X,
            "y_binary": to_binary(sub[label_col]),
            "y_multi": sub[label_col].to_numpy(),
            "record_ids": sub[record_col].to_numpy(),
        }
    return out
