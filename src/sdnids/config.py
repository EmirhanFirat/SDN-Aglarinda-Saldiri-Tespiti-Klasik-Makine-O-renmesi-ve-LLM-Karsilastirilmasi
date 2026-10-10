"""Deney yapılandırması: tek bir JSON dosyasından okunur, varsayılanlarla birleştirilir."""

from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS: dict = {
    "seed": 42,
    # InSDN'in ham zaman damgaları gün/ay/yıl sırasındadır (bkz. docs/veri_sozlugu.md).
    "data": {"dayfirst": True},
    "dedupe": True,
    "split": {
        "ratios": [0.6, 0.2, 0.2],
        "names": ["train", "val", "test"],
        "strategy": "random",
        "time_block_seconds": None,
        "min_test_support": 30,
    },
    "cross_session": {
        "target_files": ["metasploitable-2.csv"],
        "shared_files": ["Normal_data.csv"],
        "shared_target_fraction": 0.2,
        "trainval_ratios": [0.75, 0.25],
        "time_block_seconds": 300,
    },
    "audit": {"leak_threshold": 0.99, "leak_max_rows": 50000},
    "models": {
        "n_seeds": 5,
        "n_jobs": -1,
        "feature_sets": ["full", "llm_subset"],
        "latency": {"n_single": 500, "warmup": 50, "batch_size": 1000},
        "bootstrap": {"n_boot": 300, "alpha": 0.05, "time_block_seconds": 300},
        "logreg": {"max_iter": 2000},
        "rf": {"n_estimators": 150},
        "xgboost": {"n_estimators": 300},
        "iforest": {"n_estimators": 200, "train_normal_quantile": 0.99},
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULTS)
    return _merge(DEFAULTS, json.loads(Path(path).read_text(encoding="utf-8")))
