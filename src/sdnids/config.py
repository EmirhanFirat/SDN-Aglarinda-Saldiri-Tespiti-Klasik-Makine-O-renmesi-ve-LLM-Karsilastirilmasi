"""Deney yapılandırması: tek bir JSON dosyasından okunur, varsayılanlarla birleştirilir."""

from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS: dict = {
    "seed": 42,
    "data": {"dayfirst": False},
    "split": {
        "ratios": [0.6, 0.2, 0.2],
        "names": ["train", "val", "test"],
        "strategy": "random",
        "time_block_seconds": 300,
        "min_test_support": 30,
    },
    "audit": {"leak_threshold": 0.99, "leak_max_rows": 50000},
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
