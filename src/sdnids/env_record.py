"""Deney ortamını kaydeder: kütüphane sürümleri, donanım, rastgelelik tohumu.

Amaç: bir sonuç tablosunun hangi makinede, hangi sürümlerle ve hangi tohumla
üretildiğini sonradan kanıtlayabilmek (yeniden üretilebilirlik).
"""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "xgboost",
    "matplotlib",
    "psutil",
    "joblib",
    "anthropic",
    "scapy",
)


def package_version(name: str) -> str | None:
    """Kurulu paketin sürümünü döndürür; kurulu değilse None."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _run(cmd: list[str], timeout: int = 10) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def gpu_info() -> list[str]:
    """nvidia-smi varsa GPU adı, bellek ve sürücü bilgisini döndürür."""
    if shutil.which("nvidia-smi") is None:
        return []
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    return out.splitlines() if out else []


def total_ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def git_commit(repo_dir: str | os.PathLike | None = None) -> str | None:
    return _run(["git", "-C", str(repo_dir or "."), "rev-parse", "HEAD"])


def collect_environment(
    seed: int | None = None,
    repo_dir: str | os.PathLike | None = None,
    extra: dict | None = None,
) -> dict:
    """Ortam bilgisini sözlük olarak toplar."""
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": total_ram_bytes(),
        "gpus": gpu_info(),
        "kvm_available": os.path.exists("/dev/kvm"),
        "packages": {name: package_version(name) for name in TRACKED_PACKAGES},
        "git_commit": git_commit(repo_dir),
        "seed": seed,
        "extra": extra or {},
    }


def set_global_seed(seed: int) -> None:
    """Python ve NumPy rastgelelik tohumlarını ayarlar.

    Not: PYTHONHASHSEED çalışan süreçte sonradan etkili olmaz; tekrarlanabilirlik
    için sıralama gerektiren yerlerde açıkça sıralama yapılmalıdır.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def write_environment(path: str | os.PathLike, **kwargs) -> dict:
    info = collect_environment(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info
