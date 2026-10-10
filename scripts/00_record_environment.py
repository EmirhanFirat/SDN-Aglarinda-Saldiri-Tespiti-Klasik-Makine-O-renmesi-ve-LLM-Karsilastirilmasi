#!/usr/bin/env python3
"""Ortam bilgisini (sürümler, donanım, tohum) outputs/environment.json dosyasına yazar."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdnids.env_record import write_environment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "environment.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    info = write_environment(args.out, seed=args.seed, repo_dir=ROOT)
    print(f"Yazıldı: {args.out}")
    print(f"Python {info['python'].split()[0]} | CPU {info['cpu_count']} | GPU {info['gpus'] or 'yok'}")


if __name__ == "__main__":
    main()
