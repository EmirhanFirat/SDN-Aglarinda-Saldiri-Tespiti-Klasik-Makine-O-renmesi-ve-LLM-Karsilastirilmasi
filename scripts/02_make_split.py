#!/usr/bin/env python3
"""Grup bütünlüğünü koruyan train/val/test bölmesini üretir ve diske sabitler.

Çıktılar (--out dizini): split_assignment.csv, split_meta.json, leak_check_train.json
Tek özellikli sızıntı denetimi YALNIZCA eğitim bölümünde çalışır.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdnids.config import load_config  # noqa: E402
from sdnids.data.audit import single_feature_leak_check  # noqa: E402
from sdnids.data.labels import to_binary  # noqa: E402
from sdnids.data.loader import load_insdn_zip  # noqa: E402
from sdnids.data.split import make_split, save_split, verify_split_integrity  # noqa: E402
from sdnids.data.synthetic import require_real_data  # noqa: E402
from sdnids.schema import FeatureSchema, build_schema_from_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "split"))
    ap.add_argument("--schema", default=None, help="Varsayılan: <out>/../audit/schema.json")
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    ap.add_argument("--allow-synthetic", action="store_true", help="Yalnızca altyapı testi için")
    args = ap.parse_args()

    cfg = load_config(args.config if Path(args.config).exists() else None)
    df = load_insdn_zip(args.zip, dayfirst=cfg["data"]["dayfirst"])
    if not args.allow_synthetic:
        require_real_data(df)

    schema_path = Path(args.schema) if args.schema else Path(args.out).parent / "audit" / "schema.json"
    if schema_path.exists():
        schema = FeatureSchema.load(schema_path)
    else:
        schema, _ = build_schema_from_frame(df)
        print(f"Uyarı: {schema_path} yok; şema veriden yeniden üretildi.")

    sp = cfg["split"]
    result = make_split(
        df,
        schema.names,
        ratios=sp["ratios"],
        names=sp["names"],
        seed=cfg["seed"],
        strategy=sp["strategy"],
        time_block_seconds=sp["time_block_seconds"],
        min_test_support=sp["min_test_support"],
    )
    problems = verify_split_integrity(df, result.group_ids, result.assignment, schema.names)
    if problems:
        print("SIZINTI İHLALİ:", *problems, sep="\n - ")
        sys.exit(1)

    params = {**sp, "seed": cfg["seed"], "schema_digest": schema.digest(), "source_zip_sha256": df.attrs["source_zip_sha256"]}
    save_split(result, args.out, params)

    train_mask = result.assignment.reindex(df["record_id"].to_numpy()).to_numpy() == sp["names"][0]
    train = df.loc[train_mask]
    leak = single_feature_leak_check(
        schema.align(train),
        to_binary(train["label"]),
        threshold=cfg["audit"]["leak_threshold"],
        max_rows=cfg["audit"]["leak_max_rows"],
        seed=cfg["seed"],
    )
    (Path(args.out) / "leak_check_train.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

    print("Bölme yazıldı:", args.out)
    print("Kayıt sayıları:", result.report["rows_by_split"])
    for w in result.report["warnings"]:
        print(" -", w)
    if leak["flagged"]:
        print("DİKKAT: tek başına etiketi ele veren özellikler (eğitim bölümünde):", leak["flagged"])


if __name__ == "__main__":
    main()
