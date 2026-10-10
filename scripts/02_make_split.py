#!/usr/bin/env python3
"""Tekrarları tek kayda indirir ve iki ayrı deney bölmesi üretir, diske sabitler.

Tasarımlar (--design):
  within_run     Aynı koşu içi: bağlantı parçaları ve tekrarlar aynı bölmede, sınıf dengeli
                 %60/%20/%20. İYİMSER bir tahmindir (aynı saldırı koşusunun akışları hem
                 eğitimde hem testte olabilir).
  cross_session  Koşular arası: ikinci kurban makinenin (metasploitable-2) yakalaması tamamen test,
                 Normal trafiğin kronolojik son %20'si test. Farklı koşula genelleme ölçer.

Çıktılar (--out altında): dedupe/, split_<tasarım>/{split_assignment.csv, split_meta.json,
record_weights.csv, leak_check_train.json}. Tek özellikli sızıntı denetimi YALNIZCA eğitimde çalışır.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdnids.config import load_config  # noqa: E402
from sdnids.data.audit import single_feature_leak_check  # noqa: E402
from sdnids.data.dedupe import deduplicate_vectors  # noqa: E402
from sdnids.data.labels import to_binary  # noqa: E402
from sdnids.data.loader import load_insdn  # noqa: E402
from sdnids.data.split import (  # noqa: E402
    make_domain_holdout_split,
    make_split,
    save_split,
    verify_split_integrity,
)
from sdnids.data.synthetic import require_real_data  # noqa: E402
from sdnids.schema import FeatureSchema, build_schema_from_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", "--zip", dest="data", required=True, help="InSDN CSV klasörü veya .zip dosyası")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--design", choices=["within_run", "cross_session", "both"], default="both")
    ap.add_argument("--schema", default=None, help="Varsayılan: <out>/audit/schema.json")
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    ap.add_argument("--allow-synthetic", action="store_true", help="Yalnızca altyapı testi için")
    args = ap.parse_args()

    cfg = load_config(args.config if Path(args.config).exists() else None)
    out = Path(args.out)
    df = load_insdn(args.data, dayfirst=cfg["data"]["dayfirst"])
    if not args.allow_synthetic:
        require_real_data(df)

    schema_path = Path(args.schema) if args.schema else out / "audit" / "schema.json"
    if schema_path.exists():
        schema = FeatureSchema.load(schema_path)
    else:
        schema, _ = build_schema_from_frame(df)
        print(f"Uyarı: {schema_path} yok; şema veriden yeniden üretildi.")

    dedupe_report = None
    if cfg["dedupe"]:
        df, excluded, dedupe_report = deduplicate_vectors(df, schema.names)
        (out / "dedupe").mkdir(parents=True, exist_ok=True)
        (out / "dedupe" / "dedupe_report.json").write_text(
            json.dumps(dedupe_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        excluded.to_csv(out / "dedupe" / "excluded_conflicting_label.csv", index=False)
        print(
            f"Tekrar indirgeme: {dedupe_report['rows_in']:,} satır -> {dedupe_report['rows_unique_kept']:,} benzersiz; "
            f"{dedupe_report['rows_excluded_conflicting_label']} satır çelişkili etiket nedeniyle dışlandı."
        )

    designs = ["within_run", "cross_session"] if args.design == "both" else [args.design]
    sp = cfg["split"]
    for design in designs:
        if design == "within_run":
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
        else:
            cs = cfg["cross_session"]
            result = make_domain_holdout_split(
                df,
                schema.names,
                target_files=cs["target_files"],
                shared_files=cs["shared_files"],
                shared_target_fraction=cs["shared_target_fraction"],
                trainval_ratios=cs["trainval_ratios"],
                names=sp["names"],
                seed=cfg["seed"],
                time_block_seconds=cs["time_block_seconds"],
                min_test_support=sp["min_test_support"],
            )

        problems = verify_split_integrity(df, result.group_ids, result.assignment, schema.names)
        if problems:
            print(f"[{design}] SIZINTI İHLALİ:", *problems, sep="\n - ")
            sys.exit(1)

        out_dir = out / f"split_{design}"
        params = {
            "design": design,
            "seed": cfg["seed"],
            "dedupe": cfg["dedupe"],
            "dedupe_summary": {k: v for k, v in (dedupe_report or {}).items() if k != "by_class"},
            "split": sp,
            "cross_session": cfg["cross_session"] if design == "cross_session" else None,
            "schema_digest": schema.digest(),
            "source_sha256": df.attrs.get("source_sha256"),
        }
        save_split(result, out_dir, params)
        if "n_copies" in df.columns:
            df[["record_id", "n_copies"]].to_csv(out_dir / "record_weights.csv", index=False)

        names = sp["names"]
        mask = result.assignment.reindex(df["record_id"].to_numpy()).to_numpy() == names[0]
        train = df.loc[mask]
        leak = single_feature_leak_check(
            schema.align(train),
            to_binary(train["label"]),
            threshold=cfg["audit"]["leak_threshold"],
            max_rows=cfg["audit"]["leak_max_rows"],
            seed=cfg["seed"],
        )
        (out_dir / "leak_check_train.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

        print(f"\n=== {design} -> {out_dir} ===")
        table = pd.DataFrame(result.report["class_counts_by_split"]).T[names]
        table["toplam"] = table.sum(axis=1)
        print(table.to_string())
        print("Bölme başına kayıt:", result.report["rows_by_split"])
        for w in result.report["warnings"]:
            print(" -", w)
        if leak["flagged"]:
            print("DİKKAT: tek başına etiketi ele veren özellikler (eğitimde):", leak["flagged"])
        else:
            print("Tek özellikli sızıntı denetimi (eğitim): eşiği aşan özellik yok; en yüksek 3:",
                  dict(list(leak["top5"].items())[:3]))


if __name__ == "__main__":
    main()
