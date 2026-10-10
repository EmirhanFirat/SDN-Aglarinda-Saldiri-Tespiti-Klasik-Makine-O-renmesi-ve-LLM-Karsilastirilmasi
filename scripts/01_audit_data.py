#!/usr/bin/env python3
"""InSDN zip dosyasını okur, özellik şemasını üretir ve veri inceleme raporu yazar.

Çıktılar (--out dizini): schema.json, audit.json, audit.md
Bu betik model eğitmez ve hiçbir metrik üretmez.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdnids.config import load_config  # noqa: E402
from sdnids.data.audit import audit_quality, write_audit_report  # noqa: E402
from sdnids.data.loader import load_insdn  # noqa: E402
from sdnids.data.synthetic import is_synthetic, require_real_data  # noqa: E402
from sdnids.schema import build_schema_from_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", "--zip", dest="data", required=True, help="InSDN CSV klasörü veya .zip dosyası")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "audit"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    ap.add_argument("--allow-synthetic", action="store_true", help="Yalnızca altyapı testi için")
    args = ap.parse_args()

    cfg = load_config(args.config if Path(args.config).exists() else None)
    df = load_insdn(args.data, dayfirst=cfg["data"]["dayfirst"])
    if not args.allow_synthetic:
        require_real_data(df)

    schema, skipped = build_schema_from_frame(df, source=f"source_sha256={df.attrs['source_sha256']}")
    out = Path(args.out)
    schema.save(out / "schema.json")

    report = audit_quality(df, schema.names)
    report["data_origin"] = "synthetic" if is_synthetic(df) else "file"
    report["source_sha256"] = df.attrs["source_sha256"]
    report["source_file_sha256"] = df.attrs.get("source_file_sha256", {})
    report["unknown_labels"] = df.attrs["unknown_labels"]
    report["coerced_to_nan"] = df.attrs["coerced_to_nan"]
    report["non_numeric_columns_skipped"] = skipped
    report["schema_digest"] = schema.digest()
    if df.attrs["unknown_labels"]:
        report["flags"].append(f"Alias tablosunda olmayan etiketler: {df.attrs['unknown_labels']}")
    if df.attrs["coerced_to_nan"]:
        report["flags"].append("Sayıya çevrilemeyen değerler NaN yapıldı; ayrıntı coerced_to_nan alanında.")

    title = "Veri inceleme raporu" + (" (SENTETİK - ALTYAPI TESTİ)" if is_synthetic(df) else "")
    write_audit_report(report, out / "audit.json", out / "audit.md", title=title)

    print(f"{report['n_rows']} kayıt, {report['n_features']} özellik. Rapor: {out}")
    for flag in report["flags"]:
        print(" -", flag)


if __name__ == "__main__":
    main()
