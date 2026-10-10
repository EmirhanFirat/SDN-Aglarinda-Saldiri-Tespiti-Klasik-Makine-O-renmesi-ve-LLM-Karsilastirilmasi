"""Aşama 2: klasik modelleri eğitir, doğrulamada seçer ve (yalnızca --evaluate-test ile) test eder.

Önerilen kullanım (iki adım):
  1) Geliştirme: test AÇILMADAN çalıştır, doğrulama sonuçlarına bak.
       python scripts/03_train_eval.py --data /yol/InSDN_DatasetCSV
  2) Ayarlar dondurulduktan sonra bir kez test et (erişim outputs/models/<tasarım>/test_access_log.json'a yazılır):
       python scripts/03_train_eval.py --data /yol/InSDN_DatasetCSV --evaluate-test

Önkoşul: scripts/01_audit_data.py ve scripts/02_make_split.py çalıştırılmış olmalı (şema ve bölme dosyaları).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdnids.config import load_config  # noqa: E402
from sdnids.data.dedupe import deduplicate_vectors  # noqa: E402
from sdnids.data.loader import load_insdn  # noqa: E402
from sdnids.data.synthetic import require_real_data  # noqa: E402
from sdnids.experiment import load_design_context, run_model  # noqa: E402
from sdnids.modeling import FEATURE_SET_NAMES, MODEL_NAMES  # noqa: E402
from sdnids.reporting import collect_results, collect_status, to_markdown  # noqa: E402
from sdnids.schema import FeatureSchema  # noqa: E402

QUICK_OVERRIDES = {
    "n_seeds": 1,
    "bootstrap": {"n_boot": 30},
    "latency": {"n_single": 50, "warmup": 5},
    "rf": {"n_estimators": 20},
    "iforest": {"n_estimators": 30},
    "xgboost": {"n_estimators": 30},
}

SUMMARY_COLS = [
    "tasarım", "özellik_kümesi", "model", "eşik_kuralı", "tohum_sayısı",
    "accuracy", "attack_f1", "attack_f1_std", "fpr", "fnr", "macro_f1", "pr_auc",
    "gecikme_p50_ms", "gecikme_p95_ms", "akış_per_sn", "eğitim_sn_ort", "eğitim_ram_artışı_mb", "model_mb",
]


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", "--zip", dest="data", default=None, help="InSDN CSV klasörü veya .zip")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--designs", nargs="+", default=["within_run", "cross_session"])
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    ap.add_argument("--feature-sets", nargs="+", default=None, choices=list(FEATURE_SET_NAMES))
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--evaluate-test", action="store_true", help="Test bölmesini aç (ayarlar dondurulduktan sonra)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    ap.add_argument("--quick", action="store_true", help="Altyapı denemesi: küçük modeller, tek tohum")
    ap.add_argument("--allow-synthetic", action="store_true", help="Yalnızca altyapı testi için")
    ap.add_argument("--summarize-only", action="store_true", help="Eğitim yapmadan mevcut sonuçlardan özet tabloyu yaz")
    args = ap.parse_args()
    if not args.summarize_only and not args.data:
        ap.error("--data gerekli (yalnızca --summarize-only ile verilmeyebilir)")

    cfg = load_config(args.config if Path(args.config).exists() else None)
    if args.quick:
        cfg["models"] = _merge(cfg["models"], QUICK_OVERRIDES)
    out = Path(args.out)
    models_root = out / "models"
    feature_sets = args.feature_sets or cfg["models"]["feature_sets"]

    failures = []
    designs_to_run = [] if args.summarize_only else args.designs
    if designs_to_run:
        df = load_insdn(args.data, dayfirst=cfg["data"]["dayfirst"])
        if not args.allow_synthetic:
            require_real_data(df)
        schema = FeatureSchema.load(out / "audit" / "schema.json")
        if cfg["dedupe"]:
            df, _, rep = deduplicate_vectors(df, schema.names)
            print(f"Tekrar indirgeme: {rep['rows_in']:,} -> {rep['rows_unique_kept']:,} benzersiz kayıt")

    for design in designs_to_run:
        ctx = load_design_context(df, schema, out / f"split_{design}",
                                  block_seconds=cfg["models"]["bootstrap"]["time_block_seconds"])
        for fs in feature_sets:
            for model in args.models:
                try:
                    run_model(ctx, model, fs, cfg, models_root, evaluate_test=args.evaluate_test, n_seeds=args.n_seeds)
                except Exception as e:  # bir modelin hatası diğerlerini durdurmasın ama gizlenmesin
                    failures.append((design, fs, model, repr(e)))
                    run_dir = models_root / design / fs / model
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "status.json").write_text(json.dumps(
                        {"status": "failed", "model": model, "design": design, "feature_set": fs,
                         "error": repr(e), "traceback": traceback.format_exc()}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
                    print(f"[{design}/{fs}/{model}] HATA: {e!r}")

    for split in ("val", "test"):
        table = collect_results(models_root, split)
        if table.empty:
            continue
        table = table[table["tasarım"].isin(args.designs)]
        (models_root / f"summary_{split}.json").write_text(
            table.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        md = to_markdown(table, [c for c in SUMMARY_COLS if c in table.columns])
        (models_root / f"summary_{split}.md").write_text(md + "\n", encoding="utf-8")
        print(f"\n=== Özet ({'TEST' if split == 'test' else 'doğrulama'}) ===")
        print(md)

    status = collect_status(models_root)
    skipped = status[status["status"] == "skipped"] if not status.empty else status
    for _, r in skipped.iterrows():
        print(f"ATLANDI: {r['design']}/{r['feature_set']}/{r['model']}: {r['reason']}")
    if failures:
        print(f"\n{len(failures)} deney HATA verdi; ayrıntı status.json dosyalarında.")
        sys.exit(1)


if __name__ == "__main__":
    main()
