"""Deney koşucusu: (bölme tasarımı x özellik kümesi x model) için eğit, doğrulamada seç, isteğe bağlı test et.

Akış (her model için):
  1. Eğitim ve doğrulama matrisleri oluşturulur. evaluate_test=False ise TEST MATRİSİ HİÇ OLUŞTURULMAZ.
  2. Hiperparametre ızgarası doğrulama PR-AUC'siyle seçilir (ilk tohumla).
  3. Seçilen ayarla her tohum için model eğitilir; eşikler o tohumun doğrulama skorlarından hesaplanır.
  4. evaluate_test=True ise, ayar ve eşikler sabitlendikten SONRA test skorlanır; erişim kaydı yazılır.
  5. Birincil tohum için: kayıt bazında tahminler, küme bootstrap güven aralıkları, sınıf bazlı sonuçlar,
     gecikme/verim/bellek ölçümü ve model dosyası kaydedilir.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data.preprocess import split_matrices
from .data.split import load_split
from .env_record import collect_environment
from .metrics import (
    HEADLINE_METRICS,
    LATENCY_SCOPE,
    PR_AUC_METHOD,
    binary_metrics,
    cluster_bootstrap,
    cluster_ids_from_frame,
    measure_single_record_latency,
    measure_throughput,
    per_class_detection,
    pickled_size_mb,
)
from .modeling import (
    THRESHOLD_POLICY_DOCS,
    FittedModel,
    ModelUnavailableError,
    attack_score,
    fit_model,
    get_model_spec,
    resolve_feature_set,
    save_bundle,
    single_thread_copy,
    tune_hyperparameters,
)
from .schema import FeatureSchema

AGG_METRICS = tuple(HEADLINE_METRICS) + ("pr_auc",)


@dataclass
class DesignContext:
    design: str
    split_dir: Path
    df: pd.DataFrame
    schema: FeatureSchema
    assignment: pd.Series
    group_by_id: pd.Series
    cluster_by_id: pd.Series
    meta: dict


@dataclass
class SplitData:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    n_copies: np.ndarray
    groups: np.ndarray
    clusters: np.ndarray


def load_design_context(
    df: pd.DataFrame, schema: FeatureSchema, split_dir: str | Path, *, block_seconds: float = 300.0
) -> DesignContext:
    """Kaydedilmiş bölmeyi okur (SHA-256 doğrulamasıyla) ve veriyle birebir eşleştiğini denetler."""
    split_dir = Path(split_dir)
    assignment, groups, meta = load_split(split_dir)
    data_ids = pd.Index(df["record_id"].astype(str))
    if not data_ids.is_unique:
        raise ValueError("Veride yinelenen record_id var.")
    if set(data_ids) != set(assignment.index):
        missing = len(set(assignment.index) - set(data_ids))
        extra = len(set(data_ids) - set(assignment.index))
        raise ValueError(
            f"Veri ile bölme dosyası uyuşmuyor ({missing} kayıt bölmede var veride yok, {extra} kayıt tersi). "
            "Bölmeyi aynı veri ve aynı tekrar indirgeme ayarıyla yeniden üretin."
        )
    digest = meta.get("params", {}).get("schema_digest")
    if digest and digest != schema.digest():
        raise ValueError("Bölme başka bir özellik şemasıyla üretilmiş (schema_digest uyuşmuyor).")
    clusters = pd.Series(cluster_ids_from_frame(df, block_seconds=block_seconds), index=data_ids)
    return DesignContext(
        design=str(meta.get("params", {}).get("design", split_dir.name.replace("split_", ""))),
        split_dir=split_dir,
        df=df,
        schema=schema,
        assignment=assignment,
        group_by_id=pd.Series(groups, index=assignment.index),
        cluster_by_id=clusters,
        meta=meta,
    )


def make_split_data(ctx: DesignContext, names: Sequence[str], features: Sequence[str]) -> dict[str, SplitData]:
    mats = split_matrices(ctx.df, ctx.assignment, ctx.schema, only=tuple(names))
    by_id = ctx.df.set_index(ctx.df["record_id"].astype(str))
    out = {}
    for name in names:
        if name not in mats:
            raise ValueError(f"'{name}' bölmesi boş ya da yok.")
        m = mats[name]
        rids = m["record_ids"].astype(str)
        n_copies = by_id.loc[rids, "n_copies"].to_numpy() if "n_copies" in by_id.columns else np.ones(len(rids))
        out[name] = SplitData(
            name=name,
            X=m["X"][list(features)],
            y=m["y_binary"],
            labels=m["y_multi"],
            record_ids=rids,
            n_copies=np.asarray(n_copies, dtype=np.float64),
            groups=ctx.group_by_id.loc[rids].to_numpy(),
            clusters=ctx.cluster_by_id.loc[rids].to_numpy(),
        )
    return out


def _aggregate(per_seed: dict) -> dict:
    """{seed: {policy: metrics}} -> {policy: {metric: {mean,std,min,max,n}}}"""
    out: dict = {}
    seeds = list(per_seed)
    for policy in per_seed[seeds[0]]:
        out[policy] = {}
        for k in AGG_METRICS:
            vals = np.array([per_seed[s][policy][k] for s in seeds], dtype=float)
            ok = vals[~np.isnan(vals)]
            out[policy][k] = {
                "mean": float(ok.mean()) if len(ok) else float("nan"),
                "std": float(ok.std(ddof=1)) if len(ok) > 1 else 0.0 if len(ok) == 1 else float("nan"),
                "min": float(ok.min()) if len(ok) else float("nan"),
                "max": float(ok.max()) if len(ok) else float("nan"),
                "n": int(len(ok)),
            }
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    return str(o)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_json_default).encode("utf-8")).hexdigest()[:16]


def _append_test_access(log_path: Path, entry: dict) -> None:
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    log.append(entry)
    _write_json(log_path, log)


def _measure_speed(model: FittedModel, ctx: DesignContext, va: SplitData, cfg_lat: dict, seed: int) -> dict:
    """Doğrulama kayıtlarıyla (test açılmadan) tek kayıt gecikmesi ve toplu verim."""
    pipe = single_thread_copy(model)
    spec = model.spec
    X = va.X.to_numpy(dtype=np.float64)
    n = min(int(cfg_lat.get("n_single", 500)), len(X))
    rows = [X[i : i + 1] for i in range(min(len(X), 2000))]
    model_only = measure_single_record_latency(
        lambda r: attack_score(spec, pipe, r), rows, n=n, warmup=int(cfg_lat.get("warmup", 50)), seed=seed
    )
    full_rows = ctx.schema.align(ctx.df.set_index(ctx.df["record_id"].astype(str)).loc[va.record_ids[:2000]])
    df_rows = [full_rows.iloc[i : i + 1] for i in range(len(full_rows))]
    feats = model.features
    with_align = measure_single_record_latency(
        lambda r: attack_score(spec, pipe, ctx.schema.align(r)[feats].to_numpy(dtype=np.float64)),
        df_rows,
        n=n,
        warmup=int(cfg_lat.get("warmup", 50)),
        seed=seed,
    )
    thr = measure_throughput(lambda b: attack_score(spec, pipe, b), X, batch_size=int(cfg_lat.get("batch_size", 1000)))
    return {
        "model_only": model_only,
        "with_schema_align": with_align,
        "throughput_batch": thr,
        "n_jobs": 1,
        "scope": LATENCY_SCOPE,
        "measured_on": "doğrulama kayıtları (test kullanılmadı)",
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
    }


def run_model(
    ctx: DesignContext,
    model_name: str,
    feature_set: str,
    cfg: dict,
    out_root: str | Path,
    *,
    evaluate_test: bool = False,
    n_seeds: int | None = None,
    log=print,
) -> dict:
    """Tek (tasarım, özellik kümesi, model) deneyi. Durum sözlüğü döndürür ve sonuçları diske yazar."""
    spec = get_model_spec(model_name)
    mcfg_all = cfg.get("models", {})
    out = Path(out_root) / ctx.design / feature_set / model_name
    out.mkdir(parents=True, exist_ok=True)
    # Önceki çalıştırmanın çıktıları yenisiyle karışmasın (ör. eski test sonucu + yeni ayar).
    for stale in list(out.glob("*.json")) + list(out.glob("*.csv.gz")) + list(out.glob("*.joblib")):
        stale.unlink()
    started = datetime.now(timezone.utc).isoformat()

    ok, reason = spec.availability()
    if not ok:
        status = {"status": "skipped", "reason": reason, "model": model_name, "design": ctx.design,
                  "feature_set": feature_set, "at_utc": started}
        _write_json(out / "status.json", status)
        log(f"[{ctx.design}/{feature_set}/{model_name}] ATLANDI: {reason}")
        return status

    features = resolve_feature_set(feature_set, ctx.schema)
    data = make_split_data(ctx, ("train", "val"), features)
    tr, va = data["train"], data["val"]
    te = make_split_data(ctx, ("test",), features)["test"] if evaluate_test else None

    base_seed = int(cfg.get("seed", 42))
    k = int(n_seeds if n_seeds is not None else mcfg_all.get("n_seeds", 5))
    seeds = [base_seed + i for i in range(k)] if spec.stochastic else [base_seed]
    mcfg = dict(mcfg_all.get(model_name, {}))
    n_jobs = int(mcfg_all.get("n_jobs", -1))

    log(f"[{ctx.design}/{feature_set}/{model_name}] ızgara: {len(spec.grid)} ayar, tohumlar: {seeds}")
    try:
        selection = tune_hyperparameters(spec, tr.X, tr.y, va.X, va.y, seed=seeds[0], model_cfg=mcfg, n_jobs=n_jobs)
    except ModelUnavailableError as e:
        status = {"status": "skipped", "reason": str(e), "model": model_name, "design": ctx.design,
                  "feature_set": feature_set, "at_utc": started}
        _write_json(out / "status.json", status)
        return status
    params = selection["best_params"]

    val_per_seed, test_per_seed, seed_info = {}, {}, {}
    primary: dict = {}
    for seed in seeds:
        model = fit_model(spec, params, seed, tr.X, tr.y, va.X, va.y, feature_set=feature_set,
                          schema_digest=ctx.schema.digest(), model_cfg=mcfg, n_jobs=n_jobs)
        s_val, p_val = model.predict_all(va.X)
        val_per_seed[seed] = {p: binary_metrics(va.y, p_val[p], s_val) for p in spec.threshold_policies}
        seed_info[seed] = {"thresholds": model.thresholds, "train_resources": model.train_resources,
                           "warnings": model.warnings}
        # --- Bu noktada bu tohum için ayar ve eşikler SABİTTİR. Test yalnızca bundan sonra skorlanır. ---
        if te is not None:
            s_te, p_te = model.predict_all(te.X)
            test_per_seed[seed] = {p: binary_metrics(te.y, p_te[p], s_te) for p in spec.threshold_policies}
        if seed == seeds[0]:
            primary = {"model": model, "s_val": s_val, "p_val": p_val}
            if te is not None:
                primary.update({"s_te": s_te, "p_te": p_te})

    model = primary["model"]
    speed = _measure_speed(model, ctx, va, mcfg_all.get("latency", {}), seeds[0])
    size_mb = pickled_size_mb(model.pipeline)
    save_bundle(model, out / f"model_seed{seeds[0]}.joblib")

    # Kayıt bazında tahminler (birincil tohum)
    frames = []
    for sd, s, p in ((va, primary["s_val"], primary["p_val"]),) + (((te, primary["s_te"], primary["p_te"]),) if te is not None else ()):
        f = pd.DataFrame({"record_id": sd.record_ids, "split": sd.name, "label": sd.labels, "y_true": sd.y, "score": s})
        for pol, pred in p.items():
            f[f"pred_{pol}"] = pred
        frames.append(f)
    pd.concat(frames).to_csv(out / f"predictions_seed{seeds[0]}.csv.gz", index=False, compression="gzip")

    _write_json(out / "selection.json", {
        "criterion": selection["criterion"],
        "grid": selection["grid"],
        "best_params": params,
        "seeds": seeds,
        "per_seed": seed_info,
    })
    _write_json(out / "val_metrics.json", {"per_seed": val_per_seed, "aggregate": _aggregate(val_per_seed),
                                            "n_records": int(len(va.y)), "n_attack": int(va.y.sum())})

    if te is not None:
        bcfg = mcfg_all.get("bootstrap", {})
        n_boot, alpha = int(bcfg.get("n_boot", 300)), float(bcfg.get("alpha", 0.05))
        min_support = int(cfg.get("split", {}).get("min_test_support", 30))
        details = {}
        for i, pol in enumerate(spec.threshold_policies):
            pred = primary["p_te"][pol]
            details[pol] = {
                "ci_time_block": cluster_bootstrap(te.y, pred, primary["s_te"], te.clusters, n_boot=n_boot,
                                                   seed=seeds[0], alpha=alpha, include_pr_auc=(i == 0)),
                "ci_leak_group": cluster_bootstrap(te.y, pred, primary["s_te"], te.groups, n_boot=n_boot,
                                                   seed=seeds[0], alpha=alpha, include_pr_auc=(i == 0)),
                "per_class": per_class_detection(te.labels, pred, min_support=min_support).to_dict(orient="records"),
                "weighted_by_n_copies": binary_metrics(te.y, pred, primary["s_te"], sample_weight=te.n_copies),
            }
        _write_json(out / "test_metrics.json", {
            "per_seed": test_per_seed,
            "aggregate": _aggregate(test_per_seed),
            "primary_seed": seeds[0],
            "primary_seed_details": details,
            "n_records": int(len(te.y)),
            "n_attack": int(te.y.sum()),
            "n_time_block_clusters": int(pd.Series(te.clusters).nunique()),
            "n_leak_groups": int(pd.Series(te.groups).nunique()),
        })
        _append_test_access(Path(out_root) / ctx.design / "test_access_log.json", {
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "feature_set": feature_set,
            "best_params": params,
            "config_hash": _config_hash(cfg),
            "test_sha256": ctx.meta.get("record_id_sha256_by_split", {}).get("test"),
        })

    run_meta = {
        "model": model_name,
        "title": spec.title,
        "description": spec.description,
        "supervised": spec.supervised,
        "training_information": (
            "Eğitim: tüm eğitim kayıtları ve ikili etiketleri." if spec.supervised else
            "Eğitim: YALNIZCA eğitimdeki Normal kayıtlar; saldırı örneği ve etiketi görülmez. "
            "Hiperparametre seçimi ve 'val_f1' eşiği doğrulama etiketlerini kullanır; 'train_normal_q99' kullanmaz."
        ),
        "design": ctx.design,
        "feature_set": feature_set,
        "features": features,
        "n_features": len(features),
        "schema_digest": ctx.schema.digest(),
        "split_dir": ctx.split_dir.name,  # mutlak yol yazılmaz (makineye özgü)
        "split_sha256": ctx.meta.get("record_id_sha256_by_split"),
        "data_source_sha256": ctx.df.attrs.get("source_sha256"),
        "n_train": int(len(tr.y)),
        "n_val": int(len(va.y)),
        "n_train_used_for_fit": int(len(tr.y) if spec.supervised else int((tr.y == 0).sum())),
        "seeds": seeds,
        "model_config": mcfg,
        "n_jobs_training": n_jobs,
        "threshold_policies": {p: THRESHOLD_POLICY_DOCS[p] for p in spec.threshold_policies},
        "primary_policy": spec.threshold_policies[0],
        "pr_auc_method": PR_AUC_METHOD,
        "speed": speed,
        "model_size_mb_pickled": size_mb,
        "test_evaluated": te is not None,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(seed=base_seed),
    }
    _write_json(out / "run_meta.json", run_meta)
    status = {"status": "ok", "model": model_name, "design": ctx.design, "feature_set": feature_set,
              "test_evaluated": te is not None, "at_utc": run_meta["finished_utc"]}
    _write_json(out / "status.json", status)
    log(f"[{ctx.design}/{feature_set}/{model_name}] tamam (test {'değerlendirildi' if te is not None else 'açılmadı'})")
    return status
