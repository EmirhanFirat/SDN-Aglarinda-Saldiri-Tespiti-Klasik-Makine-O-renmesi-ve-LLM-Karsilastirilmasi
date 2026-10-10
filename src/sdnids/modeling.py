"""Klasik modeller: kayıt defteri, özellik kümeleri, doğrulamada ayar ve eşik seçimi.

Kurallar (kodda zorlanır):
  * Ön işleme + model tek bir sklearn Pipeline'dır; yalnızca EĞİTİM verisinde fit edilir.
  * Hiperparametre ve karar eşiği yalnızca DOĞRULAMA verisinden seçilir. Bu modülün hiçbir
    fonksiyonu test verisi almaz; test, deney koşucusunda ayarlar dondurulduktan sonra açılır.
  * Isolation Forest yalnızca eğitimdeki NORMAL kayıtlarla fit edilir (ön işleme dahil).
  * Eksik özellik sıfırla doldurulmaz; canlı kullanımda sütun listesi/sırası birebir doğrulanır.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

from .data.preprocess import make_preprocessor
from .env_record import set_global_seed
from .metrics import pr_auc, timed_resources
from .schema import FeatureSchema

# --------------------------------------------------------------------------------------
# Özellik kümeleri
# --------------------------------------------------------------------------------------

# LLM'e verilebilecek, insan analistin de okuyup yorumlayabileceği 20 özellik. Seçim alan bilgisine
# dayanır (bağlantı hacmi, hız, paket boyutu, zamanlama, TCP bayrakları); veriden öğrenilmedi ve test
# sonuçlarına bakılarak seçilmedi. Klasik modeller hem bu kümeyle hem 77 özelliğin tamamıyla çalıştırılır;
# LLM karşılaştırmasında yalnızca bu küme kullanılacaktır (adil karşılaştırma: aynı bilgi).
LLM_SUBSET_FEATURES: tuple[str, ...] = (
    "protocol",
    "flow_duration",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "totlen_fwd_pkts",
    "totlen_bwd_pkts",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "pkt_len_std",
    "flow_byts_s",
    "flow_pkts_s",
    "flow_iat_mean",
    "flow_iat_max",
    "syn_flag_cnt",
    "ack_flag_cnt",
    "rst_flag_cnt",
    "fin_flag_cnt",
    "psh_flag_cnt",
    "down_up_ratio",
    "init_bwd_win_byts",
)

FEATURE_SET_NAMES = ("full", "llm_subset")


def resolve_feature_set(name: str, schema: FeatureSchema) -> list[str]:
    """Özellik kümesinin sütunlarını ŞEMA SIRASIYLA döndürür (eğitim/canlı sıra aynı kalır)."""
    if name == "full":
        return list(schema.names)
    if name == "llm_subset":
        missing = [f for f in LLM_SUBSET_FEATURES if f not in schema.names]
        if missing:
            raise ValueError(f"LLM özellik kümesindeki sütunlar şemada yok: {missing}")
        wanted = set(LLM_SUBSET_FEATURES)
        return [n for n in schema.names if n in wanted]
    raise ValueError(f"Bilinmeyen özellik kümesi: {name!r}; geçerli olanlar: {FEATURE_SET_NAMES}")


# --------------------------------------------------------------------------------------
# Model kayıt defteri
# --------------------------------------------------------------------------------------


class ModelUnavailableError(RuntimeError):
    """Model kurulu değil (ör. xgboost). Sessizce atlanmaz; koşucu nedeni kaydeder."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    supervised: bool
    needs_scaling: bool
    log1p: bool
    stochastic: bool
    build: Callable[[dict, int, dict, int], object]
    grid: tuple[dict, ...]
    # Karar kuralları önceden BİLDİRİLİR; test sonucuna bakarak seçilmez. İlki birincil kuraldır.
    threshold_policies: tuple[str, ...]
    description: str = ""
    requires: str | None = None  # kurulması gereken paket

    def availability(self) -> tuple[bool, str | None]:
        if self.requires is None:
            return True, None
        try:
            __import__(self.requires)
            return True, None
        except ImportError as e:
            return False, f"'{self.requires}' paketi kurulu değil ({e})"


THRESHOLD_POLICY_DOCS = {
    "native": "Modelin kendi sınıf kararı (predict). Dummy için çoğunluk sınıfı.",
    "p0.5": "Olasılık >= 0,5. Eşik hiç ayarlanmaz.",
    "val_f1": "Doğrulamada saldırı F1'ini en büyükleyen eşik (doğrulama ETİKETLERİ kullanılır).",
    "train_normal_q99": (
        "Eğitimdeki Normal kayıtların anomali skorlarının %99'luk dilimi: etiket kullanmayan eşik "
        "(eğitim Normal'inin yaklaşık %1'ine alarm verir; eğitim içi skorlar olduğundan iyimser olabilir)."
    ),
}


def _build_dummy(params, seed, cfg, n_jobs):
    return DummyClassifier(strategy="most_frequent")


def _build_logreg(params, seed, cfg, n_jobs):
    return LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=int(cfg.get("max_iter", 2000)),
        random_state=seed,
    )


def _build_rf(params, seed, cfg, n_jobs):
    return RandomForestClassifier(
        n_estimators=int(cfg.get("n_estimators", 150)),
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        n_jobs=n_jobs,
        random_state=seed,
    )


def _build_xgboost(params, seed, cfg, n_jobs):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ModelUnavailableError(f"xgboost kurulu değil: {e}") from e
    return XGBClassifier(
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
        eval_metric="logloss",
    )


def _build_iforest(params, seed, cfg, n_jobs):
    return IsolationForest(
        n_estimators=int(cfg.get("n_estimators", 200)),
        max_samples=params["max_samples"],
        contamination="auto",
        n_jobs=n_jobs,
        random_state=seed,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "dummy": ModelSpec(
        "dummy", "DummyClassifier (çoğunluk sınıfı)", True, False, False, False,
        _build_dummy, ({},), ("native",),
        "Eğitimdeki çoğunluk sınıfını her kayda söyler. Hiçbir model bundan iyi olmak zorundadır.",
    ),
    "logreg": ModelSpec(
        "logreg", "Logistic Regression", True, True, True, False,
        _build_logreg,
        tuple({"C": c, "class_weight": w} for c in (0.01, 0.1, 1.0, 10.0) for w in (None, "balanced")),
        ("p0.5", "val_f1"),
        "Doğrusal sınıflandırıcı; işaretli log1p + standartlaştırma ile.",
    ),
    "rf": ModelSpec(
        "rf", "Random Forest", True, False, False, True,
        _build_rf,
        tuple({"max_depth": d, "min_samples_leaf": l} for d in (12, None) for l in (1, 5)),
        ("p0.5", "val_f1"),
        "Karar ağaçları topluluğu.",
    ),
    "xgboost": ModelSpec(
        "xgboost", "XGBoost", True, False, False, True,
        _build_xgboost,
        tuple({"max_depth": d, "learning_rate": lr} for d in (4, 8) for lr in (0.1, 0.3)),
        ("p0.5", "val_f1"),
        "Gradyan artırmalı ağaçlar.",
        requires="xgboost",
    ),
    "iforest": ModelSpec(
        "iforest", "Isolation Forest (yalnızca Normal ile eğitilir)", False, False, False, True,
        _build_iforest,
        tuple({"max_samples": m} for m in (256, 2048)),
        ("train_normal_q99", "val_f1"),
        "Denetimsiz anomali tespiti: eğitimde SALDIRI ETİKETİ ve saldırı örneği görmez.",
    ),
}

MODEL_NAMES = tuple(MODEL_REGISTRY)


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Bilinmeyen model: {name!r}; geçerli olanlar: {MODEL_NAMES}") from None


# --------------------------------------------------------------------------------------
# Eğitim / skor / eşik
# --------------------------------------------------------------------------------------


def build_pipeline(spec: ModelSpec, params: dict, seed: int, model_cfg: dict | None = None, n_jobs: int = 1) -> Pipeline:
    est = spec.build(params, seed, model_cfg or {}, n_jobs)
    prep = make_preprocessor(scale=spec.needs_scaling, log1p=spec.log1p)
    return Pipeline([("prep", prep), ("clf", est)])


def fit_pipeline(spec: ModelSpec, pipe: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    """Denetimli modeller tüm eğitimle; Isolation Forest yalnızca eğitimdeki Normal kayıtlarla fit edilir."""
    y = np.asarray(y_train)
    if spec.supervised:
        pipe.fit(X_train, y)
    else:
        normal = y == 0
        if not normal.any():
            raise ValueError("Eğitimde Normal kayıt yok; Isolation Forest eğitilemez.")
        pipe.fit(X_train.loc[normal] if isinstance(X_train, pd.DataFrame) else X_train[normal])
    return pipe


def attack_score(spec: ModelSpec, pipe: Pipeline, X) -> np.ndarray:
    """Büyük skor = saldırı olasılığı/anomalisi daha yüksek. Kalibre olasılık iddiası yoktur."""
    if spec.supervised:
        proba = pipe.predict_proba(X)
        classes = list(pipe.named_steps["clf"].classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(proba))  # eğitimde hiç saldırı yoksa (dejenere) olasılık 0
    return -pipe.score_samples(X)


def select_threshold_f1(y_true, score) -> float:
    """Saldırı F1'ini en büyükleyen eşik (>= eşik ise saldırı). Yalnızca DOĞRULAMA verisiyle çağrılmalı."""
    yt = np.asarray(y_true).astype(np.int64)
    s = np.asarray(score, dtype=np.float64)
    if yt.min() == yt.max():
        raise ValueError("Eşik seçimi için doğrulamada iki sınıf da bulunmalı.")
    precision, recall, thr = precision_recall_curve(yt, s)
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thr[int(np.argmax(f1))])


def apply_policy(policy: str, pipe: Pipeline, X, score: np.ndarray, thresholds: dict) -> np.ndarray:
    if policy == "native":
        return np.asarray(pipe.predict(X)).astype(np.int64)
    if policy == "p0.5":
        return (score >= 0.5).astype(np.int64)
    thr = thresholds[policy]
    return (score >= thr).astype(np.int64)


@dataclass
class FittedModel:
    spec_name: str
    params: dict
    seed: int
    feature_set: str
    features: list[str]
    schema_digest: str
    pipeline: Pipeline
    thresholds: dict
    train_resources: dict
    warnings: list[str] = field(default_factory=list)

    @property
    def spec(self) -> ModelSpec:
        return get_model_spec(self.spec_name)

    def score(self, X) -> np.ndarray:
        return attack_score(self.spec, self.pipeline, X)

    def predict_all(self, X) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        s = self.score(X)
        return s, {p: apply_policy(p, self.pipeline, X, s, self.thresholds) for p in self.spec.threshold_policies}

    def check_columns(self, columns) -> None:
        """Canlı/ileri kullanımda sütun listesi ve sırası eğitimdekiyle BİREBİR aynı olmalı."""
        cols = list(columns)
        if cols != self.features:
            missing = [c for c in self.features if c not in cols]
            extra = [c for c in cols if c not in self.features]
            raise ValueError(
                f"Sütunlar eğitimdekiyle aynı değil. Eksik: {missing[:5]}, fazla: {extra[:5]}, "
                f"sıra {'farklı' if not missing and not extra else 'bozuk'}. Eksik özellik sıfırla doldurulmaz."
            )


def tune_hyperparameters(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    *,
    seed: int,
    model_cfg: dict | None = None,
    n_jobs: int = 1,
) -> dict:
    """Izgara taraması: ölçüt doğrulama PR-AUC'si (eşikten bağımsız). Test verisi bu fonksiyona girmez."""
    records = []
    for params in spec.grid:
        set_global_seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = build_pipeline(spec, params, seed, model_cfg, n_jobs)
            with timed_resources() as res:
                fit_pipeline(spec, pipe, X_train, y_train)
            s = attack_score(spec, pipe, X_val)
        records.append({"params": params, "val_pr_auc": pr_auc(y_val, s), "fit_seconds": res["seconds"]})
    ranked = [(-np.inf if np.isnan(r["val_pr_auc"]) else r["val_pr_auc"], -i) for i, r in enumerate(records)]
    best_i = max(range(len(records)), key=lambda i: ranked[i])  # eşitlikte ilk (daha basit) kayıt
    return {"grid": records, "best_params": records[best_i]["params"], "criterion": "val_pr_auc"}


def fit_model(
    spec: ModelSpec,
    params: dict,
    seed: int,
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    *,
    feature_set: str,
    schema_digest: str,
    model_cfg: dict | None = None,
    n_jobs: int = 1,
) -> FittedModel:
    """Eğitir, ardından eşikleri DOĞRULAMA (ve etiketsiz kural için eğitim-Normal) skorlarından hesaplar."""
    model_cfg = model_cfg or {}
    set_global_seed(seed)
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        pipe = build_pipeline(spec, params, seed, model_cfg, n_jobs)
        with timed_resources() as res:
            fit_pipeline(spec, pipe, X_train, y_train)
        s_val = attack_score(spec, pipe, X_val)
        thresholds: dict = {}
        for pol in spec.threshold_policies:
            if pol in ("native", "p0.5"):
                thresholds[pol] = None if pol == "native" else 0.5
            elif pol == "val_f1":
                thresholds[pol] = select_threshold_f1(y_val, s_val)
            elif pol == "train_normal_q99":
                q = float(model_cfg.get("train_normal_quantile", 0.99))
                normal = np.asarray(y_train) == 0
                s_tr = attack_score(spec, pipe, X_train.loc[normal])
                thresholds[pol] = float(np.quantile(s_tr, q))
            else:
                raise ValueError(f"Bilinmeyen eşik kuralı: {pol}")
    for w in wlist:
        msg = f"{w.category.__name__}: {w.message}"
        if msg not in caught:
            caught.append(msg)
    return FittedModel(
        spec_name=spec.name,
        params=params,
        seed=seed,
        feature_set=feature_set,
        features=list(X_train.columns),
        schema_digest=schema_digest,
        pipeline=pipe,
        thresholds=thresholds,
        train_resources=res,
        warnings=caught,
    )


def single_thread_copy(model: FittedModel) -> Pipeline:
    """Gecikme ölçümü için n_jobs=1'e indirilmiş kopya (denetleyiciye gömülü tek iş parçacıklı kullanım)."""
    pipe = copy.deepcopy(model.pipeline)
    clf = pipe.named_steps["clf"]
    if "n_jobs" in clf.get_params():
        clf.set_params(n_jobs=1)
    return pipe


def save_bundle(model: FittedModel, path) -> None:
    import joblib

    joblib.dump(
        {
            "pipeline": model.pipeline,
            "model": model.spec_name,
            "params": model.params,
            "seed": model.seed,
            "feature_set": model.feature_set,
            "features": model.features,
            "schema_digest": model.schema_digest,
            "thresholds": model.thresholds,
            "policies": list(model.spec.threshold_policies),
        },
        path,
    )


def load_bundle(path) -> FittedModel:
    import joblib

    b = joblib.load(path)
    return FittedModel(
        spec_name=b["model"],
        params=b["params"],
        seed=b["seed"],
        feature_set=b["feature_set"],
        features=b["features"],
        schema_digest=b["schema_digest"],
        pipeline=b["pipeline"],
        thresholds=b["thresholds"],
        train_resources={},
    )
