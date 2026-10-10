"""SENTETİK akış üretici: YALNIZCA altyapı testleri içindir.

Buradaki sayılar gerçek InSDN/SDN-MG25 verisini temsil etmez. Bu verilerle elde
edilen hiçbir metrik rapora "sonuç" olarak girmemelidir. Güvenlik önlemleri:
  * df.attrs["data_origin"] = "synthetic"
  * kaynak dosya adları "SYNTHETIC_" ile başlar
  * require_real_data(df) bu verilerde hata verir.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "flow_duration",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "totlen_fwd_pkts",
    "totlen_bwd_pkts",
    "fwd_pkt_len_mean",
    "flow_byts_s",
    "flow_pkts_s",
    "fwd_iat_mean",
    "syn_flag_cnt",
    "pkt_len_mean",
    "protocol",
]

# Gerçek CICFlowMeter benzeri başlıklar (zip yazarken kullanılır).
RAW_NAMES = {
    "flow_id": "Flow ID",
    "src_ip": "Src IP",
    "src_port": "Src Port",
    "dst_ip": "Dst IP",
    "dst_port": "Dst Port",
    "protocol": "Protocol",
    "timestamp": "Timestamp",
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Tot Fwd Pkts",
    "tot_bwd_pkts": "Tot Bwd Pkts",
    "totlen_fwd_pkts": "TotLen Fwd Pkts",
    "totlen_bwd_pkts": "TotLen Bwd Pkts",
    "fwd_pkt_len_mean": "Fwd Pkt Len Mean",
    "flow_byts_s": "Flow Byts/s",
    "flow_pkts_s": "Flow Pkts/s",
    "fwd_iat_mean": "Fwd IAT Mean",
    "syn_flag_cnt": "SYN Flag Cnt",
    "pkt_len_mean": "Pkt Len Mean",
    "label": "Label",
}

# sınıf -> (kayıt sayısı, oturum sayısı, log-ortalama kaydırması, kaynak dosya, ham etiket)
_DEFAULT_CLASSES = {
    "Normal": (3000, 30, 0.0, "SYNTHETIC_Normal_data.csv", "Normal"),
    "DoS": (600, 12, 0.7, "SYNTHETIC_OVS.csv", "DoS"),
    "Probe": (400, 10, -0.6, "SYNTHETIC_OVS.csv", "Probe"),
    "BOTNET": (40, 4, 0.4, "SYNTHETIC_OVS.csv", "Botnet"),  # bilerek nadir sınıf
}


class SyntheticDataError(RuntimeError):
    """Sentetik veri, gerçek sonuç üretilen bir yola sokulmaya çalışıldı."""


def is_synthetic(df: pd.DataFrame) -> bool:
    if df.attrs.get("data_origin") == "synthetic":
        return True
    if "source_file" in df.columns:
        return bool(df["source_file"].astype(str).str.upper().str.contains("SYNTHETIC").any())
    return False


def require_real_data(df: pd.DataFrame) -> None:
    if is_synthetic(df):
        raise SyntheticDataError(
            "Bu veri sentetiktir ve yalnızca altyapı testi içindir; sonuç üretiminde kullanılamaz."
        )


def make_synthetic_flows(
    seed: int = 0,
    *,
    classes: dict | None = None,
    add_defects: bool = True,
    leaky_feature: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = classes or _DEFAULT_CLASSES
    base_time = pd.Timestamp("2026-01-01 08:00:00")
    parts: list[pd.DataFrame] = []
    session_counter: dict[str, int] = {}

    for cls, (n, n_sessions, shift, fname, _raw) in classes.items():
        sizes = rng.multinomial(n, np.ones(n_sessions) / n_sessions)
        for k, size in enumerate(sizes):
            if size == 0:
                continue
            # DoS ve Probe oturumlarının yarısı "ikinci kurban" dosyasında (gerçek InSDN'deki
            # metasploitable-2.csv gibi); böylece koşular arası bölme sınanabilir.
            sess_file = "SYNTHETIC_Meta.csv" if (cls in ("DoS", "Probe") and k % 2 == 1) else fname
            sidx = session_counter.get(sess_file, 0)
            session_counter[sess_file] = sidx + 1
            offset = rng.normal(0, 0.3, size=len(FEATURES) - 2)  # oturuma özgü kayma
            start = base_time + pd.Timedelta(seconds=3600 * sidx)
            ts = start + pd.to_timedelta(rng.uniform(0, 250, size=size), unit="s")
            mu = shift + offset
            feats = np.exp(rng.normal(mu, 1.0, size=(size, len(FEATURES) - 2)) + 3.0)
            attacker = cls != "Normal"
            src = f"192.168.{20 if attacker else 10}.{rng.integers(2, 250)}"
            dst_port = rng.choice([80, 443, 53, 22] if not attacker else [80, 22, 21, 8080], size=size)
            src_port = rng.integers(1024, 65535, size=size)
            proto = rng.choice([6, 17, 1], size=size, p=[0.8, 0.15, 0.05] if not attacker else [0.6, 0.2, 0.2])
            flow_ids = [f"192.168.1.10-{src}-{dp}-{sp}-{pr}" for dp, sp, pr in zip(dst_port, src_port, proto)]
            # bağlantı parçalanması: bazı akışlar aynı flow_id'yi paylaşır
            for i in rng.choice(size, size=max(1, size // 30), replace=False):
                flow_ids[i] = flow_ids[int(rng.integers(0, size))]
            frame = pd.DataFrame(
                {
                    "flow_id": flow_ids,
                    "src_ip": src,
                    "src_port": src_port,
                    "dst_ip": "192.168.1.10",
                    "dst_port": dst_port,
                    "timestamp": ts,
                    "flow_duration": feats[:, 0],
                    "tot_fwd_pkts": np.round(feats[:, 1]),
                    "tot_bwd_pkts": np.round(feats[:, 2]),
                    "totlen_fwd_pkts": feats[:, 3],
                    "totlen_bwd_pkts": feats[:, 4],
                    "fwd_pkt_len_mean": feats[:, 5],
                    "flow_byts_s": feats[:, 6],
                    "flow_pkts_s": feats[:, 7],
                    "fwd_iat_mean": feats[:, 8],
                    "syn_flag_cnt": rng.poisson(1.0 + shift, size=size),
                    "pkt_len_mean": feats[:, 9],
                    "protocol": proto,
                    "label": cls,
                    "source_file": sess_file,
                }
            )
            parts.append(frame)

    df = pd.concat(parts, ignore_index=True)
    if add_defects:
        k = max(1, len(df) // 200)
        df.loc[rng.choice(len(df), k, replace=False), "flow_byts_s"] = np.nan
        df.loc[rng.choice(len(df), k, replace=False), "flow_pkts_s"] = np.inf
        # birebir tekrarlar (farklı flow_id ile)
        dups = df.sample(n=max(2, len(df) // 100), random_state=seed).copy()
        dups["flow_id"] = [f"dup-{i}" for i in range(len(dups))]
        df = pd.concat([df, dups], ignore_index=True)
        # çelişkili etiketli tekrarlar
        conflicts = df[df["label"] == "Normal"].sample(n=5, random_state=seed).copy()
        conflicts["label"] = "DoS"
        conflicts["flow_id"] = [f"conf-{i}" for i in range(len(conflicts))]
        conflicts["source_file"] = "SYNTHETIC_OVS.csv"
        df = pd.concat([df, conflicts], ignore_index=True)
    df = df.sort_values(["source_file", "timestamp"], kind="stable").reset_index(drop=True)
    if leaky_feature:
        df["leaky_feature"] = (df["label"] != "Normal").astype(float)
    df.insert(0, "record_id", [f"{s}#{i}" for i, s in zip(df.groupby("source_file").cumcount(), df["source_file"])])
    df.attrs["data_origin"] = "synthetic"
    return df


def write_synthetic_insdn_zip(zip_path: str | Path, seed: int = 0) -> None:
    """InSDN'e benzer düzende (CICFlowMeter başlıkları, 'Infinity' dizgisi) sentetik zip yazar."""
    df = make_synthetic_flows(seed=seed)
    raw_label = {cls: spec[4] for cls, spec in _DEFAULT_CLASSES.items()}
    zp = Path(zip_path)
    zp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, sub in df.groupby("source_file"):
            out = sub.drop(columns=["record_id", "source_file"]).copy()
            out["label"] = out["label"].map(raw_label)
            # Gerçek InSDN dosyalarındaki karışıklığı taklit et: bir dosyada iki zaman biçimi.
            ts = out["timestamp"]
            minute_24h = ts.dt.strftime("%d/%m/%Y %H:%M")
            seconds_12h = ts.dt.strftime("%d/%m/%Y %I:%M:%S %p")
            out["timestamp"] = minute_24h.where(np.arange(len(out)) % 2 == 0, seconds_12h)
            for col in FEATURES:
                s = out[col].astype(object)
                out[col] = s.where(~np.isinf(out[col].astype(float)), "Infinity")
            out = out.rename(columns=RAW_NAMES)
            buf = io.StringIO()
            out.to_csv(buf, index=False, na_rep="NaN")
            zf.writestr(f"SYNTHETIC_InSDN/{fname}", buf.getvalue())
