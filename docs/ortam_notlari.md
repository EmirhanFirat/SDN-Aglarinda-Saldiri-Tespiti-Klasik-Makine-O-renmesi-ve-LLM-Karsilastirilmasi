# Ortam notları

## Geliştirme oturumu (bu kodun yazıldığı ve test edildiği ortam)

Ubuntu 24.04 (Docker), 2 CPU, 7,8 GB RAM, GPU yok, `/dev/kvm` yok, Python 3.13.
Test edilen sürümler: numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, psutil 7.2.2, joblib 1.6.0.

Bu oturumda **yapılamayanlar** (çıkış politikası veya eksik bileşen; aşılmadı):

| Engel | Sonuç |
|---|---|
| `aseados.ucd.ie` 403 | InSDN dosyaları indirilemedi |
| `pypi.org` 403, `apt` kapalı | `xgboost`, `pytest`, `scapy` kurulamadı; testler standart `unittest` ile yazıldı |
| Mininet, Open vSwitch, POX, `iproute2` yok | Mininet topolojisi bu ortamda **çalıştırılamaz**; kullanıcının Ubuntu sanal makinesinde çalıştırılacak şekilde hazırlanacak |
| GPU yok | Yerel LLM çalıştırılamaz; LLM karşılaştırması bir API gerektirir (ücretli; onay olmadan başlatılmaz) |

## Hedef ortam (öneri)

Mininet için Ubuntu 22.04/24.04 sanal makinesi (VirtualBox veya VMware), root yetkisi, en az 4 GB RAM.
Kontrolör ve Open vSwitch o makinede kurulacak; kurulum kılavuzu Aşama 4'te eklenecek.

## Test çalıştırma

```bash
cd sdn-ids
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
