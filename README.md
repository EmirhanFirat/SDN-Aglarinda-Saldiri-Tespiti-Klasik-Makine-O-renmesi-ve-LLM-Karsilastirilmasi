# SDN Ağlarında Saldırı Tespiti: Klasik ML ve LLM Karşılaştırması

BSM-451 Ağ Programlama dersi proje ödevi (konu 3: SDN ağı, Mininet + POX/Floodlight/NOX).
Amaç: aynı ağ akış kayıtlarında klasik makine öğrenmesi modelleri ile LLM tabanlı sınıflandırmayı doğruluk,
yanlış alarm ve gecikme açısından adil biçimde karşılaştırmak; seçilen yöntemi Mininet'teki bir SDN ağına bağlamak.

> **Dürüstlük notu:** Henüz hiçbir model sonucu (accuracy, F1 vb.) ölçülmedi. Gerçek InSDN üzerinde yalnızca veri
> incelemesi ve bölme yapıldı (`docs/veri_sozlugu.md`). Birim testler sentetik veriyle yalnızca altyapıyı sınar.
> Literatür rakamları `docs/kaynaklar.md` içinde, bizim ölçümlerimizden ayrı tutulur.

## Durum (2026-10-10)

| Aşama | Durum |
|---|---|
| 0. Ortam ve kaynak doğrulama | Tamamlandı (`docs/kaynaklar.md`, `docs/ortam_notlari.md`) |
| 1. Şema, veri inceleme, grup bazlı bölme, sızıntı testleri | Tamamlandı: 74 birim test geçiyor (sentetik veriyle) |
| 1b. Gerçek InSDN üzerinde inceleme ve bölme | Tamamlandı: 343.889 satır okundu, tekrarlar indirgendi (184.667 benzersiz), iki bölme tasarımı üretildi (`docs/veri_sozlugu.md`, `docs/ozellik_ve_sizinti_kararlari.md`) |
| 2. Klasik modeller + metrikler | Henüz yok |
| 3. LLM değerlendirme modülü | Henüz yok (ücretli API; onay gerekir) |
| 4. Mininet + kontrolör + canlı hat | Henüz yok (geliştirme ortamında Mininet kurulamıyor) |
| 5. Grafikler, UML sıralama diyagramı, Türkçe rapor | Henüz yok |
| SDN-MG25 | Erişilemiyor (IEEE DataPort aboneliği gerekir); InSDN ile devam |

## Klasör yapısı

```
src/sdnids/
  schema.py          özellik şeması; eğitim ve canlı hatta aynı sütunlar, aynı sıra
  config.py          deney ayarları (configs/experiment.json)
  env_record.py      sürümler, donanım, tohum kaydı
  data/
    loader.py        InSDN klasör/zip okuyucu, zaman damgası ayrıştırma (gün-önce)
    labels.py        etiket kanonikleştirme, ikili etiket
    audit.py         eksik/sonsuz, tekrar, şüpheli etiket, sızıntı işaretleri
    dedupe.py        birebir aynı özellik vektörlerini indirger, çelişkili etiketleri dışlar
    split.py         grup bütünlüğünü koruyan bölmeler (within_run, cross_session)
    preprocess.py    yalnızca eğitimde öğrenen ön işleme
    synthetic.py     SENTETİK veri (yalnızca altyapı testi)
scripts/             00 ortam kaydı, 01 veri inceleme, 02 bölme
tests/               unittest tabanlı testler
docs/                kaynaklar, veri sözlüğü, özellik/sızıntı kararları, ortam notları
```

## Hızlı başlangıç

```bash
pip install -r requirements.txt
make test                                   # 74 test; sentetik veri kullanır

# InSDN CSV klasörünü (veya zip'ini) resmî adresten indirdikten sonra:
make env
make audit DATA=/yol/InSDN_DatasetCSV       # outputs/audit/{schema.json,audit.json,audit.md}
make split DATA=/yol/InSDN_DatasetCSV       # outputs/dedupe/, outputs/split_within_run/, outputs/split_cross_session/
```

InSDN lisansı CC BY-NC 4.0'dır (ticari olmayan kullanım, atıf gerekir). Veri dosyaları depoya eklenmez (`.gitignore`).

## Temel kurallar (kodda zorlanır)

- Etiket, dosya adı, IP, zaman damgası ve `flow_id` model girdisine giremez (`ForbiddenColumnError`).
- Eksik özellik sıfırla doldurulmaz; sütun eksik veya farklı sıradaysa hata verilir (`SchemaMismatchError`).
- Birebir aynı özellik vektörleri bölmeden önce indirgenir; aynı bağlantının parçaları (aynı dosya + `flow_id` + ≤300 sn) aynı bölmeye gider.
- Ön işleme, eşik ve prompt seçimi eğitim/doğrulama bölümünde yapılır; test bölümü sabittir (SHA-256 ile doğrulanır).
- Sentetik veri, `--allow-synthetic` bayrağı olmadan betiklerce reddedilir.
