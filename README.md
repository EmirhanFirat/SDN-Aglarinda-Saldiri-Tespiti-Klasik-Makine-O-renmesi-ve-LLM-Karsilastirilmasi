# SDN Ağlarında Saldırı Tespiti: Klasik ML ve LLM Karşılaştırması

BSM-451 Ağ Programlama dersi proje ödevi (konu 3: SDN ağı, Mininet + POX/Floodlight/NOX).
Amaç: aynı ağ akış kayıtlarında klasik makine öğrenmesi modelleri ile LLM tabanlı sınıflandırmayı doğruluk,
yanlış alarm ve gecikme açısından adil biçimde karşılaştırmak; seçilen yöntemi Mininet'teki bir SDN ağına bağlamak.

> **Dürüstlük notu:** Bu depodaki hiçbir sonuç gerçek veri üzerinde henüz ölçülmedi. Kod ve testler sentetik
> veriyle yalnızca altyapıyı sınar. Literatür rakamları `docs/kaynaklar.md` içinde, bizim ölçümlerimizden ayrı tutulur.

## Durum (2026-10-10)

| Aşama | Durum |
|---|---|
| 0. Ortam ve kaynak doğrulama | Tamamlandı (`docs/kaynaklar.md`, `docs/ortam_notlari.md`) |
| 1. Şema, veri inceleme, grup bazlı bölme, sızıntı testleri | Yazıldı, 46 birim test geçiyor (sentetik veriyle) |
| 1b. Gerçek InSDN üzerinde inceleme ve bölme | **Bekliyor:** `InSDN_DatasetCSV.zip` dosyası gerekli |
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
    loader.py        InSDN zip okuyucu (CICFlowMeter başlıkları varsayımıyla)
    labels.py        etiket kanonikleştirme, ikili etiket
    audit.py         eksik/sonsuz, tekrar, şüpheli etiket, sızıntı işaretleri
    split.py         grup bütünlüğünü koruyan train/val/test bölmesi
    preprocess.py    yalnızca eğitimde öğrenen ön işleme
    synthetic.py     SENTETİK veri (yalnızca altyapı testi)
scripts/             00 ortam kaydı, 01 veri inceleme, 02 bölme
tests/               unittest tabanlı testler
docs/                kaynaklar, özellik/sızıntı kararları, ortam notları
```

## Hızlı başlangıç

```bash
pip install -r requirements.txt
make test                                   # 46 test; sentetik veri kullanır

# InSDN zip dosyasını resmî adresten indirdikten sonra:
make env
make audit ZIP=/yol/InSDN_DatasetCSV.zip    # outputs/audit/{schema.json,audit.json,audit.md}
make split ZIP=/yol/InSDN_DatasetCSV.zip    # outputs/split/{split_assignment.csv,split_meta.json,...}
```

InSDN lisansı CC BY-NC 4.0'dır (ticari olmayan kullanım, atıf gerekir). Veri dosyaları depoya eklenmez (`.gitignore`).

## Temel kurallar (kodda zorlanır)

- Etiket, dosya adı, IP, zaman damgası ve `flow_id` model girdisine giremez (`ForbiddenColumnError`).
- Eksik özellik sıfırla doldurulmaz; sütun eksik veya farklı sıradaysa hata verilir (`SchemaMismatchError`).
- Aynı bağlantı parçaları, birebir tekrar kayıtlar ve aynı zaman bloğu aynı bölmeye gider.
- Ön işleme, eşik ve prompt seçimi eğitim/doğrulama bölümünde yapılır; test bölümü sabittir (SHA-256 ile doğrulanır).
- Sentetik veri, `--allow-synthetic` bayrağı olmadan betiklerce reddedilir.
