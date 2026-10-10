# SDN Ağlarında Saldırı Tespiti: Klasik ML ve LLM Karşılaştırması

BSM-451 Ağ Programlama dersi proje ödevi (konu 3: SDN ağı, Mininet + POX/Floodlight/NOX).
Amaç: aynı ağ akış kayıtlarında klasik makine öğrenmesi modelleri ile LLM tabanlı sınıflandırmayı doğruluk,
yanlış alarm ve gecikme açısından adil biçimde karşılaştırmak; seçilen yöntemi Mininet'teki bir SDN ağına bağlamak.

> **Dürüstlük notu:** Model sonuçları yalnızca gerçek InSDN üzerinde ölçüldü (`outputs/models/`, `docs/asama2_klasik_modeller.md`).
> Birim testler sentetik veriyle yalnızca altyapıyı sınar; sentetik veriden hiçbir sonuç rapora girmez.
> Literatür rakamları `docs/kaynaklar.md` içinde, bizim ölçümlerimizden ayrı tutulur. LLM deneyi (Aşama 3) henüz yapılmadı.

## Durum (2026-10-10)

| Aşama | Durum |
|---|---|
| 0. Ortam ve kaynak doğrulama | Tamamlandı (`docs/kaynaklar.md`, `docs/ortam_notlari.md`) |
| 1. Şema, veri inceleme, grup bazlı bölme, sızıntı testleri | Tamamlandı (testler: toplam 133 birim test, sentetik veriyle) |
| 1b. Gerçek InSDN üzerinde inceleme ve bölme | Tamamlandı: 343.889 satır okundu, tekrarlar indirgendi (184.667 benzersiz), iki bölme tasarımı üretildi (`docs/veri_sozlugu.md`, `docs/ozellik_ve_sizinti_kararlari.md`) |
| 2. Klasik modeller + metrikler | Tamamlandı (XGBoost hariç: kod hazır, geliştirme ortamında paket kurulamadı). Dummy, LR, RF, Isolation Forest; iki bölme tasarımı × iki özellik kümesi; test bir kez açıldı (`docs/asama2_klasik_modeller.md`, `notebooks/02_klasik_modeller.ipynb`) |
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
  metrics.py         metrikler, küme bootstrap güven aralığı, gecikme/verim/bellek ölçümü
  modeling.py        model kayıt defteri, özellik kümeleri, doğrulamada ayar ve eşik seçimi
  experiment.py      deney koşucusu (test yalnızca --evaluate-test ile açılır, erişim kaydı tutulur)
  reporting.py       sonuçları tabloya çevirir
  plots.py           rapor grafikleri
scripts/             00 ortam, 01 veri inceleme, 02 bölme, 03 eğitim/değerlendirme, 04 grafikler
notebooks/           adım adım açıklamalı notebook'lar (çıktısız tutulur)
figures/             rapor grafikleri (04_make_figures.py üretir)
tests/               unittest tabanlı testler
docs/                kaynaklar, veri sözlüğü, sızıntı kararları, Aşama 2 yöntemi, ortam notları
```

## Hızlı başlangıç

```bash
pip install -r requirements.txt
make test                                   # 133 test; sentetik veri kullanır (yaklaşık 1 dakika)

# InSDN CSV klasörünü (veya zip'ini) resmî adresten indirdikten sonra:
make env
make audit DATA=/yol/InSDN_DatasetCSV       # outputs/audit/{schema.json,audit.json,audit.md}
make split DATA=/yol/InSDN_DatasetCSV       # outputs/dedupe/, outputs/split_within_run/, outputs/split_cross_session/
make train DATA=/yol/InSDN_DatasetCSV       # yalnızca doğrulama (test açılmaz)
make test-eval DATA=/yol/InSDN_DatasetCSV   # ayarlar dondurulduktan sonra testi BİR KEZ açar
make figures                                # figures/*.png
```

Notebook'lar: `notebooks/02_klasik_modeller.ipynb` dosyasını VS Code'da (Python + Jupyter eklentisi) ya da
`jupyter lab` ile açın. Veri klasörünü `INSDN_DIR` ortam değişkeniyle veya ilk kod hücresinde verin.

Üretilen raporlar (`outputs/**/*.json|md`) depoda tutulur; kayıt bazlı büyük CSV'ler (`split_assignment.csv`, `record_weights.csv`) tutulmaz,
`make split` ile yeniden üretilir (aynı tohum ve aynı veri → aynı bölme; `split_meta.json` içindeki SHA-256 özetleriyle doğrulanır).
`outputs/environment.json` makineye özgüdür: kendi makinenizde `make env` ile üretin.

InSDN lisansı CC BY-NC 4.0'dır (ticari olmayan kullanım, atıf gerekir). Veri dosyaları depoya eklenmez (`.gitignore`).

## Temel kurallar (kodda zorlanır)

- Etiket, dosya adı, IP, zaman damgası ve `flow_id` model girdisine giremez (`ForbiddenColumnError`).
- Eksik özellik sıfırla doldurulmaz; sütun eksik veya farklı sıradaysa hata verilir (`SchemaMismatchError`).
- Birebir aynı özellik vektörleri bölmeden önce indirgenir; aynı bağlantının parçaları (aynı dosya + `flow_id` + ≤300 sn) aynı bölmeye gider.
- Ön işleme, eşik ve prompt seçimi eğitim/doğrulama bölümünde yapılır; test bölümü sabittir (SHA-256 ile doğrulanır).
- Test matrisi yalnızca `--evaluate-test` ile oluşturulur; her erişim `test_access_log.json` dosyasına yazılır.
- Güven aralıkları kayıt değil zaman bloğu kümeleri yeniden örneklenerek hesaplanır (ilişkili akışlar).
- Sentetik veri, `--allow-synthetic` bayrağı olmadan betiklerce reddedilir.
