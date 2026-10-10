# Aşama 2: Klasik modeller, metrikler ve ölçüm yöntemi

Bu belge Aşama 2'de **nasıl** ölçtüğümüzü ve **ne bulduğumuzu** anlatır. Tam sayılar `outputs/models/summary_test.md` dosyasında ve
`notebooks/02_klasik_modeller.ipynb` içinde, grafikler `figures/` klasöründedir. Bulgular ve yorum en alttadır.

## Karşılaştırılan yöntemler ve eğitimde gördükleri bilgi

| Yöntem | Eğitimde ne görür? | Ön işleme | Rastgelelik |
|---|---|---|---|
| DummyClassifier | Eğitimdeki çoğunluk sınıfı | yok | yok (1 koşu) |
| Logistic Regression | Tüm eğitim kayıtları + ikili etiket | medyan doldurma, işaretli log1p, standartlaştırma | yok (1 koşu) |
| Random Forest (150 ağaç) | Tüm eğitim kayıtları + ikili etiket | medyan doldurma | 5 tohum |
| XGBoost (300 ağaç) | Tüm eğitim kayıtları + ikili etiket | medyan doldurma | 5 tohum |
| Isolation Forest (200 ağaç) | **Yalnızca eğitimdeki Normal kayıtlar**, etiket yok | medyan doldurma (yalnızca Normal'den öğrenilir) | 5 tohum |

**Isolation Forest farklıdır:** saldırı örneği ve etiketi hiç görmez. Hiperparametresi ve `val_f1` eşiği doğrulama
etiketleriyle seçilir (yarı denetimli kolaylık). `train_normal_q99` eşiği ise hiç etiket kullanmaz; gerçek denetimsiz
kullanım budur. Sonuçlar bu fark belirtilerek okunmalıdır.

**XGBoost:** kod hazır ve testli. Geliştirme ortamında paket kurulamadığı için bu koşuda "atlandı" olarak kaydedildi
(`status.json`). Kendi makinenizde `pip install xgboost` sonrası
`python scripts/03_train_eval.py --data <veri> --models xgboost --evaluate-test` ile eklenir.

## Özellik kümeleri

* `full`: şemadaki 77 akış özelliği (IP, port, zaman damgası, flow_id, dosya adı **asla** girmez).
* `llm_subset`: 20 özellik (`src/sdnids/modeling.py: LLM_SUBSET_FEATURES`). Aşama 3'te LLM'e **aynı 20 özellik** verilecek;
  klasik modeller her iki kümeyle de çalıştırılır. Liste alan bilgisiyle seçildi (hacim, hız, paket boyu, zamanlama,
  TCP bayrakları), veriden veya test sonucundan seçilmedi. Hiçbiri sabit özellik değildir.

## Seçim yöntemi (tamamı doğrulama verisinde)

| Model | Izgara | Seçim ölçütü |
|---|---|---|
| Logistic Regression | C ∈ {0,01; 0,1; 1; 10} × class_weight ∈ {yok, dengeli} | doğrulama PR-AUC |
| Random Forest | max_depth ∈ {12, sınırsız} × min_samples_leaf ∈ {1, 5} | doğrulama PR-AUC |
| XGBoost | max_depth ∈ {4, 8} × learning_rate ∈ {0,1; 0,3} | doğrulama PR-AUC |
| Isolation Forest | max_samples ∈ {256, 2048} | doğrulama PR-AUC |

Izgaralar ve eşik kuralları test açılmadan önce koda yazıldı. Test sonucuna bakarak değiştirilmedi.

**Eşik kuralları** (her model için önceden bildirilir, ilki birincil):

| Kural | Anlamı | Hangi modeller |
|---|---|---|
| `native` | modelin kendi kararı | Dummy |
| `p0.5` | olasılık ≥ 0,5, ayar yok | LR, RF, XGBoost (birincil) |
| `val_f1` | doğrulamada saldırı F1'ini en büyükleyen eşik | LR, RF, XGBoost, IF |
| `train_normal_q99` | eğitim Normal skorlarının %99'luk dilimi (etiketsiz) | IF (birincil) |

## Test protokolü

1. Ayar ve eşik seçen fonksiyonlar test verisi almaz (`tune_hyperparameters`, `fit_model`, `select_threshold_f1`).
2. `--evaluate-test` verilmeden çalıştırıldığında test matrisi **hiç oluşturulmaz** (testle doğrulanır).
3. Test her açıldığında `outputs/models/<tasarım>/test_access_log.json` dosyasına model, ayar ve test kümesi özeti yazılır.
4. **Negatif kontrol testi:** test etiketleri rastgele karıştırıldığında seçilen ayarlar, eşikler ve doğrulama sonuçları
   birebir aynı kalır, yalnızca test sonucu değişir (`tests/test_experiment.py`).
5. Bu çalışmada test, hızlı bir yalnız-doğrulama denemesinden (küçük modeller, hata ayıklama amaçlı) sonra **bir kez** açıldı.

## Metrikler

Pozitif sınıf **saldırı**dır. TN, FP, FN, TP confusion matrix hücreleridir.

| Metrik | Formül | Ne anlatır? |
|---|---|---|
| Accuracy | (TP+TN) / toplam | genel doğruluk; dengesiz veride yanıltıcı olabilir |
| Saldırı precision | TP / (TP+FP) | "saldırı" dediklerimizin ne kadarı gerçekten saldırı |
| Saldırı recall | TP / (TP+FN) | saldırıların ne kadarını yakaladık |
| Saldırı F1 | 2TP / (2TP+FP+FN) | precision ve recall dengesi |
| FPR (yanlış alarm) | FP / (FP+TN) | normal trafiğin ne kadarına alarm verdik |
| FNR (kaçırma) | FN / (FN+TP) | saldırıların ne kadarını kaçırdık |
| Macro-F1 | iki sınıfın F1 ortalaması | iki sınıfa eşit ağırlık |
| Weighted-F1 | F1'lerin sınıf büyüklüğüyle ağırlıklı ortalaması | |
| PR-AUC | `sklearn.metrics.average_precision_score` (basamaklı toplam, interpolasyonsuz) | eşikten bağımsız sıralama başarısı |

Payda sıfırsa değer NaN'dır; sessizce 0 yazılmaz. Sınıf başına tabloda Normal için özgüllük (1 − FPR), saldırı
sınıfları için recall verilir; 30'dan az örnekli sınıflar `güvenilir = False` olarak işaretlenir.

LLM'in kendi yazdığı güven yüzdesi (Aşama 3) ile klasik modellerin skorları kalibre olasılık sayılmaz; PR-AUC yalnızca
sıralama için kullanılır.

## Belirsizlik: tohumlar ve güven aralıkları

* **Tohumlar arası sapma (`_std`):** rastgelelik içeren modeller 5 tohumla yeniden eğitildi. Bu, *eğitimin*
  rastgeleliğini gösterir; test örneklemi belirsizliğini göstermez.
* **%95 güven aralığı (GA):** birincil tohumun test tahminleri üzerinde **küme bootstrap** (300 tekrar).
  Bağımsız birim olarak kayıt değil **aynı dosya + 5 dakikalık zaman bloğu** kullanıldı; zamanda yakın akışlar
  (aynı koşu/oturum) ilişkili olduğundan kayıtları bağımsız saymak aralığı yapay olarak daraltırdı. İkinci bir aralık
  sızıntı gruplarıyla (`ci_leak_group`) da hesaplanır; genelde daha dardır. 5 dakikalık blok da gerçek oturum sınırını
  bilmediğimiz için bir yaklaşımdır; aralıklar hâlâ iyimser olabilir.
* Aralıkları örtüşen modeller arasındaki küçük farklar **üstünlük** olarak yorumlanmamalıdır.

## Hız ve kaynak ölçümü

| Ölçüm | Kapsam |
|---|---|
| `gecikme_p50_ms`, `gecikme_p95_ms` | tek kayıt (1 × F dizi): sonsuz→eksik, doldurma, (varsa) log1p/ölçekleme, model. Özellik çıkarma, serileştirme, ağ **dahil değil** |
| `gecikme_hizalamalı_p95_ms` | aynısı + tek satırlık tablodan şema denetimi ve sütun sıralama |
| `akış_per_sn` | 1000'lik parçalarla toplu tahmin verimi |
| `eğitim_sn_ort` | tohum başına eğitim süresi (seçilen ayarla) |
| `model_mb` | modelin bellekteki serileştirilmiş boyutu |
| `eğitim_ram_artışı_mb` | eğitim sırasında süreç belleğindeki artış (tohum ortalaması). **Zayıf bir ölçüdür:** süreç önceki eğitimlerde ayırdığı belleği yeniden kullandığı için çoğu zaman 0'a yakın çıkar. Bellek için asıl gösterge `model_mb`'dir. Tohum başına öncesi/sonrası ve süreç tepe belleği `selection.json` içindedir |

Gecikme **doğrulama kayıtlarıyla** ve tek iş parçacığıyla (`n_jobs=1`) ölçüldü: denetleyiciye gömülü, tek tek akış
işleyen bir kullanımı taklit eder. Ölçümler **2 vCPU'lu bir bulut kapsayıcısında** yapıldı; mutlak süreler makineye
bağlıdır, modeller arası oran daha anlamlıdır. GPU kullanılmadı (VRAM yok). Uçtan uca gecikme (trafikten alarma)
Aşama 4'te Mininet üzerinde ayrıca ölçülecek.

## İkincil analiz: kopya ağırlıklı metrikler

Ana metrikler tekrarları indirgenmiş (benzersiz) kayıtlar üzerindedir. `weighted_by_n_copies`, her kaydı ham veride
kaç kopyası varsa o kadar sayar: "ham veri dağılımında" ne olurdu sorusunun cevabıdır. DDoS'un %98,8'i kopya
olduğundan bu ağırlıklı sonuçlar DDoS'a çok fazla ağırlık verir; yalnızca karşılaştırma için raporlanır.

## Yeniden üretim

```bash
python scripts/01_audit_data.py --data /yol/InSDN_DatasetCSV
python scripts/02_make_split.py --data /yol/InSDN_DatasetCSV
python scripts/03_train_eval.py --data /yol/InSDN_DatasetCSV                  # yalnızca doğrulama
python scripts/03_train_eval.py --data /yol/InSDN_DatasetCSV --evaluate-test  # testi bir kez aç
python scripts/04_make_figures.py
```

Tohum 42; kütüphane sürümleri ve donanım her koşunun `run_meta.json` dosyasındadır.

## Bulgular (gerçek InSDN, test kümesi, ölçüm tarihi 2026-10-10)

Aşağıdaki değerler **bizim ölçümümüzdür**. Her model için birincil eşik kuralı verilmiştir. "±" 5 tohum arası standart
sapmadır, köşeli parantez birincil tohumun %95 küme bootstrap GA'sıdır. Tam tablo: `outputs/models/summary_test.md`.
Test kümesi within_run'da 36.931 kayıt (169 zaman bloğu), cross_session'da 60.354 kayıttır (%79,6 saldırı, 100 zaman bloğu).
Küme sayısı az olduğu için cross_session güven aralıkları geniştir.

### within_run (aynı yakalamalar; iyimser)

| Model | Özellik | Saldırı F1 | FPR | Macro-F1 |
|---|---|---|---|---|
| Dummy | – | 0,790 | 1,000 | 0,395 |
| Logistic Regression | 77 | 0,9985 [0,9966; 0,9993] | 0,0013 | 0,998 |
| Logistic Regression | 20 | 0,9914 | 0,0175 | 0,988 |
| Random Forest | 77 | 0,9998 [0,9994; 1,0000] | 0,0005 | 1,000 |
| Random Forest | 20 | 0,9998 | 0,0005 | 1,000 |
| Isolation Forest (etiketsiz eşik) | 77 | 0,004 | 0,005 | 0,259 |
| Isolation Forest (etiketsiz eşik) | 20 | 0,153 | 0,006 | 0,344 |

### cross_session (görülmemiş oturum; genelleme)

| Model | Özellik | Saldırı F1 | Recall | FPR | Macro-F1 |
|---|---|---|---|---|---|
| Dummy | – | 0,886 | 1,000 | 1,000 | 0,443 |
| Logistic Regression | 77 | 0,936 [0,833; 0,986] | 0,880 | 0,0023 [0,0012; 0,0036] | 0,872 |
| Logistic Regression | 20 | 0,869 | 0,770 | 0,0097 | 0,777 |
| Random Forest | 77 | 0,994 ± 0,002 [0,856; 0,997] | 0,988 | 0,0008 [0,0002; 0,0017] | 0,985 |
| Random Forest | 20 | 0,948 ± 0,033 [0,939; 0,992] | 0,903 | 0,0010 | 0,897 |
| Isolation Forest (etiketsiz eşik) | 77 | 0,005 | 0,002 | 0,011 | 0,170 |
| Isolation Forest (etiketsiz eşik) | 20 | 0,536 | 0,367 | 0,010 | 0,490 |

Sınıf bazında (cross_session, Random Forest, 77 özellik): Probe %99,5, BFA %100, DDoS %84,0, DoS %77,9 yakalandı;
Normal'in %99,9'u doğru sınıflandı. Logistic Regression DDoS (%97,6) ve DoS'ta (%83,8) daha iyi, Probe'da (%87,7) daha
kötü. Yani sınıf bazında tek bir kazanan yok. U2R'nin 16/17'si yakalandı, ama 17 kayıtla güvenilir ölçüm iddia edilemez;
üstelik bu sınıf eğitimde hiç yoktu.

Hız (2 vCPU bulut kapsayıcısı, tek iş parçacığı, tek kayıt p95): Logistic Regression ≈ 1,4–1,8 ms, Random Forest ≈ 13–21 ms,
Isolation Forest ≈ 29–36 ms. Toplu verim: LR ≈ 0,3–0,6 milyon akış/sn, RF ≈ 44–50 bin akış/sn. Model boyutu: LR ≈ 6 KB,
RF ≈ 3,4–4,2 MB. Tohum başına eğitim: LR 1,5–7,5 sn, RF 8–22 sn.

### Yorum (yalnızca klasik modeller; LLM henüz ölçülmedi)

1. **within_run neredeyse kusursuz, ama bu bir tavan değil bir uyarıdır.** Aynı yakalamalardan gelen test, Random Forest'ı
   0,9998 F1'e taşıyor. Normal ve saldırı farklı yakalamalardan geldiği için bu kadar kolay ayrışma, kısmen "yakalama
   imzası" öğrenildiğini düşündürür (bkz. sızıntı kararları belgesi).
2. **Genelleme (araştırma sorusu 5):** Görülmemiş oturumda başarı düşüyor; en çok Logistic Regression'da (recall 0,998 →
   0,880) ve 20 özellikli kümede (RF recall 0,9998 → 0,903). Random Forest + 77 özellik en iyi korunan kombinasyon.
   Güven aralıkları geniş olduğu için LR ile RF arasındaki farkın büyüklüğü belirsizdir. Yön tutarlı olsa da kesin büyüklük
   iddia edilmemelidir.
3. **Yanlış alarm (araştırma sorusu 2):** En düşük FPR Random Forest'ta (cross_session ≈ %0,1), ardından LR (77 özellik,
   ≈ %0,2). Dummy her normal akışa alarm verir (FPR = 1) ama saldırı oranı yüksek olduğu için saldırı F1'i 0,886'dır.
   **Saldırı F1'i tek başına yanıltıcıdır**; FPR ve macro-F1 ile birlikte okunmalıdır.
4. **Doğrulamada ayarlanan eşik yeni oturuma taşınmıyor.** cross_session'da RF'nin `val_f1` eşiği (≈ 0,59–0,67) F1'i
   0,708 ± 0,212'ye düşürdü (`p0.5` ile 0,994). Kaynak alanda sınıflar o kadar net ayrışıyor ki eşik yükseğe kayıyor;
   yeni oturumdaki saldırılar daha düşük olasılık alıyor. Canlı Mininet deneyinde eşik seçimine bu yüzden dikkat edilmeli.
5. **Isolation Forest (denetimsiz) bu veride işe yaramıyor.** Etiket kullanmayan eşikle saldırıların neredeyse tamamını
   kaçırıyor (77 özellik). Saldırı akışları (DoS/Probe) kısa ve tekdüze, normal trafik çok çeşitli; "olağandışı = saldırı"
   varsayımı burada tutmuyor. Doğrulama etiketleriyle eşik seçildiğinde recall yükseliyor ama FPR %23–59'a çıkıyor; alarm
   sistemi için kabul edilemez.
6. **Hız (araştırma sorusu 4):** Tamamlanmış akış başına tespit için üç model de milisaniyeler mertebesinde. LR yaklaşık 10 kat
   hızlı ve çok küçük; RF daha isabetli. Akışlar toplu işlenirse RF de saniyede on binlerce akışa yetişir.
7. **Nadir sınıflar (araştırma sorusu 3):** Bu veriyle güvenilir ölçülemiyor. within_run testinde BOTNET 32, Web-Attack 36,
   U2R 3 kayıt; cross_session testinde BOTNET ve Web-Attack yok.
