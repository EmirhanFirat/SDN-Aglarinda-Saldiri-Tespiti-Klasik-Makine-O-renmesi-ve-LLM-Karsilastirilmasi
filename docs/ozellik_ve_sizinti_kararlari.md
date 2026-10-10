# Özellik ve sızıntı kararları

Bu belge, hangi sütunun neden modele girdiğini veya girmediğini açıklar. Kod karşılığı: `src/sdnids/schema.py`.

## Sütun grupları

| Grup | Sütunlar | Modele girer mi? | Neden |
|---|---|---|---|
| Etiket | `label`, `label_raw`, `binary_label` | Hayır | Cevabın kendisi |
| Defter tutma | `record_id`, `source_file`, `group_id`, `split`, `scenario_id` | Hayır | `source_file` dosya adı etiketi ele verir (ör. `Normal_data.csv` ⇒ Normal) |
| Kimlik | `flow_id`, `src_ip`, `dst_ip`, `timestamp` | Hayır | Aşağıya bakın; denetim ve bölme için saklanır |
| Portlar | `src_port`, `dst_port` | Hayır (varsayılan) | `src_port` geçici ve rastgele; `dst_port` servisi gösterdiği için meşru olabilir ama saldırı araçlarının hedef portları veri kümesine özgü bir iz bırakabilir. "dst_port ile / dst_port olmadan" karşılaştırması Aşama 2'de **yapılmadı**; isteğe bağlı ek deney olarak bekliyor. Şimdilik bütün modeller (ve ileride LLM) portsuz, aynı bilgiyle karşılaştırılıyor |
| Akış özellikleri | `protocol` dahil CICFlowMeter sayısal özellikleri | Evet | Davranışı tanımlar |

## Neden IP ve zaman damgası modele verilmiyor?

- **IP:** InSDN README'sine göre etiketler kaynak/hedef IP'ye göre verilmiş. IP'yi özellik yaparsak model saldırıyı değil etiketleme kuralını öğrenir. Mininet'teki canlı ağda adresleme farklı olacağından bu "başarı" aktarılamaz.
- **Zaman damgası ve flow_id:** koşu kimliği gibi davranır; zamanda yakın akışlar birbirine benzer. Bölme ve denetimde kullanılır, modele girmez.
- Bu karar değiştirilecekse ayrı bir ablation deneyi ve gerekçe gerekir.

## Veri bölme

### Önce tekrarları indirge

Gerçek InSDN'de kayıtların yaklaşık yarısı (343.889 → 184.667 benzersiz) birebir aynı özellik vektörüdür; en uç örnek DDoS:
121.942 satır yalnızca 1.520 benzersiz vektör. Bunları bölmeye dağıtmak, aynı kaydı hem eğitime hem teste koymak demektir.
Bu yüzden önce `deduplicate_vectors` ile aynı vektörler tek kayıtta toplanır (`n_copies` ağırlık olarak saklanır, **özellik olarak asla kullanılmaz**).
Aynı vektör iki farklı etiketle görülüyorsa (79 vektör, 207 satır) kayıtlar dışlanır ve `outputs/dedupe/excluded_conflicting_label.csv` içinde listelenir.

### Gruplama kuralları

1. **Bağlantı parçaları:** aynı kaynak dosya + aynı `flow_id` + başlangıçları en çok 300 sn arayla. CICFlowMeter uzun bağlantıyı zaman aşımıyla
   parçalar; parçalar aynı bağlantıdır. Zaman sınırı şart: haftalar sonra aynı 5'li ile görülen akış yeni bağlantıdır; sınırsız birleştirme
   gerçek veride tüm Normal trafiği tek dev gruba zincirliyordu (hata bulundu, düzeltildi, testi yazıldı).
2. **Birebir aynı özellik vektörü** aynı gruba gider (dedupe sonrası normalde kalmaz; savunma amaçlı).
3. **Zaman bloğu** (varsayılan kapalı): saldırı yakalamaları çok kısa olduğundan blok + tekrar kuralı birlikte dev gruplar üretti. Yalnızca `cross_session` tasarımında Normal dosyasına uygulanır.

Doğrulama (`verify_split_integrity`) bölme kodundan bağımsız olarak "grup bölünmüş mü, aynı vektör iki bölmede mi, ardışık parçalar ayrılmış mı" kontrol eder.

### İki bölme tasarımı

| | `within_run` | `cross_session` |
|---|---|---|
| Fikir | Gruplara göre rastgele %60/%20/%20 (sınıf bazında dengeli) | Test = hiç görülmemiş yakalama oturumu |
| Test | Eğitimle aynı yakalamalardan | `metasploitable-2.csv` tamamı + `Normal_data.csv`'nin kronolojik olarak son %20'si (7 Şubat 13:50–19:53) |
| Ne ölçer | Aynı ortam içinde ayırt edebilme (**iyimser** tahmin) | Başka oturuma genelleme (araştırma sorusu 5) |
| Zayıflığı | Aynı koşunun komşu akışları hâlâ birbirine benzer olabilir | Bazı sınıflar testte yok / eğitimde yok (aşağıda) |

`cross_session` sınıf kapsamı (benzersiz kayıtlar): BOTNET ve Web-Attack yalnızca `OVS.csv`'de bulunduğundan **testte yok** (ölçülemez);
U2R yalnızca `metasploitable-2.csv`'de bulunduğundan **eğitimde yok** ve testte 17 kayıt (< 30). DDoS'un eğitim/doğrulamadaki payı yalnızca 94 benzersiz vektör.
Bu uyarılar `split_meta.json` ve betik çıktısında otomatik üretilir; rapora aynen taşınacaktır.

### Kaçınılmaz sınırlılık: yakalama ortamı karışıklığı

Normal trafik ayrı bir yakalamadan (`Normal_data.csv`), saldırılar başka yakalamalardan gelir. Model "saldırıyı" değil "hangi yakalamadan geldiğini"
öğreniyor olabilir. Elimizdeki bilgiyle bunu ayıramayız; `cross_session` bu riski azaltır ama ortadan kaldırmaz. Rapor bu sınırlılığı açıkça yazacaktır.

### Zaman damgası güvenilirliği

Üç dosyada da birden fazla zaman damgası biçimi var (elektronik tabloda yeniden kaydedilmiş görünüyor). Gün/ay sırasını gün-önce olarak okuyoruz
(`5/2/2020` = 5 Şubat; dosya tarihleriyle tutarlı). Çoğu damga dakika hassasiyetinde; saniye düzeyinde sıralama yapılamaz. Bölme artık bu hassasiyetle
yetinecek biçimde (300 sn blok) tasarlandı.

### Kayıt ve değişmezlik

Bölme `split_assignment.csv` olarak kaydedilir, her bölmenin kayıt kimliklerinin SHA-256 özeti `split_meta.json` içinde tutulur; değiştirilirse `load_split` hata verir.
Test bölümünde 30'dan az kaydı olan sınıf için "güvenilir ölçüm iddia edilemez" uyarısı üretilir (`min_test_support`).

## Öğrenilenler yalnızca eğitimden

Eksik değer doldurma (medyan), ölçekleme ve (ileride) özellik seçimi/yeniden örnekleme yalnızca eğitim bölümünde öğrenilir.
Hiperparametre, eşik ve LLM prompt seçimi doğrulama bölümünde yapılır; test bölümü nihai değerlendirmeye kadar kullanılmaz.
Tek özellikli sızıntı denetimi (`single_feature_leak_check`) yalnızca eğitim bölümünde çalışır.

## Sentetik veri

`src/sdnids/data/synthetic.py` yalnızca altyapı testleri içindir. Betikler sentetik veriyi `--allow-synthetic` bayrağı olmadan reddeder;
sentetik veriyle üretilen hiçbir sayı rapora girmez.
