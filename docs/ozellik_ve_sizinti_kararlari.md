# Özellik ve sızıntı kararları

Bu belge, hangi sütunun neden modele girdiğini veya girmediğini açıklar. Kod karşılığı: `src/sdnids/schema.py`.

## Sütun grupları

| Grup | Sütunlar | Modele girer mi? | Neden |
|---|---|---|---|
| Etiket | `label`, `label_raw`, `binary_label` | Hayır | Cevabın kendisi |
| Defter tutma | `record_id`, `source_file`, `group_id`, `split`, `scenario_id` | Hayır | `source_file` dosya adı etiketi ele verir (ör. `Normal_data.csv` ⇒ Normal) |
| Kimlik | `flow_id`, `src_ip`, `dst_ip`, `timestamp` | Hayır | Aşağıya bakın; denetim ve bölme için saklanır |
| Portlar | `src_port`, `dst_port` | Hayır (varsayılan) | `src_port` geçici ve rastgele; `dst_port` servisi gösterdiği için meşru olabilir ama saldırı araçlarının hedef portları veri kümesine özgü bir iz bırakabilir. Aşama 2'de "dst_port ile / dst_port olmadan" karşılaştırması yapılacak |
| Akış özellikleri | `protocol` dahil CICFlowMeter sayısal özellikleri | Evet | Davranışı tanımlar |

## Neden IP ve zaman damgası modele verilmiyor?

- **IP:** InSDN README'sine göre etiketler kaynak/hedef IP'ye göre verilmiş. IP'yi özellik yaparsak model saldırıyı değil etiketleme kuralını öğrenir. Mininet'teki canlı ağda adresleme farklı olacağından bu "başarı" aktarılamaz.
- **Zaman damgası ve flow_id:** koşu kimliği gibi davranır; zamanda yakın akışlar birbirine benzer. Bölme ve denetimde kullanılır, modele girmez.
- Bu karar değiştirilecekse ayrı bir ablation deneyi ve gerekçe gerekir.

## Veri bölme

- Gruplama: aynı `flow_id`, birebir aynı özellik vektörü, aynı dosyada aynı zaman bloğu (varsayılan 300 sn).
- Grup bütünlüğü oranlardan önceliklidir; oranlar tutmazsa uyarı üretilir, grup bölünmez.
- **Sınırlılık:** InSDN'de senaryo/koşu kimliği olduğuna dair bilgi README'de yok; zaman bloğu yalnızca bir yaklaşımdır. Gerçek zaman damgaları gelince blok boyutu duyarlılığı raporlanacak.
- Bölme `split_assignment.csv` olarak kaydedilir, test kayıt kimliklerinin SHA-256 özeti `split_meta.json` içinde tutulur; değiştirilirse `load_split` hata verir.
- Test bölümünde 30'dan az kaydı olan sınıf için "güvenilir ölçüm iddia edilemez" uyarısı üretilir (`min_test_support`).

## Öğrenilenler yalnızca eğitimden

Eksik değer doldurma (medyan), ölçekleme ve (ileride) özellik seçimi/yeniden örnekleme yalnızca eğitim bölümünde öğrenilir.
Hiperparametre, eşik ve LLM prompt seçimi doğrulama bölümünde yapılır; test bölümü nihai değerlendirmeye kadar kullanılmaz.
Tek özellikli sızıntı denetimi (`single_feature_leak_check`) yalnızca eğitim bölümünde çalışır.

## Sentetik veri

`src/sdnids/data/synthetic.py` yalnızca altyapı testleri içindir. Betikler sentetik veriyi `--allow-synthetic` bayrağı olmadan reddeder;
sentetik veriyle üretilen hiçbir sayı rapora girmez.
