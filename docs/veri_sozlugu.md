# InSDN veri sözlüğü (gerçek CSV'lerden)

Bu belgedeki sayılar `outputs/audit/audit.json` ve `scripts/01_audit_data.py` çıktısından gelir; yani **bizim ölçümümüzdür**.
Kaynak: InSDN (UCD, CC BY-NC 4.0). Literatürdeki rakamlar `kaynaklar.md` içindedir ve burada karıştırılmamıştır.

## Dosyalar

| Dosya | Satır | İçerik (etiket → satır) |
|---|---:|---|
| `Normal_data.csv` | 68.424 | Normal 68.424 |
| `OVS.csv` | 138.722 | DoS 52.471, DDoS 48.413, Probe 36.372, BFA 1.110, Web-Attack 192, BOTNET 164 |
| `metasploitable-2.csv` | 136.743 | DDoS 73.529, Probe 61.757, DoS 1.145, BFA 295, U2R 17 |
| **Toplam** | **343.889** | |

Her dosya tek bir ikili sınıfa sahiptir: Normal_data.csv = %100 Normal, diğer ikisi = %100 saldırı. Bu yüzden **`source_file` ikili etiketi tamamen ele verir**
ve model girdisine verilemez (`schema.py` bunu zorlar).

## Sütunlar (84)

| Grup | Sütun sayısı | Sütunlar | Kullanım |
|---|---:|---|---|
| Etiket | 1 | `Label` | Hedef; model girdisi değil |
| Kimlik / zaman | 6 | `Flow ID`, `Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Timestamp` | Denetim ve bölme için saklanır; modele verilmez (gerekçe: `ozellik_ve_sizinti_kararlari.md`) |
| Akış özellikleri | 77 | `Protocol` ve CICFlowMeter istatistikleri | Model girdisi (şema: `outputs/audit/schema.json`) |

77 özelliğin ailesi (CICFlowMeter adlandırması; `schema.py` bunları `snake_case`'e çevirir):

- **Süre ve hacim:** `flow_duration`, `tot_fwd_pkts`, `tot_bwd_pkts`, `totlen_fwd_pkts`, `totlen_bwd_pkts`, `subflow_*`.
- **Paket boyu istatistikleri:** `fwd_pkt_len_{max,min,mean,std}`, `bwd_pkt_len_*`, `pkt_len_{min,max,mean,std,var}`, `pkt_size_avg`, `*_seg_size_*`.
- **Hız:** `flow_byts_s`, `flow_pkts_s`, `fwd_pkts_s`, `bwd_pkts_s`.
- **Paketler arası süre (IAT):** `flow_iat_*`, `fwd_iat_*`, `bwd_iat_*` (toplam/ortalama/std/max/min).
- **TCP bayrakları:** `fin/syn/rst/psh/ack/urg_flag_cnt`, `fwd_psh_flags`, `bwd_psh_flags`, `fwd_urg_flags`, `bwd_urg_flags`, `cwe_flag_count`, `ece_flag_cnt`.
- **Pencere/başlık:** `fwd_header_len`, `bwd_header_len`, `init_fwd_win_byts`, `init_bwd_win_byts`, `fwd_act_data_pkts`, `fwd_seg_size_min`.
- **Aktif/boşta:** `active_{mean,std,max,min}`, `idle_{mean,std,max,min}`.
- **Toplu (bulk) özellikleri:** `fwd_byts_b_avg`, `fwd_pkts_b_avg`, `fwd_blk_rate_avg` ve `bwd_*` karşılıkları.
- `down_up_ratio`, `protocol`.

**Birimler.** CICFlowMeter geleneğinde süreler mikrosaniye, hızlar paket/bayt başına saniyedir. InSDN README'si birimleri yazmıyor; bu bilgi
**CICFlowMeter geleneğinden varsayımdır, bu veri kümesi için doğrulanmadı.** Canlı hatta özellik çıkarırken (Aşama 4) aynı aracın aynı
birimleri ürettiği ayrıca sınanmalıdır; `schema.py` içindeki `guess_unit` yalnızca ipucudur ve şemanın özet değerine (digest) girmez.

## Veri kalitesi bulguları (ölçüldü)

| Konu | Bulgu |
|---|---|
| Eksik / sonsuz değer | Yok (NaN ve ±inf sayısı 0) |
| Sabit özellik | 12: `fwd_psh_flags`, `fwd_urg_flags`, `cwe_flag_count`, `ece_flag_cnt`, `fwd_byts_b_avg`, `fwd_pkts_b_avg`, `fwd_blk_rate_avg`, `bwd_byts_b_avg`, `bwd_pkts_b_avg`, `bwd_blk_rate_avg`, `init_fwd_win_byts`, `fwd_seg_size_min` (şemada kalır; bilgi taşımazlar) |
| Birebir tekrar vektör | 159.143 fazladan kopya; DDoS'ta %98,8 (121.942 satır → 1.520 benzersiz), Probe %31,6, Normal %6,3 |
| Aynı vektör, farklı etiket | 79 vektör, 207 satır → dışlandı, listesi `outputs/dedupe/excluded_conflicting_label.csv` |
| IP saflığı | `Src IP`: satırların %89'u yalnızca tek etiketli IP'lerden; `Dst IP`: %56. Etiketleme IP'ye dayandığı için IP model girdisi olamaz |
| Zaman damgası | Her dosyada 2–3 farklı biçim (tablo programında yeniden kaydedilmiş). Gün-önce okunur; çoğu dakika hassasiyetinde |
| Yakalama süreleri | Normal: 31 Ara 2019 – 7 Şub 2020; OVS: 25 Ara 2019 – 4 Şub 2020; metasploitable-2: yalnızca 9–10 Ocak 2020 |

## README ile CSV arasındaki fark

- Sınıf toplamları birebir tutuyor; dosya toplamları README'de yanlış/yer değiştirmiş (bkz. `kaynaklar.md`).
- README "Exploitation (R2L)" der, CSV'de bu 17 kayıt `U2R` etiketlidir. Biz CSV'deki adı koruruz; saldırı türünün gerçekte ne olduğunu belge doğrulamıyor.
- `Web_attack` (README) = `Web-Attack` (CSV), `brute-force-attack` (README) = `BFA` (CSV); etiket takma adları `labels.py` içinde tek yerde tutulur, tanınmayan etiket sessizce eşlenmez.

## Bu bulguların sonuçları

1. **Tekrar indirgeme zorunlu.** Yapılmasaydı test kümesinin yarısı eğitimdeki kayıtların kopyası olurdu ve başarı olduğundan yüksek çıkardı.
2. **DDoS çeşitliliği düşük:** benzersiz vektör sayısı çok az; bu sınıfta yüksek skor "kolay sınıf" demek olabilir, rapor bunu söylemeli.
3. **Nadir sınıflar (U2R 17, BOTNET 164, Web-Attack 192):** `within_run` testinde U2R yalnızca 3 kayıt; güvenilir ölçüm iddia edilmeyecek.
4. **Yakalama ortamı karışıklığı:** Normal ve saldırı farklı yakalamalardan geliyor. Model "yakalama imzasını" öğrenmiş olabilir; bu sınırlılık raporda yer alacak.
