# Kaynaklar ve doğrulama durumu

Tarih: 2026-10-10. Yöntem: sayfalar WebFetch/WebSearch ile okundu. **Hiçbir veri dosyası indirilemedi**
(oturumun çıkış politikası `aseados.ucd.ie` ve `pypi.org` için 403 veriyor; aşılmadı).
Sayfa içerikleri bir özetleme aracı üzerinden okunduğundan, kritik rakamları (özellikle README sayıları)
asıl sayfadan bir kez de siz teyit edin.

Bu belgedeki rakamlar **kaynakların kendi iddialarıdır**; bizim ölçtüğümüz sonuçlar değildir.

## 1. InSDN (ana veri kümesi)

| Konu | Durum |
|---|---|
| Resmî adres | https://aseados.ucd.ie/datasets/SDN/ (dizin listesi okundu) |
| Dosyalar | `InSDN_DatasetCSV.zip` (18 MB, 2020-02-19), `InSDN_PCAPS_Groups/` (2020-07-13), `README.txt` (2021-02-28), `README_old.txt` (2020-03-05) |
| Sürüm numarası | Sayfada yok; yalnızca dosya tarihleri var |
| Lisans / erişim | README: açık erişim, **CC BY-NC 4.0** (ticari olmayan) |
| Atıf (README'deki biçim) | Elsayed, Le-Khac, Jurcut, "InSDN: SDN Intrusion Dataset", IEEE Access 8, 165263–165284, 2020, DOI 10.1109/ACCESS.2020.3022633 (makale başlığını yayıncı sayfasından ayrıca doğrulayın) |
| Toplama düzeneği | 4 sanal makine: Kali (saldırgan), ONOS kontrolörlü Ubuntu, Mininet+OVS'li Ubuntu, Metasploitable 2. Mininet'te h1/h2 kötü amaçlı, h3 normal, h4 web sunucusu; OVS makinesinde Docker içinde DVWA. Trafik Wireshark ile PCAP olarak yakalanmış |
| Özellikler | CICFlowMeter ile çift yönlü akış özellikleri, README "80'den fazla" diyor. **Tam sütun listesi README'de yok** (atıf yapılan makalede) |
| Etiketleme | README: etiketleme için kaynak ve hedef IP kullanılmış → **IP özelliği etiketleme kuralını ezberletir** (bkz. `ozellik_ve_sizinti_kararlari.md`) |

### README'de yazan sınıf sayıları (aynen)

| Dosya | README'deki toplam | Sınıflar |
|---|---:|---|
| `Normal_data.csv` | 68.424 | Normal |
| `OVS.csv` | 136.743 | DoS 52.471, DDoS 48.413, Probe 36.372, brute-force-attack 1.110, Web_attack 192, Botnet 164 |
| `metasploitable-2.csv` | 138.772 | DoS 1.145, DDoS 73.529, Probe 61.757, brute-force-attack 295, Exploitation (R2L) 17 |
| Genel (README) | normal 68.424, saldırı 275.515, toplam 343.939 | |

**README'nin kendi içinde tutarsızlıkları** (aritmetik bu oturumda kontrol edildi):

- OVS sınıflarının toplamı **138.722**, metasploitable-2 sınıflarının toplamı **136.743**. README bu iki dosyanın toplamlarını yer değiştirmiş yazmış; ayrıca 138.772 değeri 138.722 ile 50 fark ediyor.
- Sınıf toplamlarından saldırı = 275.465, genel = 343.889; README'deki 275.515 / 343.939 ile 50 kayıt farkı var.
- Sınıf adları README'de `brute-force-attack`, `Web_attack`, `Exploitation (R2L)` biçiminde; CSV'deki gerçek yazım ve `Label` sütunu adı **doğrulanamadı**.

Kendi sayımımız gerçek CSV'lerden çıkarılacak ve bu tabloyla karşılaştırılacaktır.

### Doğrulanamayanlar

Zip içeriği, gerçek sütun adları/sırası, sınıf adlarının CSV'deki yazımı, `README_old.txt` ile farkı,
`InSDN_PCAPS_Groups/` içeriği.

## 2. SDN-MG25 (ikinci aşama; şu an erişilemiyor)

**IEEE DataPort sayfası** (https://ieee-dataport.org/documents/sdn-mg25, okundu):

- Lisans CC BY 4.0; sayfada "bu veri kümesi IEEE DataPort aboneliği gerektirir" ifadesi var, dosyalar için giriş gerekiyor. Oluşturulma: 19 Ağustos 2025. Sürüm numarası belirtilmemiş.
- Dosyalar: `benign traffic.zip` (1,05 GB), `attack traffic.zip` (455,86 MB), `benign/attack system calls.zip`, `benign/attack SDN states.zip`, `power measurement.zip`, `benign power measurement.zip`, `aes encryption.zip` (20,19 MB). Biçimler: jsonl, pcap, pml, tbl.
- Ağ trafiği PCAP'ten CICFlowMeter ile akış özelliklerine çevrilmiş.
- Sayfada gözlenen tutarsızlıklar: alıntı bölümünde "UNSW-MG24" adı geçiyor; "SDN-OVS durumları" ve "güç ölçümleri" açıklamaları birbirinin yerine yazılmış görünüyor.

**Hakemli makale** (DOI 10.1109/OJCS.2025.3639408): DOI, IEEE Xplore `document/11271830` adresine yönleniyor;
sayfa içeriği boş döndü, **yayımlanmış sürümün ayrıntıları doğrulanamadı**.

**Ön baskı** (Preprints.org 202509.0735 v1, hakemsiz, yayımlanan sürümden farklı olabilir):
başlık "SDN-MG25: A Comprehensive Dataset for Cybersecurity Analysis in Software Defined Networking-Enabled Microgrid Systems"
(Zhang, Turnbull, Kasra Kermanshahi, Pota, Hu).

- Floodlight kontrolörü, OpenFlow 1.3; Edge/Distribution/Core katmanlı kurumsal tipte ağ + DMZ + veri merkezi; mikro şebeke SCADA/EMS iletişimi; VMware üzerinde sanal makineler, Kali saldırgan.
- Süre: bölümlere göre 5 veya 10 gün (tutarsız). Metin "on bir" saldırı senaryosu diyor, on tane listeliyor.
- Etiketler **senaryo zaman çizelgesinden otomatik türetilmiş** (zaman aralığı temelli).
- Akış sayıları tutarsız: ~20.920 normal akış (bölüm 4) ile >209.200 normal ve >181.400 kötü amaçlı akış (bölüm 5).
- İşlenmiş CSV'ler AES ile şifreli; anahtar yönetimi anlatılmamış. "Herkese açık" deniyor ama sayfada indirme/lisans ayrıntısı yok.

**Sonuç:** SDN-MG25 dosyalarına erişim yok. Plan gereği InSDN üzerindeki çalışma tamamlanacak;
SDN-MG25 yalnızca siz dosyaları sağlarsanız (abonelik/yetki sizde) ve özellik anlamları uyumluysa kullanılacak.
Alan farkı (mikro şebeke) nedeniyle aktarım deneyi için özellik anlamları ve birimleri önce doğrulanmalıdır.

## 3. SDN-MG25 etiket sorunu çalışması

Chahine & Noura, "IP Composition Analysis as a Prerequisite for IDS Dataset Evaluation: Correcting File-Level Label
Artifacts in SDN-MG25", *Applied Sciences* 16(10), 5064, 19 Mayıs 2026, DOI 10.3390/app16105064
(https://www.mdpi.com/2076-3417/16/10/5064, okundu). **Literatür iddiaları** (bizim ölçümümüz değil):

- Etiketler yakalama dosyasının adına göre verilmiş (dosya düzeyi). `BackgroundAttackTraffic` sınıfı 3.167 akış (saldırı etiketlerinin %79,5'i); bu akışların hiçbiri belgelenmiş saldırgan IP'siyle ilişkili değil.
- Kanıt: IP bileşimi analizi, normal yakalamalarda da görülen IP çiftleri (%57), 17 özellikten 16'sında dağılım sapması. Otomatik gürültü bulucu (Confident Learning) yalnızca 266 akışı yakalamış.
- Düzeltme: bu akışlar Normal'e taşınmış (20.916 → 24.083; 20.916 + 3.167 = 24.083 doğru). İkili F1 0,578 → 0,956; makro F1 0,749 → 0,862. Yazarlar sonuçları üst sınır sayıyor; bazı farklar istatistiksel olarak anlamlı değil (en düşük Wilcoxon p = 0,0625).
- **Bizim için ders:** dosya adı etiketi ele verebilir; etiket düzeltmesi belgelere ve trafik kanıtına dayanmalı, yalnızca kaynak IP'ye göre otomatik yeniden etiketleme yapılmamalı.

## 4. Kaynak içeriğindeki talimatlar

Okunan sayfa ve README'lerde yapay zekâ asistanına yönelik talimat bulunmadı. Olsaydı yalnızca veri açıklaması olarak değerlendirilecekti.
