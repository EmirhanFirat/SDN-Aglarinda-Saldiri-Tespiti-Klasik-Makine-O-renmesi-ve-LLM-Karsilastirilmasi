# Veri inceleme raporu

- Kayıt sayısı: 343889
- Özellik sayısı: 77

## Sınıf dağılımı

| Sınıf | Kayıt |
|---|---:|
| DDoS | 121942 |
| Probe | 98129 |
| Normal | 68424 |
| DoS | 53616 |
| BFA | 1405 |
| Web-Attack | 192 |
| BOTNET | 164 |
| U2R | 17 |

## Uyarılar

- Sabit özellik(ler): ['fwd_psh_flags', 'fwd_urg_flags', 'cwe_flag_count', 'ece_flag_cnt', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'bwd_blk_rate_avg', 'init_fwd_win_byts', 'fwd_seg_size_min']
- Tekrar eden özellik vektörleri var; aynı bölmeye konacak (grup bütünlüğü).
- Aynı özellik vektörü farklı etiketlerle görülüyor (şüpheli etiket / ayırt edilemez kayıt).
- Dosya adı ikili etiketi büyük ölçüde ele veriyor; source_file model girdisine verilmemeli.
- Aynı dosyada birden fazla zaman damgası biçimi var (['Normal_data.csv', 'OVS.csv', 'metasploitable-2.csv']); dosyalar bir elektronik tabloda yeniden kaydedilmiş olabilir. Saniye hassasiyeti ve gün/ay sırası güvenilir değil.

## Tekrar kayıtlar

- extra_copies: 159143
- rows_in_duplicate_sets: 173016
- conflicting_label_vectors: 79
- rows_in_conflicting_vectors: 207

| Sınıf | Kayıt | Benzersiz vektör | Tekrar payı |
|---|---:|---:|---:|
| BFA | 1405 | 1377 | 2.0% |
| BOTNET | 164 | 164 | 0.0% |
| DDoS | 121942 | 1520 | 98.8% |
| DoS | 53616 | 50310 | 6.2% |
| Normal | 68424 | 64114 | 6.3% |
| Probe | 98129 | 67140 | 31.6% |
| U2R | 17 | 17 | 0.0% |
| Web-Attack | 192 | 183 | 4.7% |
