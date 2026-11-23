# Laporan Analisis — Churn & Sales Forecasting

- **Proyek:** Business Intelligence — Analisis Data & Dashboard
- **Objek:** PT Graha Material Nusantara (data sintetis, distributor material bangunan B2B)
- **Bagian:** Churn & Sales Forecasting (ML)
- **Periode data:** Januari 2019 – Oktober 2026 (94 bulan)
- **Notebook:** `02_churn_model.ipynb`, `03_sales_forecast.ipynb`
- **Keluaran:** `data/processed/churn/`, `data/processed/forecast/`, `data/processed/charts/`

---

## 1. Ringkasan eksekutif

1. **Churn meningkat dan belum berhenti.** Prevalensi churn 180 hari naik dari 1,7% (Des 2020)
   menjadi 14,1% (Des 2024), 21,0% (Des 2025), dan **30,0%** pada 31 Oktober 2026
   (30 dari 100 pelanggan). Pembeli unik per tahun turun dari 88 (2024) menjadi 73 (2026);
   pada snapshot Oktober hanya 70 pelanggan yang masih aktif.
2. **Churn terkonsentrasi, bukan merata.** Toko Bangunan menyumbang 27 dari 30 churn
   (**churn rate 49,1%**), jauh di atas Kontraktor 4,0%, Korporat 16,7%, dan Developer 0%.
   Segmen Bronze 54,2%, Gold 5,6%, Platinum 0%.
3. **Pemicu utama adalah layanan dan penurunan aktivitas.** Baris pelanggan yang berakhir churn
   punya keterlambatan kirim +139%, keterlambatan >3 hari +127%, retur +47%, pesanan 6 bulan
   −52%, dan omzet 6 bulan −55% dibanding pelanggan yang bertahan.
4. **Model churn layak dipakai untuk prioritas.** Ensambel Logistic Regression + Random Forest +
   XGBoost mencapai ROC-AUC 0,878 dan PR-AUC 0,356 pada cross-validation per pelanggan; pada
   uji waktu 2025–April 2026 PR-AUC 0,450 dan **recall@20% 0,688** — 20% observasi
   pelanggan-bulan berisiko tertinggi menangkap sekitar 69% churner.
5. **Forecast omzet membaik dibanding baseline.** SARIMA menghasilkan WAPE **8,56%**
   (baseline seasonal naive 13,70%) dan meramal **Rp 2,42 miliar untuk 6 bulan**
   (Nov 2026–Apr 2027) serta **Rp 4,66 miliar untuk 12 bulan**.
6. **Ada 14 pelanggan pita risiko Tinggi** yang mewakili **Rp 575 juta omzet 12 bulan**; 11 di
   antaranya Toko Bangunan. Ini target retensi paling konkret, bukan kampanye merata.

---

## 2. Latar belakang dan pertanyaan bisnis

Omzet 2024 mencapai puncak Rp 6,09 miliar, lalu turun menjadi Rp 5,30 miliar (2025) dan
Rp 4,00 miliar pada Januari–Oktober 2026. Dibandingkan periode yang sama (Jan–Okt) 2024,
omzet 2026 turun **19,4%**. Penurunan nominal lebih kecil daripada penurunan volume karena
harga jual naik 3–7% per tahun — artinya masalahnya bukan harga, melainkan **berkurangnya
pelanggan yang aktif membeli**.

Pertanyaan yang dijawab bagian ini:

1. Siapa pelanggan yang berisiko churn dalam 180 hari ke depan, dan apa pemicunya?
2. Segmen, wilayah, dan jenis pelanggan mana yang paling perlu diprioritaskan?
3. Berapa omzet yang bisa dipertahankan jika kampanye retensi dijalankan?
4. Berapa perkiraan penjualan 6–12 bulan ke depan, dan seberapa besar ketidakpastiannya?
5. Model peramalan mana yang paling layak dipakai, dan seberapa baik dibanding baseline?

---

## 3. Data dan definisi

### 3.1 Sumber dan ukuran

| Hal | Nilai |
|---|---|
| Sumber | Dataset sintetis repositori `graha-material-business-intelligence` (seed 42) |
| Tabel | 6 berkas bersih: pelanggan, produk, kategori, wilayah, tenaga penjualan, transaksi |
| Transaksi | 15.000 baris, 7.924 pesanan, 100 pelanggan, 80 produk |
| Periode | 1 Januari 2019 – 31 Oktober 2026 (94 bulan penuh) |
| Grain transaksi | 1 baris = 1 item dalam pesanan |

Data bersih dari notebook cleaning masih memuat satu masalah: nilai uang seperti
`Rp 231.100` terbaca **231,1** karena pemisah ribuan titik. Dampaknya **970 baris (6,47%)**
punya harga 1.000× terlalu kecil; omzet total terundervalue dari **Rp 38,41 miliar** menjadi
**Rp 35,96 miliar**. Notebook ML mengoreksi lewat kolom `harga_satuan_bersih` dan
`harga_modal_bersih` di `data/processed/transaksi_analitik.csv`, tanpa mengubah notebook
cleaning milik bagian lain.

KPI churn **tidak terpengaruh** koreksi ini karena berbasis aktivitas order, bukan nilai uang.
Namun omzet, fitur monetary, dan level forecast memakai angka terkoreksi. Dashboard harus
membaca `data/processed`, bukan `data/cleaned`, agar konsisten.

### 3.2 Definisi churn

| Istilah | Definisi |
|---|---|
| Aktivitas | Transaksi berstatus `Selesai` atau `Retur`. `Dibatalkan` tidak dihitung. |
| Churn deskriptif (KPI) | Tidak ada aktivitas 180 hari terakhir. Dinilai hanya untuk pelanggan terdaftar ≥90 hari dan punya ≥2 pesanan. |
| Label churn ML | Pada akhir bulan *m*, pelanggan masih aktif (ada aktivitas dalam 180 hari terakhir), lalu **tidak ada aktivitas pada 180 hari berikutnya**. |
| Batas label | Hanya sampai April 2026, karena jendela 180 hari setelahnya harus berada dalam periode data (31 Okt 2026). |
| Omzet | `jumlah × harga_satuan_bersih × (1 − diskon_persen/100)` |

Kedua definisi sengaja dipisahkan: angka **30%** adalah fakta churn saat ini (Halaman 1),
sedangkan **probabilitas** adalah perkiraan risiko 180 hari ke depan (Halaman 2). Keduanya
tidak boleh digabung dalam satu sumbu.

---

## 4. Metode churn

### 4.1 Mengapa panel pelanggan-bulan

Snapshot tunggal hanya menghasilkan 100 baris dengan 30 kejadian churn — terlalu sedikit untuk
melatih model. Karena itu dibentuk panel **pelanggan × bulan**:

- Bulan pengamatan: Januari 2020 – April 2026 (fitur 12 bulan butuh data mulai 2019).
- Syarat baris: terdaftar ≥90 hari, ≥2 pesanan sampai bulan itu, dan masih aktif dalam 180 hari.
- Hasil: **5.669 baris, 184 positif (3,2%), 99 pelanggan unik**.
  (Satu pelanggan hanya punya 2 pesanan pada 2023–2024 sehingga tidak pernah memenuhi syarat.)

### 4.2 Fitur (24 numerik + 5 kategorikal)

| Kelompok | Fitur |
|---|---|
| Recency | hari sejak order terakhir |
| Tenure | bulan sejak registrasi |
| Frequency | order 3/6/12 bulan, tren order 3 bulan vs 3 bulan sebelumnya, rata-rata & simpangan jarak antar-pesanan |
| Monetary | omzet 3/6/12 bulan, average order value 12 bulan, item 6 bulan |
| Keragaman | produk unik 12 bulan, kategori unik 12 bulan |
| Layanan | rasio terlambat, rasio terlambat >3 hari, rata-rata hari terlambat, rasio retur, rasio batal |
| Komersial | diskon rata-rata 6 bulan, perubahan diskon 3 bulan vs sebelumnya, rasio belum lunas |
| Profil | jenis pelanggan, segmen, region, kanal akuisisi, termin pembayaran, limit kredit |

Semua fitur hanya memakai data **sampai akhir bulan pengamatan**. Label memakai 180 hari
**setelahnya**, sehingga tidak ada kebocoran masa depan.

### 4.3 Validasi

| Skema | Rincian |
|---|---|
| Cross-validation | `StratifiedGroupKFold` 5 lipatan, **dikelompokkan per pelanggan** agar baris satu pelanggan tidak terpecah antar latih/uji |
| Uji waktu | Latih ≤ Desember 2024 (4.384 baris, 107 positif), uji Januari 2025–April 2026 (1.285 baris, 77 positif) |
| Kalibrasi | Isotonic regression dilatih pada prediksi out-of-fold; probabilitas dipakai sebagai perkiraan peluang, bukan skor mentah |
| Ambang | Dipilih dari kurva precision–recall out-of-fold (maksimum F1), lalu diterapkan ke data uji tanpa penyetelan ulang |
| Metrik | ROC-AUC, PR-AUC, recall@20%, F1, Brier; PR-AUC dan recall@20% menjadi acuan utama karena positif hanya 3,2% |

### 4.4 Model

| Model | Konfigurasi ringkas | Peran |
|---|---|---|
| Logistic Regression | `class_weight=balanced`, C=1, standardisasi + one-hot | baseline interpretable |
| Random Forest | 600 pohon, kedalaman 6, min. 8 sampel per daun, `balanced_subsample` | menangkap interaksi nonlinier |
| XGBoost | 400 pohon, kedalaman 3, learning rate 0,05, `scale_pos_weight` | boosting untuk pola kompleks |
| Ensambel | rata-rata probabilitas tiga model terkalibrasi | **rekomendasi** |

Model dipilih berdasarkan PR-AUC cross-validation; uji waktu dipakai sebagai konfirmasi, bukan
tempat memilih model.

---

## 5. Evaluasi model churn

### 5.1 Perbandingan model

| Model | CV ROC-AUC | CV PR-AUC | CV recall@20% | Uji ROC-AUC | Uji PR-AUC | Uji recall@20% | Uji F1 | Brier |
|---|---|---|---|---|---|---|---|---|
| **Ensambel** | **0,878** | **0,356** | **0,776** | **0,856** | **0,450** | **0,688** | 0,463 | 0,043 |
| Random Forest | 0,871 | 0,309 | 0,766 | 0,817 | 0,396 | 0,649 | 0,366 | 0,044 |
| XGBoost | 0,819 | 0,262 | 0,664 | 0,835 | 0,394 | 0,623 | 0,407 | 0,044 |
| Logistic Regression | 0,843 | 0,199 | 0,701 | 0,828 | 0,293 | 0,649 | 0,393 | 0,049 |

Pada periode uji, proporsi positif 6,0%. Garis dasar acak akan memberi PR-AUC sekitar 0,06;
ensambel mencapai **0,450**.

### 5.2 Matriks konfusi (ensambel, ambang 0,219)

|  | Prediksi bertahan | Prediksi churn |
|---|---|---|
| **Aktual bertahan** | 1.192 | 16 |
| **Aktual churn** | 49 | 28 |

Presisi 63,6%, recall 36,4% pada ambang tersebut. Recall terlihat rendah karena ambang
memaksimalkan F1 pada data yang sangat tidak seimbang; untuk kampanye retensi, yang lebih
relevan adalah **recall@20% = 68,8%** — dengan menyasar 20% observasi pelanggan-bulan berisiko
tertinggi, sekitar 69% churner tertangkap. Jika perusahaan ingin menangkap lebih banyak, ambang
bisa diturunkan dengan konsekuensi lebih banyak pelanggan yang dihubungi.

### 5.3 Fitur paling berpengaruh (permutation importance, Random Forest)

| Peringkat | Fitur | Penurunan PR-AUC |
|---|---|---|
| 1 | jenis pelanggan | 0,108 |
| 2 | recency (hari sejak order terakhir) | 0,055 |
| 3 | order 3 bulan | 0,040 |
| 4 | rasio keterlambatan 12 bulan | 0,038 |
| 5 | rasio belum lunas 12 bulan | 0,028 |
| 6 | omzet 3 bulan | 0,025 |
| 7 | rata-rata hari terlambat | 0,023 |
| 8 | region | 0,018 |

Jenis pelanggan menjadi fitur terkuat karena perbedaan perilaku Toko Bangunan dan Kontraktor
sangat tajam di data ini, bukan semata karena model "menghafal" nama pelanggan.

---

## 6. Temuan churn

### 6.1 Pemicu: churn vs bertahan

| Metrik (rata-rata baris panel) | Bertahan | Churn | Selisih |
|---|---|---|---|
| Hari sejak order terakhir | 22,4 | 84,1 | +275% |
| Keterlambatan kirim | 13,3% | 31,7% | +139% |
| Terlambat >3 hari | 7,8% | 17,7% | +127% |
| Retur | 3,2% | 4,7% | +47% |
| Pesanan 6 bulan | 6,9 | 3,4 | −52% |
| Omzet 6 bulan | Rp 35,2 juta | Rp 15,8 juta | −55% |
| Perubahan diskon (3 bln vs 3 bln sebelumnya) | −3,1 pp | −8,2 pp | diskon ditarik lebih tajam |

Pola yang konsisten: pelanggan tidak langsung pergi. Mereka **mengurangi frekuensi lebih dulu**,
sering setelah mengalami keterlambatan atau retur, lalu berhenti. Penarikan diskon yang tajam
juga mendahului churn.

### 6.2 Konsentrasi segmen

| Dimensi | Kategori | Churn historis | Pelanggan aktif | Risiko Tinggi |
|---|---|---|---|---|
| Jenis | Toko Bangunan | **49,1%** (27/55) | 28 | 11 |
| | Korporat | 16,7% (2/12) | 10 | 2 |
| | Kontraktor | 4,0% (1/25) | 24 | 1 |
| | Developer | 0,0% (0/8) | 8 | 0 |
| Segmen | Bronze | 54,2% (13/24) | 11 | 3 |
| | Silver | 30,8% (16/52) | 36 | 11 |
| | Gold | 5,6% (1/18) | 17 | 0 |
| | Platinum | 0,0% (0/6) | 6 | 0 |
| Region | Jawa | 35,6% (21/59) | 38 | 9 |
| | Sumatera | 35,7% (5/14) | 9 | 4 |
| | Kalimantan | 30,8% (4/13) | 9 | 1 |
| | Sulawesi | 0,0% (0/14) | 14 | 0 |

Provinsi dengan churn historis tertinggi: Sumatera Selatan 66,7%, Banten 60,0%, Jawa Barat
50,0%, Sumatera Barat 50,0%, Jawa Tengah 45,5%. Terendah: DKI Jakarta 7,1%, Sulawesi 0%.

### 6.3 Daftar risiko saat ini

- **14 pelanggan pita Tinggi**, **21 Sedang**, **35 Rendah**, dan **30 win-back**.
- 11 dari 14 pelanggan pita Tinggi adalah Toko Bangunan; sisanya 2 Korporat dan 1 Kontraktor.
- 14 pelanggan tersebut mewakili **Rp 575 juta omzet 12 bulan** — cukup besar untuk dijadikan
  target kampanye khusus.
- Win-back: 30 pelanggan yang sudah tidak aktif. Jangan dicampur dengan pelanggan aktif saat
  membuat peringkat, karena probabilitas mereka kosong (di luar wilayah data latih).

---

## 7. Metode sales forecasting

### 7.1 Seri yang diramalkan

| Seri | Definisi | Nilai total 2019–Okt 2026 |
|---|---|---|
| Omzet | jumlah omzet baris `Selesai` per bulan | Rp 38,41 miliar |
| Pesanan | jumlah `nomor_pesanan` unik per bulan | 7.924 pesanan |
| Volume | jumlah kuantitas barang per bulan | — |

Data berakhir Oktober 2026, jadi tersedia **94 titik bulanan penuh**. 2026 hanya 10 bulan;
tidak ada penyetahunkan otomatis.

### 7.2 Tujuh model yang dibandingkan

| Kode | Model | Peran |
|---|---|---|
| `naive_seasonal` | nilai bulan yang sama tahun lalu | baseline wajib |
| `ets` | Holt-Winters tren meredam + musim aditif | kandidat utama |
| `ets_48` | ETS hanya 48 bulan terakhir | uji perubahan rezim |
| `sarima` | SARIMA musiman, orde dipilih AIC | kandidat utama |
| `regresi_musiman` | tren linear + dummy bulan | pembanding yang mudah dijelaskan |
| `xgboost_lag` | lag 1–12 + rata-rata bergerak + bulan | opsi nonlinier |
| `ensamble` | rata-rata ETS dan SARIMA | opsi stabil |

Orde SARIMA dipilih sekali memakai data sampai April 2025, lalu dipakai ulang pada seluruh
backtest agar tidak bocor dari masa depan:

- Omzet: SARIMA(1,1,2)(0,1,1,12)
- Pesanan: SARIMA(1,1,2)(0,1,1,12)
- Volume: SARIMA(0,1,2)(0,1,1,12)

### 7.3 Backtest

- **12 origin**: Mei 2025 – April 2026.
- Setiap origin hanya memakai data sampai bulan itu, lalu diramal 1–6 bulan ke depan.
- Total 1.512 titik evaluasi (3 seri × 7 model × 12 origin × 6 horizon).
- Metrik: MAE, RMSE, MAPE, **WAPE** (`Σ|error| / Σ|aktual|`). WAPE dipilih karena tidak
  meledak saat nilai aktual kecil.
- Model terbaik per seri dipilih dari WAPE gabungan horizon 1–6, bukan dari satu titik.

---

## 8. Evaluasi forecasting

### 8.1 Perbandingan model omzet (backtest 6 bulan)

| Model | WAPE | MAE | RMSE |
|---|---|---|---|
| **SARIMA (rekomendasi)** | **8,56%** | Rp 35,7 juta | Rp 45,7 juta |
| Ensambel | 9,72% | Rp 40,5 juta | Rp 54,2 juta |
| ETS 48 bulan | 11,66% | Rp 48,6 juta | Rp 64,6 juta |
| ETS penuh | 11,76% | Rp 49,0 juta | Rp 67,0 juta |
| Seasonal naive | 13,70% | Rp 57,1 juta | Rp 68,0 juta |
| XGBoost lag | 16,91% | Rp 70,5 juta | Rp 80,8 juta |
| Regresi musiman | 27,53% | Rp 114,7 juta | Rp 121,9 juta |

Semua model kecuali regresi linear mengalahkan baseline. Regresi kalah karena tren berbalik
pada 2024–2026 dan model mengekstrapolasi pertumbuhan lama. XGBoost kalah karena hanya 94 titik
dan 12 lag — terlalu banyak parameter untuk data sependek itu.

### 8.2 Stabilitas menurut horizon (SARIMA)

| Horizon | Omzet WAPE | Pesanan WAPE | Volume WAPE |
|---|---|---|---|
| 1 bulan | 8,99% | 5,59% | 27,48% |
| 2 bulan | 9,02% | 6,29% | 29,21% |
| 3 bulan | 7,50% | 6,05% | 29,57% |
| 4 bulan | 8,90% | 6,84% | 30,90% |
| 5 bulan | 8,52% | 7,98% | 26,71% |
| 6 bulan | 8,44% | 9,58% | 30,73% |

Omzet relatif stabil; pesanan makin sulit pada horizon jauh; volume paling sulit diprediksi
(WAPE ±28–31%). Karena itu dashboard sebaiknya menampilkan omzet sebagai seri utama, pesanan
sebagai pendukung, dan volume hanya di halaman detail.

### 8.3 Ramalan

| Seri | Model | 6 bulan | 12 bulan |
|---|---|---|---|
| Omzet | SARIMA | **Rp 2,42 miliar** | **Rp 4,66 miliar** |
| Pesanan | SARIMA | 269 pesanan | 458 pesanan |
| Volume | Ensambel | 33.836 unit | 63.449 unit |

Contoh interval omzet: November 2026 diprediksi **Rp 431 juta** (80%: Rp 360–502 juta;
95%: Rp 323–540 juta). Maret 2027 diprediksi Rp 530 juta karena pola musiman yang berulang.
Interval melebar cepat setelah 6 bulan.

### 8.4 Skenario retensi

| Skenario | Pelanggan dipertahankan | Tambahan omzet 12 bulan | Total dengan skenario |
|---|---|---|---|
| Pertahankan 10% dari 14 risiko tinggi | 1 | Rp 72 juta | Rp 4,73 miliar |
| **Pertahankan 20%** | **3** | **Rp 216 juta** | **Rp 4,87 miliar** |
| Pertahankan 30% | 4 | Rp 289 juta | Rp 4,94 miliar |

Asumsi: pelanggan yang dipertahankan berbelanja seperti rata-rata pelanggan aktif
(Rp 72,2 juta per tahun). Ini **skenario**, bukan keluaran model time-series; tulis asumsinya
di slide.

---

## 9. Rekomendasi

| Prioritas | Tindakan | Dasar temuan | Target/metrik |
|---|---|---|---|
| 1 | **Program retensi khusus Toko Bangunan**, terutama di Jawa Barat dan Sumatera | churn 49% vs Kontraktor 4%; 11 dari 14 risiko tinggi | turunkan churn Toko Bangunan ke <35% dalam 12 bulan |
| 2 | **Perbaiki SLA kirim**, prioritaskan Sumatera dan Kalimantan | keterlambatan pelanggan churn 31,7% vs 13,3%; terlambat >3 hari naik 3,8% (2019) → 17,7% (2025) | terlambat >3 hari <10%; pantau per region |
| 3 | **Desain diskon bertahap**, jangan menarik diskon mendadak | perubahan diskon churn −8,2 pp vs bertahan −3,1 pp | uji coba diskon berjenjang 3 bulan |
| 4 | **Kampanye win-back 30 pelanggan** + retensi 14 risiko tinggi | Rp 575 juta omzet 12 bulan pada 14 risiko tinggi; skenario +Rp 216 juta | hubungi top 20% risiko; ukur konversi |
| 5 | **Pakai forecast sebagai baseline perencanaan** | SARIMA WAPE 8,56%; 6 bulan Rp 2,42 miliar | tinjau WAPE bulanan; revisi jika >12% |

Catatan: rekomendasi 1–4 memakai model churn; rekomendasi 5 memakai model forecasting.
Keduanya bertemu di skenario retensi.

---

## 10. Keterbatasan dan asumsi

1. **Data sintetis** dengan 100 pelanggan dan 33 kejadian churn sepanjang periode (30 di
   antaranya masih berstatus churn pada snapshot). Metrik sangat sensitif terhadap perubahan
   seed; jangan menyimpulkan performa untuk populasi nyata tanpa validasi ulang.
2. **Baris panel saling berkorelasi** karena satu pelanggan muncul puluhan kali. Validasi sudah
   dikelompokkan per pelanggan, tetapi tetap bukan uji independen sempurna.
3. **Probabilitas kecil itu wajar**: proporsi dasar churn per pelanggan-bulan hanya 3,2%.
   Puncak 0,40 berarti risiko relatif sangat tinggi, bukan "40% pasti churn".
4. **Label hanya sampai April 2026**; skor snapshot Oktober 2026 tidak punya kebenaran aktual
   untuk dievaluasi.
5. **Forecast 12 bulan belum tervalidasi** (backtest maksimal 6 bulan). Tampilkan 6 bulan
   pertama sebagai angka utama.
6. **2026 hanya 10 bulan**; perbandingan tahunan harus memakai Jan–Okt.
7. **Harga modal statis** sepanjang periode, sehingga tren margin (jika ditampilkan) sebagian
   artefak data.
8. **Omzet `data/cleaned` masih salah 6,47% baris**. Dashboard wajib memakai
   `data/processed/transaksi_analitik.csv` atau CSV hasil notebook ini.

---

## 11. Peta kebutuhan dashboard

### Halaman 1 — Business Overview

| Komponen | Berkas | Kolom | Saran visual | Catatan |
|---|---|---|---|---|
| Kartu KPI churn | `churn_snapshot_kpi.csv` | `churn_rate`, `pelanggan_aktif`, `pelanggan_churn`, `omzet_12_bulan` | KPI card | 30% = fakta, bukan probabilitas |
| Tren churn | `churn_rate_monthly.csv` | `periode`, `churn_rate` | line chart | beri garis rata-rata 3 bulan |
| Omzet bulanan | `forecast_values.csv` (jenis=aktual, seri=omzet) | `bulan`, `nilai` | line/area | pakai data terkoreksi |
| Omzet per tahun | `transaksi_analitik.csv` | `tahun`, `omzet` | bar chart | tandai 2026 parsial |
| Churn per jenis pelanggan | `churn_segment_risk.csv` | `dimensi=jenis_pelanggan`, `churn_rate_historis` | bar chart | sudah termasuk Toko Bangunan 49% |
| Filter | `transaksi_analitik`, `churn_scores`, `forecast_values` | region, jenis, segmen, periode | slicer | konsisten di kedua halaman |

### Halaman 2 — Analytics & Insights

| Komponen | Berkas | Kolom | Saran visual | Catatan |
|---|---|---|---|---|
| Peta risiko pelanggan | `churn_scores.csv` | `recency_hari`, `omzet_12`, `prob_ensamble`, `band_kuantil` | scatter + tabel Top 20 | hanya `status=Aktif` untuk probabilitas |
| Distribusi risiko | `churn_scores.csv` | `band_kuantil` | bar/donut | 14/21/35/30 |
| Pemicu churn | `churn_drivers.csv` + `churn_pemicu.png` | `metrik`, `selisih_relatif` | bar chart | keterlambatan +139% |
| Risiko per segmen | `churn_segment_risk.csv` | `rata2_prob`, `n_risiko_tinggi` | heatmap/bar | 11 dari 14 adalah Toko Bangunan |
| Evaluasi model | `churn_model_metrics.csv` + `churn_roc_pr.png` + `churn_confusion_matrix.png` | `cv_pr_auc`, `uji_pr_auc`, `recall_top20` | tabel + kurva | jelaskan recall@20% |
| Aktual vs ramalan | `forecast_values.csv` | `jenis`, `nilai`, `lo80`, `hi80` | line + pita | filter `model=sarima`, `seri=omzet`, `horizon_ke<=6` |
| Perbandingan model | `forecast_metrics.csv` + `forecast_perbandingan_model.png` | `wape`, `mae` | bar chart | SARIMA 8,56% vs naive 13,70% |
| Tabel ramalan | `forecast_values.csv` | `bulan`, `nilai`, `lo95`, `hi95` | tabel | 6 bulan pertama |
| Skenario retensi | `forecast_scenario.csv` | `skenario`, `tambahan_omzet_12_bulan` | kartu/skenario | tulis asumsinya |
| Insight & rekomendasi | laporan ini bagian 9 | — | teks | 3–5 poin, bukan paragraf |

**Larangan pencampuran:** jangan menaruh `churn_rate` (30%) dan `prob_ensamble` (puncak 0,40)
di sumbu yang sama tanpa penjelasan; satuannya berbeda (fakta vs perkiraan).

---

## 12. Lampiran

### 12.1 Berkas keluaran

| Folder | Berkas | Baris |
|---|---|---|
| `data/processed/churn/` | `churn_snapshot_kpi.csv` | 1 |
| | `churn_rate_monthly.csv` | 82 |
| | `churn_scores.csv` | 100 |
| | `churn_segment_risk.csv` | 35 |
| | `churn_drivers.csv` | 10 |
| | `churn_feature_importance.csv` | 29 |
| | `churn_model_metrics.csv` | 4 |
| | `churn_curve_roc.csv` | 399 |
| | `churn_curve_pr.csv` | 562 |
| `data/processed/forecast/` | `forecast_values.csv` | 534 |
| | `forecast_metrics.csv` | 147 |
| | `forecast_recommendation.csv` | 21 |
| | `forecast_backtest.csv` | 1.512 |
| | `forecast_seasonality.csv` | 12 |
| | `forecast_scenario.csv` | 3 |
| `data/processed/` | `transaksi_analitik.csv` | 15.000 |
| `data/processed/charts/` | 12 PNG 300 dpi | — |
| `docs/` | `handoff_ml.md`, `ml_manifest.json`, laporan ini | — |

### 12.2 Glosarium

| Istilah | Arti |
|---|---|
| PR-AUC | Area under precision–recall curve; lebih informatif daripada ROC-AUC saat kelas positif langka |
| Recall@20% | Proporsi churner yang tertangkap jika menyasar 20% observasi pelanggan-bulan dengan skor tertinggi |
| Brier score | Rata-rata kuadrat selisih probabilitas dan hasil aktual; makin kecil makin baik |
| WAPE | Weighted absolute percentage error; total error dibagi total aktual |
| Rolling-origin | Backtest berulang dengan titik latih yang bergerak maju |
| Kalibrasi | Penyesuaian skor model agar mendekati peluang sebenarnya |

### 12.3 Reproduksi

```
python generator.py --seed 42 --out data/raw
# jalankan notebooks/01_cleaning_preprocessing.ipynb
# jalankan notebooks/02_churn_model.ipynb
# jalankan notebooks/03_sales_forecast.ipynb
python src/buat_manifest.py
```

Semua model memakai `random_state=42`. Kebutuhan paket ada di `requirements.txt`.

### 12.4 Referensi internal

- `docs/handoff_ml.md` — panduan teknis penggunaan CSV/PNG untuk dashboard.
- `docs/ml_manifest.json` — daftar berkas, kolom, kunci, dan rekomendasi model.
- `notebooks/02_churn_model.ipynb` — metode dan evaluasi churn.
- `notebooks/03_sales_forecast.ipynb` — metode dan evaluasi forecasting.
