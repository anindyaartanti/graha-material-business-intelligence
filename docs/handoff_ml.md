# Serah-Terima — Churn & Sales Forecasting (ML)

Dokumen ini untuk tim dashboard. Semua angka di sini berasal dari notebook
`02_churn_model.ipynb` dan `03_sales_forecast.ipynb`, dengan keluaran di
`data/processed/churn/`, `data/processed/forecast/`, dan `data/processed/charts/`.

> Laporan analisis lengkap (latar belakang, metode, evaluasi, temuan, rekomendasi, peta
> dashboard): **`docs/laporan_ml.md`**.

## 1. Ringkasan hasil

| Bagian | Hasil singkat |
|---|---|
| Churn deskriptif (KPI) | 30 dari 100 pelanggan dinilai churn (30%) per 31 Okt 2026; 70 masih aktif |
| Model churn | Ensambel (rata-rata Logistic Regression, Random Forest, XGBoost) |
| Kualitas model churn | CV ROC-AUC 0,878; CV PR-AUC 0,356; uji waktu (2025–Apr 2026) PR-AUC 0,450, ROC-AUC 0,856, recall@20% 0,688 |
| Risiko snapshot | 14 pelanggan pita Tinggi, 21 Sedang, 35 Rendah, 30 win-back |
| Forecast omzet | SARIMA, WAPE backtest 8,56% (baseline seasonal naive 13,70%) |
| Ramalan omzet | Rp 2,42 miliar untuk 6 bulan dan Rp 4,66 miliar untuk 12 bulan (Nov 2026–Okt 2027) |
| Forecast pesanan | SARIMA, WAPE 7,02% |
| Forecast volume | Ensambel, WAPE 27,88% (paling sulit diprediksi) |

## 2. Mulai dari mana (untuk dashboard)

1. **Halaman 1:** `churn_snapshot_kpi.csv` + `churn_rate_monthly.csv` untuk KPI dan tren.
2. **Halaman 2 churn:** `churn_scores.csv` (tabel/scatter), `churn_segment_risk.csv`
   (perbandingan segmen), `churn_drivers.csv` + `churn_pemicu.png` (faktor penyebab),
   `churn_model_metrics.csv` + `churn_roc_pr.png` (evaluasi).
3. **Halaman 2 forecast:** `forecast_values.csv` (grafik aktual vs ramalan),
   `forecast_metrics.csv` + `forecast_perbandingan_model.png` (perbandingan model),
   `forecast_scenario.csv` (skenario retensi).

**Default yang disarankan:** model dengan `is_recommended = True`. Opsi lain tetap tersedia
supaya tim dashboard bisa berganti model lewat filter tanpa menjalankan Python lagi.

## 3. Definisi yang harus dijaga

| Istilah | Definisi |
|---|---|
| Aktivitas | Transaksi berstatus `Selesai` atau `Retur`. `Dibatalkan` tidak dihitung. |
| Omzet | `jumlah × harga_satuan_bersih × (1 − diskon_persen/100)` |
| Churn deskriptif | Tidak ada aktivitas 180 hari terakhir. Hanya dinilai untuk pelanggan terdaftar ≥90 hari dan punya ≥2 pesanan. |
| Label churn ML | Pada akhir bulan *m*, pelanggan masih aktif, lalu tidak ada aktivitas dalam 180 hari berikutnya. Label tersedia sampai April 2026. |
| Peluang churn | Probabilitas terkalibrasi model untuk 180 hari ke depan. **Bukan** churn rate KPI. |
| Pita risiko | Kuantil antar pelanggan aktif: 20% teratas Tinggi, 30% berikutnya Sedang. |
| Komparasi adil 2026 | 2026 hanya Jan–Okt; bandingkan dengan Jan–Okt tahun lain, bukan setahun penuh. |

## 4. Koreksi harga (penting)

Notebook cleaning membaca `Rp 231.100` sebagai `231,1` (pemisah ribuan titik). Akibatnya
**970 baris (6,47%)** punya harga 1.000× terlalu kecil dan omzet kurang **Rp 2,45 miliar**
(Rp 35,96 → Rp 38,41 miliar). Notebook ML mengoreksi dengan kolom `harga_satuan_bersih` dan
`harga_modal_bersih` di `data/processed/transaksi_analitik.csv`.

- KPI churn rate **tidak terpengaruh** karena berbasis aktivitas order.
- Omzet, fitur monetary, dan level forecast **memakai angka terkoreksi**.
- Dashboard sebaiknya membaca `data/processed`, bukan `data/cleaned` langsung, agar angkanya
  konsisten dengan notebook dan tidak berbeda ±6,4% dari grafik forecast.

## 5. Berkas churn

| Berkas | Baris | Kunci | Kolom penting |
|---|---|---|---|
| `churn_snapshot_kpi.csv` | 1 | tanggal_snapshot | total_pelanggan, pelanggan_aktif, pelanggan_churn, churn_rate, omzet_12_bulan |
| `churn_rate_monthly.csv` | 82 | periode | n_dinilai, n_churn, churn_rate |
| `churn_scores.csv` | 100 | id_pelanggan | status, recency_hari, order_6, omzet_12, late_rate_12, prob_logistik/random_forest/xgboost/ensamble, band_kuantil, peringkat_risiko |
| `churn_segment_risk.csv` | 35 | dimensi + kategori | n_aktif, churn_rate_historis, n_risiko_tinggi, rata2_prob, omzet_12_bulan |
| `churn_drivers.csv` | 10 | metrik | bertahan, churn, selisih_relatif |
| `churn_feature_importance.csv` | 29 | fitur | importance, std, model |
| `churn_model_metrics.csv` | 4 | model | cv_*, uji_*, ambang, tp/fp/fn, is_recommended |
| `churn_curve_roc.csv` / `churn_curve_pr.csv` | 399 / 562 | model | titik kurva untuk grafik evaluasi |

### Cara membaca `churn_scores.csv`

- `status = Aktif` (70): punya peluang churn dan pita risiko.
- `status = Win-back` (30): sudah tidak aktif; kolom probabilitas kosong. Jangan diurutkan
  bersama pelanggan aktif seolah-olah probabilitasnya 0.
- `prob_ensamble` adalah rata-rata tiga model terkalibrasi. Nilainya kecil (puncak ±0,4) karena
  proporsi dasar churn hanya 3% per pelanggan-bulan; gunakan `peringkat_risiko` dan
  `band_kuantil` untuk prioritas kampanye.
- Pelanggan pita Tinggi dapat dilihat langsung dari `band_kuantil = "Tinggi"`.

### Faktor pemicu (dari `churn_drivers.csv`)

| Metrik | Bertahan | Churn | Selisih |
|---|---|---|---|
| Hari sejak order terakhir | 22 | 84 | +275% |
| Keterlambatan kirim | 13,3% | 31,7% | +139% |
| Terlambat >3 hari | 7,8% | 17,7% | +127% |
| Retur | 3,2% | 4,7% | +47% |
| Pesanan 6 bulan | 6,9 | 3,4 | −52% |
| Omzet 6 bulan | Rp 35,2 jt | Rp 15,8 jt | −55% |
| Perubahan diskon (3 bln vs 3 bln sebelumnya) | −3,1 pp | −8,2 pp | diskon ditarik lebih tajam |

## 6. Berkas forecast

| Berkas | Baris | Kunci | Kolom penting |
|---|---|---|---|
| `forecast_values.csv` | 534 | seri + model + bulan | jenis (aktual/prediksi), nilai, lo80, hi80, lo95, hi95, horizon_ke, is_recommended |
| `forecast_metrics.csv` | 147 | seri + model + horizon | mae, rmse, mape, wape, n_obs |
| `forecast_recommendation.csv` | 21 | seri + model | wape, mae, rmse, mape, peringkat, is_recommended |
| `forecast_backtest.csv` | 1.512 | seri + model + origin + horizon | aktual, prediksi, error |
| `forecast_seasonality.csv` | 12 | bulan | indeks omzet/pesanan/volume |
| `forecast_scenario.csv` | 3 | skenario | pelanggan_dipertahankan, tambahan_omzet_12_bulan, total_dengan_skenario |

### Perbandingan model omzet (WAPE backtest 6 bulan, makin kecil makin baik)

| Model | WAPE | MAE |
|---|---|---|
| `sarima` (rekomendasi) | 8,56% | Rp 35,7 jt |
| `ensamble` | 9,72% | Rp 40,5 jt |
| `ets_48` | 11,66% | Rp 48,6 jt |
| `ets` | 11,76% | Rp 49,0 jt |
| `naive_seasonal` | 13,70% | Rp 57,1 jt |
| `xgboost_lag` | 16,91% | Rp 70,5 jt |
| `regresi_musiman` | 27,53% | Rp 114,7 jt |

Semua model mengalahkan baseline kecuali regresi linear, yang melebih-lebihkan karena tren
berbalik pada 2024–2026. XGBoost juga kalah karena data deret hanya 94 titik.

### Cara memakai `forecast_values.csv`

- Filter `seri` (omzet/pesanan/volume), lalu `model`, lalu `jenis`.
- Baris `jenis = aktual` hanya ada sekali per bulan (model = `aktual`), jadi tidak terduplikasi
  saat membandingkan model.
- Tampilkan **6 bulan pertama** (`horizon_ke <= 6`) untuk presentasi; bulan 7–12 belum bisa
  divalidasi.
- Interval 80% cocok untuk pita utama; 95% untuk garis luar.

### Skenario retensi

Jika 20% dari 14 pelanggan berisiko tinggi berhasil dipertahankan (3 pelanggan), tambahan omzet
12 bulan sekitar **Rp 216 juta**, atau menaikkan total ramalan dari Rp 4,66 miliar menjadi
Rp 4,87 miliar. Asumsinya pelanggan yang dipertahankan berbelanja seperti rata-rata pelanggan
aktif (Rp 72,2 juta per tahun); ini skenario, bukan hasil model.

## 7. Grafik siap tempel (`data/processed/charts/`)

| Berkas | Isi |
|---|---|
| `churn_rate_bulanan.png` | tren prevalensi churn 2020–2026 |
| `churn_distribusi_risiko.png` | sebaran peluang churn pelanggan aktif |
| `churn_roc_pr.png` | kurva ROC dan precision–recall 4 model |
| `churn_confusion_matrix.png` | matriks konfusi model rekomendasi |
| `churn_feature_importance.png` | 12 fitur paling berpengaruh |
| `churn_pemicu.png` | perbedaan pemicu churn vs bertahan |
| `churn_segmen_risiko.png` | churn per jenis pelanggan & risiko per region |
| `forecast_aktual_prediksi.png` | aktual + ramalan 12 bulan + interval |
| `forecast_perbandingan_model.png` | WAPE semua model omzet |
| `forecast_backtest.png` | aktual vs prediksi satu langkah pada 12 origin |
| `forecast_dekomposisi.png` | dekomposisi STL omzet |
| `forecast_musiman.png` | indeks musiman omzet |

## 8. Menyegarkan hasil

```
python generator.py --seed 42 --out data/raw
# jalankan notebooks/01_cleaning_preprocessing.ipynb
# jalankan notebooks/02_churn_model.ipynb
# jalankan notebooks/03_sales_forecast.ipynb
python src/buat_manifest.py
```

Butuh: pandas, numpy, scikit-learn, statsmodels, xgboost, matplotlib (lihat
`requirements.txt`). Seed 42, `random_state=42` di semua model.

## 9. Keterbatasan (tulis di slide)

1. Data **sintetis**, 100 pelanggan, 33 kejadian churn. Metrik sensitif terhadap seed.
2. Label churn berulang tiap bulan sehingga baris satu pelanggan berkorelasi; karena itu
   validasi dikelompokkan per pelanggan, bukan acak per baris.
3. Peluang churn sudah dikalibrasi tetapi proporsi dasarnya kecil (±3%); jangan menafsirkan
   puncak 0,4 sebagai "40% pelanggan akan churn".
4. Forecast 12 bulan masih dalam ketidakpastian besar; hanya 6 bulan pertama yang layak
   ditonjolkan.
5. `harga_modal` statis sepanjang periode, jadi tren margin (kalau ditampilkan) sebagian
   artefak data.
6. Harga pada `data/cleaned` masih mengandung bug 6,47% baris; gunakan data `processed`.

## 10. Untuk slide bagian ML

| Slide | Isi yang dipakai |
|---|---|
| Metode churn | definisi label, panel 5.669 baris/184 positif, fitur utama, validasi per pelanggan |
| Evaluasi churn | tabel 4 model + ROC/PR + confusion matrix |
| Temuan churn | `churn_pemicu.png`, `churn_segmen_risiko.png`, 14 pelanggan risiko tinggi |
| Metode forecast | 7 model, backtest 12 origin, WAPE |
| Hasil forecast | `forecast_aktual_prediksi.png`, Rp 2,42 M (6 bulan), interval |
| Perbandingan | `forecast_perbandingan_model.png`, SARIMA vs baseline |
| Rekomendasi | prioritas Toko Bangunan & Sumatera, perbaikan SLA, retensi 14 pelanggan berisiko |
