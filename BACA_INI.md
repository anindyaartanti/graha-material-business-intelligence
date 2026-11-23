# BACA INI

Repo ini berisi kode dan dokumentasi bagian **Churn & Sales Forecasting (ML)** untuk proyek
Business Intelligence PT Graha Material Nusantara. Branch kerja: `feat/churn-ml`.

## Data tidak ada di repo

Folder `data/` masuk `.gitignore`, jadi CSV dan PNG **tidak ikut** saat clone/pull. Ada dua cara
mendapatkannya:

1. **Minta zip paket data** ke pemilik bagian ML (`graha-material-churn-forecast.zip`, ±3 MB).
   Setelah diekstrak, letakkan folder `data/` di root repo ini.
2. **Jalankan ulang pipeline** (deterministik, seed 42):

   ```bash
   python generator.py --seed 42 --out data/raw
   ```

   lalu jalankan berurutan:
   `notebooks/01_cleaning_preprocessing.ipynb` → `02_churn_model.ipynb` →
   `03_sales_forecast.ipynb`.

Jangan meng-commit folder `data/`; tetap gunakan zip/Drive.

## Yang ada di repo

| Path | Isi |
|---|---|
| `notebooks/02_churn_model.ipynb` | panel churn, fitur, 3 model + ensambel, evaluasi, skor risiko |
| `notebooks/03_sales_forecast.ipynb` | 7 model × 3 seri, backtest 12 origin, ramalan 12 bulan |
| `src/graha.py` | koreksi harga, omzet, lead time, utilitas bersama |
| `src/buat_manifest.py` | pembuat `docs/ml_manifest.json` |
| `docs/laporan_ml.md` | latar belakang, metode, evaluasi, temuan, rekomendasi, peta dashboard |
| `docs/handoff_ml.md` | panduan teknis tiap berkas dan kolom |
| `docs/ml_manifest.json` | daftar berkas, kolom, kunci, rekomendasi model |

## Kalau membangun dashboard

- Gunakan **`data/processed/`**, bukan `data/cleaned/`.
- Untuk omzet dan transaksi, pakai `data/processed/transaksi_analitik.csv`. Nilai di
  `data/cleaned/transaksi.csv` masih mengandung bug pemisah ribuan: 970 baris (6,47%) harganya
  1.000× terlalu kecil, omzet kurang ±Rp 2,45 miliar.
- **Churn rate 30%** (fakta per 31 Okt 2026) berbeda dari **`prob_ensamble`** (perkiraan risiko
  180 hari ke depan). Jangan digabung di satu sumbu tanpa penjelasan.
- **2026 hanya Januari–Oktober**; jangan menyetahunkan omzet 2026.
- Forecast: filter kolom `model` di `forecast_values.csv`; default `is_recommended = True`
  (omzet & pesanan: SARIMA; volume: ensambel). Tampilkan 6 bulan pertama sebagai angka utama.

## Berkas kunci untuk dashboard

| Kebutuhan | Berkas |
|---|---|
| KPI & tren churn | `churn_snapshot_kpi.csv`, `churn_rate_monthly.csv` |
| Skor risiko pelanggan | `churn_scores.csv` |
| Risiko per segmen | `churn_segment_risk.csv` |
| Pemicu churn | `churn_drivers.csv` |
| Evaluasi model | `churn_model_metrics.csv`, `forecast_metrics.csv` |
| Aktual vs ramalan | `forecast_values.csv` |
| Skenario retensi | `forecast_scenario.csv` |
| Grafik siap tempel | `charts/*.png` (12 file) |

Panduan lengkap: **`docs/handoff_ml.md`**. Penjelasan metode dan temuan: **`docs/laporan_ml.md`**.
