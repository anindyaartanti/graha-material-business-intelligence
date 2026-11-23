"""Bangun dokumen serah-terima ML: docs/ml_manifest.json.

Dijalankan setelah notebook 02 dan 03 selesai. Isinya kamus berkas, definisi,
rekomendasi model, dan angka koreksi harga yang dihitung ulang dari data.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from graha import AKAR, CHURN_DIR, FORECAST_DIR, GAMBAR_DIR, PROSES, muat_transaksi

DESKRIPSI_CHURN = {
    "churn_snapshot_kpi.csv": "KPI churn deskriptif per 31 Okt 2026 (Halaman 1).",
    "churn_rate_monthly.csv": "Prevalensi churn 180 hari per bulan, 2020-2026 (tren Halaman 1).",
    "churn_scores.csv": "Skor risiko per pelanggan: 3 model + ensambel, pita kuantil, status aktif/win-back.",
    "churn_segment_risk.csv": "Churn historis dan risiko prediksi per jenis, segmen, region, provinsi, kanal, termin.",
    "churn_drivers.csv": "Rata-rata pemicu pada baris panel churn vs bertahan.",
    "churn_feature_importance.csv": "Permutation importance model terbaik (penurunan PR-AUC).",
    "churn_model_metrics.csv": "Evaluasi model: AUC, PR-AUC, recall@20%, F1, Brier, ambang.",
    "churn_curve_roc.csv": "Titik kurva ROC per model pada periode uji.",
    "churn_curve_pr.csv": "Titik kurva precision-recall per model pada periode uji.",
}

DESKRIPSI_FORECAST = {
    "forecast_values.csv": "Long format seri x model x bulan: aktual, prediksi, interval 80/95, horizon.",
    "forecast_metrics.csv": "MAE, RMSE, MAPE, WAPE per model dan horizon (backtest 12 origin).",
    "forecast_recommendation.csv": "Peringkat model dan penanda rekomendasi per seri.",
    "forecast_backtest.csv": "Aktual vs prediksi pada 12 origin backtest.",
    "forecast_seasonality.csv": "Indeks musiman per bulan untuk omzet, pesanan, volume.",
    "forecast_scenario.csv": "Skenario retensi pelanggan berisiko tinggi (bukan prediksi model).",
}

KUNCI = {
    "churn_snapshot_kpi.csv": ["tanggal_snapshot"],
    "churn_rate_monthly.csv": ["periode"],
    "churn_scores.csv": ["id_pelanggan"],
    "churn_segment_risk.csv": ["dimensi", "kategori"],
    "churn_drivers.csv": ["metrik"],
    "churn_feature_importance.csv": ["fitur"],
    "churn_model_metrics.csv": ["model"],
    "churn_curve_roc.csv": ["model"],
    "churn_curve_pr.csv": ["model"],
    "forecast_values.csv": ["seri", "model", "bulan"],
    "forecast_metrics.csv": ["seri", "model", "horizon"],
    "forecast_recommendation.csv": ["seri", "model"],
    "forecast_backtest.csv": ["seri", "model", "origin", "horizon"],
    "forecast_seasonality.csv": ["bulan"],
    "forecast_scenario.csv": ["skenario"],
}


def ringkas_berkas(folder: Path, deskripsi: dict[str, str]) -> list[dict]:
    hasil = []
    for berkas in sorted(folder.glob("*.csv")):
        tabel = pd.read_csv(berkas, nrows=5)
        jumlah_baris = sum(1 for _ in berkas.open(encoding="utf-8")) - 1
        hasil.append(
            {
                "path": str(berkas.relative_to(AKAR)).replace("\\", "/"),
                "baris": jumlah_baris,
                "kolom": len(tabel.columns),
                "kunci": KUNCI.get(berkas.name, []),
                "deskripsi": deskripsi.get(berkas.name, ""),
            }
        )
    return hasil


def main() -> None:
    transaksi = muat_transaksi()
    dikoreksi = transaksi["harga_satuan"] != transaksi["harga_satuan_bersih"]
    selesai = transaksi["status_pesanan"] == "Selesai"
    omzet_sebelum = (
        transaksi.loc[selesai, "jumlah"]
        * transaksi.loc[selesai, "harga_satuan"]
        * (1 - transaksi.loc[selesai, "diskon_persen"] / 100)
    ).sum()
    omzet_sesudah = transaksi.loc[selesai, "omzet"].sum()

    metrik = pd.read_csv(CHURN_DIR / "churn_model_metrics.csv")
    rekom_churn = metrik.loc[metrik["is_recommended"] == True, "model"].iloc[0]
    rekom_forecast = pd.read_csv(FORECAST_DIR / "forecast_recommendation.csv")
    rekom_forecast = rekom_forecast[rekom_forecast["is_recommended"] == True]

    manifest = {
        "proyek": "PT Graha Material Nusantara — Churn & Sales Forecasting",
        "dibuat": str(date.today()),
        "pemilik_bagian": "Churn & Sales Forecasting (ML)",
        "sumber_data": {
            "dataset": "Data sintetis operasional penjualan, seed 42",
            "periode": "2019-01-01 s.d. 2026-10-31",
            "tabel_bersih": "data/cleaned/*.csv (6 berkas)",
            "tabel_analitik": "data/processed/transaksi_analitik.csv",
        },
        "definisi": {
            "aktivitas": "Transaksi status Selesai atau Retur; Dibatalkan tidak dihitung.",
            "omzet": "jumlah x harga_satuan_bersih x (1 - diskon_persen/100)",
            "churn_deskriptif": "Tidak ada aktivitas 180 hari terakhir; dinilai hanya untuk pelanggan terdaftar >=90 hari dan >=2 pesanan.",
            "churn_label_ml": "Pada akhir bulan m, pelanggan aktif dan tidak ada aktivitas dalam 180 hari berikutnya. Label tersedia sampai April 2026.",
            "pita_risiko": "Kuantil di antara pelanggan aktif: 20% teratas Tinggi, 30% berikutnya Sedang, sisanya Rendah.",
        },
        "koreksi_harga": {
            "alasan": "Notebook cleaning membaca 'Rp 231.100' sebagai 231,1 (pemisah ribuan titik).",
            "aturan": "Harga < 1% harga acuan produk dikali 1.000; harga modal diperlakukan sama.",
            "baris_dikoreksi": int(dikoreksi.sum()),
            "persen_baris": round(float(dikoreksi.mean() * 100), 2),
            "omzet_sebelum": round(float(omzet_sebelum), 2),
            "omzet_sesudah": round(float(omzet_sesudah), 2),
            "selisih": round(float(omzet_sesudah - omzet_sebelum), 2),
            "dampak": "KPI churn rate tidak terpengaruh (berbasis aktivitas), tetapi omzet, fitur monetary, dan level forecast memakai angka terkoreksi.",
        },
        "rekomendasi": {
            "model_churn": rekom_churn,
            "model_forecast": {
                baris["seri"]: baris["model"] for _, baris in rekom_forecast.iterrows()
            },
            "cara_memilih": "Semua model tersedia per baris (kolom model); gunakan is_recommended untuk default, lalu bandingkan lewat forecast_metrics.csv.",
        },
        "berkas_churn": ringkas_berkas(CHURN_DIR, DESKRIPSI_CHURN),
        "berkas_forecast": ringkas_berkas(FORECAST_DIR, DESKRIPSI_FORECAST),
        "berkas_pendukung": [
            {
                "path": "data/processed/transaksi_analitik.csv",
                "baris": int(len(transaksi)),
                "kolom": int(len(transaksi.columns)),
                "kunci": ["id_transaksi"],
                "deskripsi": "Tabel fakta dengan harga terkoreksi, omzet, lead time, keterlambatan, dan dimensi. Sumber angka bersama untuk dashboard.",
            }
        ],
        "grafik": sorted(p.name for p in GAMBAR_DIR.glob("*.png")),
        "catatan": [
            "Dataset sintetis: 100 pelanggan dan 33 kejadian churn, metrik bisa berubah bila seed diganti.",
            "Probabilitas churn sudah dikalibrasi; proporsi dasar sekitar 3%, jadi angka absolut kecil itu wajar.",
            "2026 hanya 10 bulan: jangan menyetahunkan omzet 2026 tanpa penyesuaian.",
            "Ramalan 12 bulan disediakan, tetapi bulan ke-7 sampai ke-12 belum bisa divalidasi; tampilkan 6 bulan pertama.",
            "Harga modal produk statis sepanjang periode, sehingga tren margin sebagian artefak data.",
            "Angka dashboard harus memakai data/processed (bukan data/cleaned langsung) agar konsisten dengan notebook.",
        ],
    }

    tujuan = AKAR / "docs"
    tujuan.mkdir(exist_ok=True)
    (tujuan / "ml_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("tersimpan: docs/ml_manifest.json")
    print(f"berkas churn: {len(manifest['berkas_churn'])}, forecast: {len(manifest['berkas_forecast'])}")
    print(f"grafik: {len(manifest['grafik'])}")


if __name__ == "__main__":
    main()
