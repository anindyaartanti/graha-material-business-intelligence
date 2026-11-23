"""Fondasi bersama untuk notebook churn dan sales forecasting.

Modul ini menyatukan tiga hal yang harus konsisten di kedua notebook:
1. lokasi berkas dan folder keluaran,
2. koreksi parsing harga (bug pemisah ribuan pada notebook cleaning),
3. kolom turunan analitik: omzet, lead time, dan keterlambatan kirim.

Dipakai oleh notebooks/02_churn_model.ipynb dan notebooks/03_sales_forecast.ipynb.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BERKAS_DIMENSI = ["pelanggan", "produk", "kategori", "wilayah", "tenaga_penjualan"]
STATUS_AKTIVITAS = ("Selesai", "Retur")


def akar_proyek(mulai: Path | None = None) -> Path:
    """Naik dari folder kerja sampai menemukan folder data/cleaned."""
    awal = Path(mulai or Path.cwd()).resolve()
    for kandidat in [awal, *awal.parents]:
        if (kandidat / "data" / "cleaned").exists():
            return kandidat
    raise FileNotFoundError("Folder data/cleaned tidak ditemukan. Jalankan notebook cleaning dulu.")


AKAR = akar_proyek()
RAW = AKAR / "data" / "raw"
BERSIH = AKAR / "data" / "cleaned"
PROSES = AKAR / "data" / "processed"
CHURN_DIR = PROSES / "churn"
FORECAST_DIR = PROSES / "forecast"
GAMBAR_DIR = PROSES / "charts"

AKHIR_DATA = pd.Timestamp("2026-10-31")
JENDELA_CHURN = 180  # hari


def siapkan_folder() -> None:
    for folder in (PROSES, CHURN_DIR, FORECAST_DIR, GAMBAR_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def simpan(df: pd.DataFrame, berkas: Path, **kwargs) -> None:
    berkas.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(berkas, index=False, **kwargs)
    print(f"  tersimpan: {berkas.relative_to(AKAR)} ({len(df):,} baris)")


def muat_dimensi() -> dict[str, pd.DataFrame]:
    """Muat lima tabel dimensi bersih dengan tipe data yang benar."""
    pelanggan = pd.read_csv(BERSIH / "pelanggan.csv", parse_dates=["tanggal_registrasi"])
    produk = pd.read_csv(BERSIH / "produk.csv")
    kategori = pd.read_csv(BERSIH / "kategori.csv")
    wilayah = pd.read_csv(BERSIH / "wilayah.csv")
    sales = pd.read_csv(
        BERSIH / "tenaga_penjualan.csv", parse_dates=["tanggal_masuk", "tanggal_keluar"]
    )
    return {
        "pelanggan": pelanggan,
        "produk": produk,
        "kategori": kategori,
        "wilayah": wilayah,
        "tenaga_penjualan": sales,
    }


def koreksi_harga(transaksi: pd.DataFrame, produk: pd.DataFrame) -> pd.DataFrame:
    """Perbaiki harga yang terpotong 1.000x karena pemisah ribuan titik.

    Notebook cleaning membaca "Rp 231.100" sebagai 231,1. Aturan koreksi:
    harga yang kurang dari 1% harga acuan produk (atau harga modal) dikali 1.000.
    """
    acuan = produk.set_index("id_produk")[["harga_jual_acuan", "harga_modal"]]
    hasil = transaksi.merge(acuan, left_on="id_produk", right_index=True, how="left")

    rasio_jual = hasil["harga_satuan"] / hasil["harga_jual_acuan"]
    rasio_modal = hasil["harga_modal"] / hasil["harga_jual_acuan"]

    hasil["harga_satuan_bersih"] = np.where(
        rasio_jual < 0.01, hasil["harga_satuan"] * 1000, hasil["harga_satuan"]
    ).round()
    hasil["harga_modal_bersih"] = np.where(
        rasio_modal < 0.01, hasil["harga_modal"] * 1000, hasil["harga_modal"]
    ).round()
    return hasil


def muat_transaksi(koreksi: bool = True) -> pd.DataFrame:
    """Muat transaksi bersih dan tambahkan kolom turunan analitik."""
    transaksi = pd.read_csv(
        BERSIH / "transaksi.csv",
        parse_dates=["tanggal_transaksi", "tanggal_kirim"],
    )
    dimensi = muat_dimensi()
    produk = dimensi["produk"]
    pelanggan = dimensi["pelanggan"]
    wilayah = dimensi["wilayah"]

    if koreksi:
        transaksi = koreksi_harga(transaksi, produk)
    else:
        transaksi["harga_satuan_bersih"] = transaksi["harga_satuan"]
        transaksi["harga_modal_bersih"] = transaksi["harga_satuan"]

    transaksi = transaksi.merge(
        produk[["id_produk", "id_kategori"]], on="id_produk", how="left"
    )
    transaksi = transaksi.merge(
        pelanggan[["id_pelanggan", "jenis_pelanggan", "segmen", "id_wilayah", "tanggal_registrasi"]],
        on="id_pelanggan",
        how="left",
    )
    transaksi = transaksi.merge(
        wilayah[["id_wilayah", "provinsi", "kota", "region", "sla_kirim_hari"]],
        on="id_wilayah",
        how="left",
    )

    transaksi["omzet"] = (
        transaksi["jumlah"] * transaksi["harga_satuan_bersih"] * (1 - transaksi["diskon_persen"] / 100)
    )
    transaksi["lead_time_hari"] = (transaksi["tanggal_kirim"] - transaksi["tanggal_transaksi"]).dt.days
    transaksi["terlambat_hari"] = (transaksi["lead_time_hari"] - transaksi["sla_kirim_hari"]).clip(lower=0)
    transaksi["is_retur"] = (transaksi["status_pesanan"] == "Retur").astype(int)
    transaksi["is_batal"] = (transaksi["status_pesanan"] == "Dibatalkan").astype(int)
    transaksi["is_terlambat"] = (transaksi["terlambat_hari"] > 0).astype(int)
    transaksi["is_terlambat_parah"] = (transaksi["terlambat_hari"] > 3).astype(int)
    transaksi["is_belum_lunas"] = transaksi["status_pembayaran"].isin(["Belum Lunas", "Sebagian"]).astype(int)
    transaksi["periode"] = transaksi["tanggal_transaksi"].dt.to_period("M")
    transaksi["tahun"] = transaksi["tanggal_transaksi"].dt.year
    return transaksi


def aktivitas(transaksi: pd.DataFrame) -> pd.DataFrame:
    """Baris yang dihitung sebagai aktivitas pelanggan: Selesai atau Retur."""
    return transaksi[transaksi["status_pesanan"].isin(STATUS_AKTIVITAS)].copy()


def atur_gaya() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
        }
    )


PALET = {
    "biru": "#2F6DB3",
    "oranye": "#E28A3B",
    "merah": "#C0473F",
    "hijau": "#3E8E6E",
    "abu": "#8A8F98",
    "ungu": "#7A5AA6",
}


def rupiah(nilai: float) -> str:
    """Format ringkas untuk judul grafik: Rp 5,64 M."""
    if abs(nilai) >= 1e12:
        return f"Rp {nilai / 1e12:.2f} T"
    if abs(nilai) >= 1e9:
        return f"Rp {nilai / 1e9:.2f} M"
    if abs(nilai) >= 1e6:
        return f"Rp {nilai / 1e6:.1f} jt"
    return f"Rp {nilai:,.0f}"
