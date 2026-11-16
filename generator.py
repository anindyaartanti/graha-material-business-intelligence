from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MULAI = date(2019, 1, 1)
SELESAI = date(2026, 10, 31)
N_BULAN = (SELESAI.year - MULAI.year) * 12 + (SELESAI.month - MULAI.month) + 1

BARIS_PER_TAHUN = {
    2019: 1780,
    2020: 1540,
    2021: 1900,
    2022: 2180,
    2023: 2320,
    2024: 2260,
    2025: 1800,
    2026: 1220,
}

INDEKS_HARGA = {
    2019: 1.00,
    2020: 1.03,
    2021: 1.08,
    2022: 1.15,
    2023: 1.22,
    2024: 1.30,
    2025: 1.38,
    2026: 1.46,
}

DISKON_TAHUN = {
    2019: 4.0,
    2020: 5.0,
    2021: 6.0,
    2022: 9.0,
    2023: 11.5,
    2024: 12.0,
    2025: 7.5,
    2026: 6.0,
}

LEBARAN = {
    2019: date(2019, 6, 5),
    2020: date(2020, 5, 24),
    2021: date(2021, 5, 13),
    2022: date(2022, 5, 2),
    2023: date(2023, 4, 22),
    2024: date(2024, 4, 10),
    2025: date(2025, 3, 31),
    2026: date(2026, 3, 20),
}

BOBOT_BULAN = np.array([0.90, 0.87, 1.00, 1.02, 1.03, 1.00, 0.95, 1.00, 1.00, 1.02, 1.02, 1.16])

WILAYAH = [
    ("Bandung", "Jawa Barat", "Jawa", "Zona 1", 3),
    ("Bekasi", "Jawa Barat", "Jawa", "Zona 1", 2),
    ("Bogor", "Jawa Barat", "Jawa", "Zona 1", 3),
    ("Depok", "Jawa Barat", "Jawa", "Zona 1", 2),
    ("Cimahi", "Jawa Barat", "Jawa", "Zona 1", 3),
    ("Jakarta Selatan", "DKI Jakarta", "Jawa", "Zona 1", 2),
    ("Jakarta Timur", "DKI Jakarta", "Jawa", "Zona 1", 2),
    ("Jakarta Barat", "DKI Jakarta", "Jawa", "Zona 1", 2),
    ("Tangerang", "Banten", "Jawa", "Zona 1", 2),
    ("Serang", "Banten", "Jawa", "Zona 2", 3),
    ("Semarang", "Jawa Tengah", "Jawa", "Zona 2", 3),
    ("Surakarta", "Jawa Tengah", "Jawa", "Zona 2", 3),
    ("Magelang", "Jawa Tengah", "Jawa", "Zona 2", 4),
    ("Surabaya", "Jawa Timur", "Jawa", "Zona 2", 3),
    ("Malang", "Jawa Timur", "Jawa", "Zona 2", 4),
    ("Sidoarjo", "Jawa Timur", "Jawa", "Zona 2", 3),
    ("Medan", "Sumatera Utara", "Sumatera", "Zona 3", 6),
    ("Palembang", "Sumatera Selatan", "Sumatera", "Zona 3", 6),
    ("Padang", "Sumatera Barat", "Sumatera", "Zona 3", 7),
    ("Pekanbaru", "Riau", "Sumatera", "Zona 3", 6),
    ("Balikpapan", "Kalimantan Timur", "Kalimantan", "Zona 4", 7),
    ("Banjarmasin", "Kalimantan Selatan", "Kalimantan", "Zona 4", 8),
    ("Makassar", "Sulawesi Selatan", "Sulawesi", "Zona 4", 7),
    ("Manado", "Sulawesi Utara", "Sulawesi", "Zona 4", 8),
]

KATEGORI = [
    ("Material Dasar", ""),
    ("Besi dan Baja", "Material Dasar"),
    ("Agregat", "Material Dasar"),
    ("Material Dinding", "Material Dasar"),
    ("Atap dan Rangka", "Material Dasar"),
    ("Finishing", ""),
    ("Cat", "Finishing"),
    ("Keramik dan Lantai", "Finishing"),
    ("Instalasi", ""),
    ("Alat dan Perlengkapan", ""),
]

PRODUK = {
    "Besi dan Baja": [
        ("Besi Beton Polos 8mm", "batang", 62000, 0.12),
        ("Besi Beton Polos 10mm", "batang", 95000, 0.12),
        ("Besi Beton Ulir 12mm", "batang", 138000, 0.12),
        ("Besi Beton Ulir 16mm", "batang", 245000, 0.12),
        ("Wiremesh M6", "lembar", 385000, 0.11),
        ("Wiremesh M8", "lembar", 520000, 0.11),
        ("Baja Ringan C75", "batang", 78000, 0.13),
        ("Hollow Galvanis 2x4", "batang", 68000, 0.13),
        ("Besi Siku 40x40", "batang", 92000, 0.12),
        ("Kawat Bendrat", "roll", 28000, 0.15),
    ],
    "Agregat": [
        ("Pasir Pasang", "m3", 210000, 0.25),
        ("Pasir Beton", "m3", 235000, 0.24),
        ("Batu Split 1-2", "m3", 265000, 0.24),
        ("Batu Belah", "m3", 195000, 0.26),
        ("Sirtu", "m3", 175000, 0.26),
        ("Tanah Urug", "m3", 120000, 0.30),
        ("Batu Kali", "m3", 205000, 0.25),
        ("Pasir Halus Plester", "m3", 225000, 0.25),
        ("Abu Batu", "m3", 180000, 0.26),
        ("Screening", "m3", 150000, 0.27),
    ],
    "Material Dinding": [
        ("Semen Portland 40kg", "sak", 58000, 0.14),
        ("Semen Portland 50kg", "sak", 68000, 0.13),
        ("Semen Putih 40kg", "sak", 78000, 0.15),
        ("Mortar Instan Plester", "sak", 72000, 0.17),
        ("Mortar Perekat Bata Ringan", "sak", 85000, 0.17),
        ("Bata Merah", "pcs", 1200, 0.28),
        ("Bata Ringan Hebel", "m3", 620000, 0.18),
        ("Batako", "pcs", 2800, 0.25),
        ("Panel Lantai", "m3", 1100000, 0.15),
        ("Semen Sumur", "sak", 52000, 0.16),
    ],
    "Atap dan Rangka": [
        ("Genteng Metal", "lembar", 78000, 0.16),
        ("Genteng Beton", "pcs", 8500, 0.22),
        ("Spandek 0.3mm", "lembar", 165000, 0.15),
        ("Seng Gelombang", "lembar", 95000, 0.16),
        ("Talang Air PVC", "batang", 42000, 0.20),
        ("Nok Genteng", "pcs", 12000, 0.22),
        ("Reng Baja Ringan", "batang", 78000, 0.17),
        ("Insulasi Atap", "roll", 210000, 0.18),
        ("Sekrup Atap", "dus", 38000, 0.24),
        ("Lisplang GRC", "lembar", 68000, 0.19),
    ],
    "Cat": [
        ("Cat Tembok 5kg", "kaleng", 145000, 0.20),
        ("Cat Tembok 25kg", "pail", 620000, 0.18),
        ("Cat Kayu 1kg", "kaleng", 78000, 0.22),
        ("Cat Besi 1kg", "kaleng", 82000, 0.22),
        ("Plamir Tembok", "sak", 45000, 0.24),
        ("Cat Dasar Alkali", "kaleng", 95000, 0.21),
        ("Kuas 3 inci", "pcs", 18000, 0.30),
        ("Roll Cat", "pcs", 25000, 0.29),
        ("Tiner", "galon", 65000, 0.23),
        ("Waterproofing", "kaleng", 175000, 0.22),
    ],
    "Keramik dan Lantai": [
        ("Keramik 40x40", "dus", 68000, 0.19),
        ("Keramik 60x60", "dus", 145000, 0.18),
        ("Granit 60x60", "dus", 185000, 0.17),
        ("Keramik Dinding 25x40", "dus", 72000, 0.19),
        ("Nat Keramik", "sak", 35000, 0.26),
        ("Lem Keramik", "sak", 58000, 0.22),
        ("Plint Keramik", "dus", 48000, 0.21),
        ("Keramik 80x80", "dus", 265000, 0.16),
        ("Granit 120x60", "dus", 340000, 0.15),
        ("Skirting", "batang", 32000, 0.23),
    ],
    "Instalasi": [
        ("Pipa PVC 3/4 inci", "batang", 32000, 0.21),
        ("Pipa PVC 4 inci", "batang", 78000, 0.19),
        ("Knee PVC 3/4", "pcs", 4500, 0.30),
        ("Kran Air", "pcs", 55000, 0.24),
        ("Closet Duduk", "unit", 850000, 0.16),
        ("Closet Jongkok", "unit", 320000, 0.18),
        ("Kabel NYM 1.5mm", "roll", 385000, 0.15),
        ("Kabel NYM 2.5mm", "roll", 545000, 0.14),
        ("Saklar Ganda", "pcs", 28000, 0.28),
        ("MCB 10A", "pcs", 42000, 0.26),
    ],
    "Alat dan Perlengkapan": [
        ("Sekop", "pcs", 45000, 0.27),
        ("Meteran 5m", "pcs", 32000, 0.30),
        ("Helm Proyek", "pcs", 28000, 0.33),
        ("Tangga Aluminium 2m", "unit", 720000, 0.20),
        ("Gerobak Sorong", "unit", 780000, 0.19),
        ("Cangkul", "pcs", 55000, 0.26),
        ("Waterpass", "pcs", 95000, 0.24),
        ("Sarung Tangan", "pasang", 12000, 0.35),
        ("Mesin Bor", "unit", 450000, 0.21),
        ("Gergaji Kayu", "pcs", 38000, 0.29),
    ],
}

SATUAN_MAKS = {
    "m3": 40,
    "sak": 400,
    "batang": 400,
    "lembar": 250,
    "pcs": 1200,
    "dus": 200,
    "roll": 60,
    "kaleng": 80,
    "pail": 40,
    "unit": 10,
    "pasang": 200,
    "galon": 30,
    "set": 20,
}

NAMA_TOKO_A = [
    "Sinar", "Jaya", "Makmur", "Sumber", "Berkah", "Cahaya", "Karya", "Prima",
    "Sentosa", "Anugerah", "Bintang", "Tunas", "Mitra", "Sahabat", "Mulia",
    "Rejeki", "Abadi", "Mega", "Global", "Cahaya Baru",
]
NAMA_TOKO_B = [
    "Jaya", "Abadi", "Sejahtera", "Makmur", "Sentosa", "Utama", "Bersama",
    "Mandiri", "Perkasa", "Bangunan", "Material", "Niaga", "Gemilang",
]
NAMA_CV_A = [
    "Karya", "Sinar", "Mitra", "Bumi", "Cipta", "Bangun", "Anugerah", "Sumber",
    "Tata", "Rekatama",
]
NAMA_CV_B = [
    "Beton", "Perkasa", "Abadi", "Sejahtera", "Konstruksi", "Persada", "Utama",
    "Mandiri", "Pratama", "Sentosa",
]
NAMA_PT_A = ["Griya", "Nusa", "Bumi", "Cipta", "Karya", "Graha", "Panca", "Surya"]
NAMA_PT_B = [
    "Asri", "Persada", "Konstruksi", "Properti", "Nusantara", "Abadi",
    "Sejahtera", "Utama",
]
NAMA_SALES = [
    "Agus Prasetyo", "Budi Santoso", "Citra Lestari", "Dedi Kurniawan",
    "Eka Wulandari", "Fajar Nugroho", "Gita Permata", "Hendra Wijaya",
    "Indra Saputra", "Joko Susilo", "Kartika Sari", "Lukman Hakim",
    "Maya Anggraini", "Nanda Pratama", "Oki Ramadhan", "Putri Handayani",
    "Rizky Aditya", "Siti Rahmawati", "Taufik Hidayat", "Yuni Astuti",
]

KOTA_ALIAS = {
    "Bandung": ["bandung", "Kota Bandung", "BDG"],
    "Bekasi": ["bekasi", "Kota Bekasi"],
    "Bogor": ["bogor", "Kota Bogor"],
    "Depok": ["depok", "Kota Depok"],
    "Cimahi": ["cimahi", "Kota Cimahi"],
    "Jakarta Selatan": ["Jakarta Selatan", "jakarta selatan", "Jkt Selatan", "Jaksel"],
    "Jakarta Timur": ["Jakarta Timur", "jakarta timur", "Jkt Timur", "Jaktim"],
    "Jakarta Barat": ["Jakarta Barat", "jakarta barat", "Jkt Barat", "Jakbar"],
    "Tangerang": ["tangerang", "Kota Tangerang", "TNG"],
    "Serang": ["serang", "Kota Serang"],
    "Semarang": ["semarang", "Kota Semarang", "SMG"],
    "Surakarta": ["Surakarta", "surakarta", "Solo"],
    "Magelang": ["magelang", "Kota Magelang"],
    "Surabaya": ["surabaya", "Kota Surabaya", "SBY"],
    "Malang": ["malang", "Kota Malang"],
    "Sidoarjo": ["sidoarjo", "Kabupaten Sidoarjo"],
    "Medan": ["medan", "Kota Medan", "MDN"],
    "Palembang": ["palembang", "Kota Palembang", "PLM"],
    "Padang": ["padang", "Kota Padang"],
    "Pekanbaru": ["pekanbaru", "Kota Pekanbaru", "PKU"],
    "Balikpapan": ["balikpapan", "Kota Balikpapan", "BPP"],
    "Banjarmasin": ["banjarmasin", "Kota Banjarmasin", "BJM"],
    "Makassar": ["makassar", "Kota Makassar", "MKS"],
    "Manado": ["manado", "Kota Manado", "MDO"],
}

PROVINSI_ALIAS = {
    "Jawa Barat": ["Jabar", "JAWA BARAT", "jawa barat", "Jawa Barat "],
    "DKI Jakarta": ["DKI JAKARTA", "Jakarta", "dki jakarta"],
    "Banten": ["BANTEN", "Banten "],
    "Jawa Tengah": ["Jateng", "JAWA TENGAH", "jawa tengah"],
    "Jawa Timur": ["Jatim", "JAWA TIMUR", "jawa timur"],
    "Sumatera Utara": ["Sumut", "SUMATERA UTARA"],
    "Sumatera Selatan": ["Sumsel", "SUMATERA SELATAN"],
    "Sumatera Barat": ["Sumbar", "SUMATERA BARAT"],
    "Riau": ["RIAU", "riau"],
    "Kalimantan Timur": ["Kaltim", "KALIMANTAN TIMUR"],
    "Kalimantan Selatan": ["Kalsel", "KALIMANTAN SELATAN"],
    "Sulawesi Selatan": ["Sulsel", "SULAWESI SELATAN"],
    "Sulawesi Utara": ["Sulut", "SULAWESI UTARA"],
}

AFINITAS = {
    "Toko Bangunan": {
        "Besi dan Baja": 1.0,
        "Agregat": 0.8,
        "Material Dinding": 1.4,
        "Atap dan Rangka": 1.0,
        "Cat": 1.1,
        "Keramik dan Lantai": 0.9,
        "Instalasi": 0.9,
        "Alat dan Perlengkapan": 0.7,
    },
    "Kontraktor": {
        "Besi dan Baja": 1.5,
        "Agregat": 1.3,
        "Material Dinding": 1.5,
        "Atap dan Rangka": 1.1,
        "Cat": 0.6,
        "Keramik dan Lantai": 0.6,
        "Instalasi": 0.7,
        "Alat dan Perlengkapan": 1.0,
    },
    "Developer": {
        "Besi dan Baja": 0.9,
        "Agregat": 0.7,
        "Material Dinding": 1.0,
        "Atap dan Rangka": 0.9,
        "Cat": 1.2,
        "Keramik dan Lantai": 1.6,
        "Instalasi": 1.3,
        "Alat dan Perlengkapan": 0.5,
    },
    "Korporat": {
        "Besi dan Baja": 0.8,
        "Agregat": 0.4,
        "Material Dinding": 0.8,
        "Atap dan Rangka": 0.7,
        "Cat": 1.0,
        "Keramik dan Lantai": 1.2,
        "Instalasi": 1.4,
        "Alat dan Perlengkapan": 1.3,
    },
}

FAKTOR_JENIS = {"Toko Bangunan": 1.25, "Kontraktor": 0.75, "Developer": 0.60, "Korporat": 0.70}
FAKTOR_REGION = {"Jawa": 1.00, "Sumatera": 1.40, "Kalimantan": 1.00, "Sulawesi": 0.75}
FAKTOR_KANAL = {
    "Referral": 0.85,
    "Kunjungan Sales": 0.95,
    "Pameran": 1.25,
    "Telemarketing": 1.20,
    "Website": 1.10,
}
FAKTOR_SEGMEN = {"Platinum": 0.55, "Gold": 0.80, "Silver": 1.10, "Bronze": 1.35}
CHURN_PER_TAHUN = {2019: 1, 2020: 0, 2021: 2, 2022: 2, 2023: 4, 2024: 7, 2025: 10, 2026: 7}
BOBOT_CHURN_BULAN = np.array(
    [0.06, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.10, 0.10]
)
PELUANG_TELAT = {
    2019: 0.055,
    2020: 0.060,
    2021: 0.065,
    2022: 0.075,
    2023: 0.090,
    2024: 0.130,
    2025: 0.190,
    2026: 0.190,
}

BULAN_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
    "Agustus", "September", "Oktober", "November", "Desember",
]


@dataclass
class Konfigurasi:
    seed: int = 42
    keluaran: Path = Path("data/raw")
    baris: int = 15000
    dirt: str = "medium"

    def __post_init__(self) -> None:
        self.keluaran = Path(self.keluaran)


def bulan_ke_tanggal(idx: int) -> date:
    tahun = MULAI.year + idx // 12
    bulan = MULAI.month + idx % 12
    return date(tahun, bulan, 1)


def indeks_bulan(tgl: date) -> int:
    return (tgl.year - MULAI.year) * 12 + (tgl.month - MULAI.month)


def akhir_bulan(tgl: date) -> date:
    if tgl.month == 12:
        return date(tgl.year, 12, 31)
    return date(tgl.year, tgl.month + 1, 1) - timedelta(days=1)


def rupiah(nilai: float) -> str:
    return "Rp " + f"{nilai:,.0f}".replace(",", ".")


def bulat_ratus(nilai: float) -> int:
    return int(round(nilai / 100.0) * 100)


def bangun_wilayah() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id_wilayah": f"WIL-{i:02d}",
                "provinsi": provinsi,
                "kota": kota,
                "region": region,
                "zona_kirim": zona,
                "sla_kirim_hari": sla,
            }
            for i, (kota, provinsi, region, zona, sla) in enumerate(WILAYAH, start=1)
        ]
    )


def bangun_kategori() -> pd.DataFrame:
    peta_id = {nama: f"KAT-{i:02d}" for i, (nama, _) in enumerate(KATEGORI, start=1)}
    baris = []
    for i, (nama, induk) in enumerate(KATEGORI, start=1):
        baris.append(
            {
                "id_kategori": f"KAT-{i:02d}",
                "nama_kategori": nama,
                "id_kategori_induk": peta_id[induk] if induk else "",
                "deskripsi": f"Kategori {nama.lower()} untuk kebutuhan proyek dan renovasi",
            }
        )
    return pd.DataFrame(baris)


def bangun_produk(rng: np.random.Generator) -> pd.DataFrame:
    peta_kategori = {nama: f"KAT-{i:02d}" for i, (nama, _) in enumerate(KATEGORI, start=1)}
    baris = []
    nomor = 1
    for nama_kategori, daftar in PRODUK.items():
        for nama, satuan, modal, margin in daftar:
            baris.append(
                {
                    "id_produk": f"PRD-{nomor:04d}",
                    "nama_produk": nama,
                    "id_kategori": peta_kategori[nama_kategori],
                    "satuan": satuan,
                    "harga_modal": int(modal),
                    "harga_jual_acuan": bulat_ratus(modal * (1 + margin)),
                    "berat_kg": round(float(rng.uniform(0.5, 60.0)), 2),
                    "status_aktif": "Aktif",
                }
            )
            nomor += 1
    return pd.DataFrame(baris)


def bangun_tenaga_penjualan(rng: np.random.Generator) -> pd.DataFrame:
    region_sales = ["Jawa"] * 8 + ["Sumatera"] * 3 + ["Kalimantan"] * 2 + ["Sulawesi"] * 2
    keluar = {
        2: date(2026, 5, 31),
        4: date(2025, 3, 31),
        6: date(2026, 1, 31),
        9: date(2025, 7, 31),
        12: date(2025, 11, 30),
    }
    cabang = {
        "Jawa": "Cabang Bandung",
        "Sumatera": "Cabang Medan",
        "Kalimantan": "Cabang Balikpapan",
        "Sulawesi": "Cabang Makassar",
    }
    baris = []
    for i, region in enumerate(region_sales, start=1):
        baris.append(
            {
                "id_sales": f"SL-{i:02d}",
                "nama_sales": NAMA_SALES[i - 1],
                "cabang": cabang[region],
                "region": region,
                "tanggal_masuk": date(2017, 1, 1) + timedelta(days=int(rng.integers(0, 900))),
                "tanggal_keluar": keluar.get(i),
                "target_bulanan": int(rng.integers(25, 60) * 1_000_000),
                "status": "Resign" if i in keluar else "Aktif",
            }
        )
    return pd.DataFrame(baris)


def bangun_pelanggan(
    rng: np.random.Generator, wilayah: pd.DataFrame, sales: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    jenis_pelanggan = (
        ["Toko Bangunan"] * 55
        + ["Kontraktor"] * 25
        + ["Developer"] * 8
        + ["Korporat"] * 12
    )
    rng.shuffle(jenis_pelanggan)

    kandidat_nama = []
    for a in NAMA_TOKO_A:
        for b in NAMA_TOKO_B:
            kandidat_nama.append(f"Toko {a} {b}")
    for a in NAMA_CV_A:
        for b in NAMA_CV_B:
            kandidat_nama.append(f"CV {a} {b}")
    for a in NAMA_PT_A:
        for b in NAMA_PT_B:
            kandidat_nama.append(f"PT {a} {b}")
    kandidat_nama = list(dict.fromkeys(kandidat_nama))
    rng.shuffle(kandidat_nama)
    nama_terpilih = kandidat_nama[:100]

    peta_region = {
        region: wilayah[wilayah["region"] == region] for region in ["Jawa", "Sumatera", "Kalimantan", "Sulawesi"]
    }
    bobot_region = {"Jawa": 0.62, "Sumatera": 0.20, "Kalimantan": 0.09, "Sulawesi": 0.09}

    pengganti = {}
    keluar_sales = {}
    for _, baris_sales in sales.iterrows():
        if baris_sales["status"] == "Resign":
            keluar_sales[baris_sales["id_sales"]] = baris_sales["tanggal_keluar"]
            sejenis = sales[
                (sales["region"] == baris_sales["region"]) & (sales["status"] == "Aktif")
            ]
            pengganti[baris_sales["id_sales"]] = str(sejenis.iloc[0]["id_sales"])

    tanggal_registrasi = [
        date(2017, 3, 1) + timedelta(days=int(rng.integers(0, 640))) for _ in range(22)
    ]
    jadwal_akuisisi = {2019: 26, 2020: 15, 2021: 15, 2022: 10, 2023: 8, 2024: 3, 2025: 1}
    for tahun, jumlah in jadwal_akuisisi.items():
        for _ in range(jumlah):
            tanggal_registrasi.append(
                date(tahun, int(rng.integers(1, 13)), int(rng.integers(1, 28)))
            )
    rng.shuffle(tanggal_registrasi)

    kanal_pilihan = ["Kunjungan Sales", "Referral", "Pameran", "Telemarketing", "Website"]
    bobot_kanal = [0.34, 0.26, 0.18, 0.12, 0.10]
    termin_pilihan = ["COD", "30 Hari", "45 Hari", "60 Hari"]
    bobot_termin = [0.40, 0.35, 0.15, 0.10]
    segmen_per_jenis = {
        "Toko Bangunan": (["Silver", "Bronze"], [0.55, 0.45]),
        "Kontraktor": (["Gold", "Silver"], [0.45, 0.55]),
        "Developer": (["Platinum", "Gold"], [0.60, 0.40]),
        "Korporat": (["Gold", "Silver"], [0.50, 0.50]),
    }
    faktor_segmen_beli = {"Platinum": 2.2, "Gold": 1.5, "Silver": 1.0, "Bronze": 0.7}

    baris = []
    profil = {}
    for i in range(100):
        region = str(rng.choice(list(bobot_region), p=list(bobot_region.values())))
        wilayah_pilih = peta_region[region].sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
        jenis = jenis_pelanggan[i]
        pilihan_segmen, bobot_segmen = segmen_per_jenis[jenis]
        segmen = str(rng.choice(pilihan_segmen, p=bobot_segmen))
        sales_region = sales[sales["region"] == region]
        sales_pilih = sales_region.sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
        kanal = str(rng.choice(kanal_pilihan, p=bobot_kanal))
        termin = str(rng.choice(termin_pilihan, p=bobot_termin))
        id_pelanggan = f"CUST-{i + 1:04d}"
        baris.append(
            {
                "id_pelanggan": id_pelanggan,
                "nama_pelanggan": nama_terpilih[i],
                "jenis_pelanggan": jenis,
                "segmen": segmen,
                "id_wilayah": wilayah_pilih["id_wilayah"],
                "id_sales": pengganti.get(str(sales_pilih["id_sales"]), str(sales_pilih["id_sales"])),
                "tanggal_registrasi": tanggal_registrasi[i],
                "kanal_akuisisi": kanal,
                "termin_pembayaran": termin,
                "limit_kredit": int(rng.integers(10, 80) * 5_000_000),
            }
        )
        profil[id_pelanggan] = {
            "bobot": float(np.exp(rng.normal(0.0, 0.55))) * faktor_segmen_beli[segmen],
            "dependensi_diskon": float(rng.beta(2.0, 4.0)),
            "prioritas_layanan": float(rng.beta(2.5, 2.5)),
            "region": region,
            "provinsi": wilayah_pilih["provinsi"],
            "jenis": jenis,
            "segmen": segmen,
            "kanal": kanal,
            "id_sales_awal": str(sales_pilih["id_sales"]),
            "id_sales_akhir": pengganti.get(
                str(sales_pilih["id_sales"]), str(sales_pilih["id_sales"])
            ),
            "tanggal_keluar_sales": keluar_sales.get(str(sales_pilih["id_sales"])),
            "id_wilayah": wilayah_pilih["id_wilayah"],
        }
    return pd.DataFrame(baris), profil


def simulasi_siklus(
    pelanggan: pd.DataFrame, profil: dict, sales: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict, list]:
    sales_keluar = {}
    for _, row in sales.iterrows():
        nilai = row["tanggal_keluar"]
        if nilai is not None and not pd.isna(nilai):
            sales_keluar[row["id_sales"]] = indeks_bulan(nilai)

    risiko = {}
    mulai = {}
    for _, row in pelanggan.iterrows():
        pid = row["id_pelanggan"]
        p = profil[pid]
        nilai = (
            FAKTOR_JENIS[p["jenis"]]
            * FAKTOR_REGION[p["region"]]
            * FAKTOR_KANAL[p["kanal"]]
            * FAKTOR_SEGMEN[p["segmen"]]
        )
        if p["provinsi"] == "Jawa Barat":
            nilai *= 1.30
        nilai *= 0.50 + 1.80 * (1 - p["prioritas_layanan"])
        nilai *= 0.70 + 1.30 * p["dependensi_diskon"]
        risiko[pid] = nilai
        mulai[pid] = max(indeks_bulan(row["tanggal_registrasi"]), 0)

    siklus = {pid: {"mulai": mulai[pid], "churn": None} for pid in mulai}
    catatan = []
    for tahun, jumlah in CHURN_PER_TAHUN.items():
        idx_tahun = [i for i in range(N_BULAN) if bulan_ke_tanggal(i).year == tahun]
        if not idx_tahun or jumlah == 0:
            continue
        bobot = np.array([BOBOT_CHURN_BULAN[bulan_ke_tanggal(i).month - 1] for i in idx_tahun])
        bobot = bobot / bobot.sum()
        alokasi = rng.multinomial(jumlah, bobot)
        for idx, k in zip(idx_tahun, alokasi):
            for _ in range(int(k)):
                kandidat = []
                bobot_kandidat = []
                for pid, s in siklus.items():
                    if s["churn"] is not None:
                        continue
                    if not (s["mulai"] < idx):
                        continue
                    bobot_pid = risiko[pid] ** 1.4
                    if bulan_ke_tanggal(idx).year >= 2024:
                        p = profil[pid]
                        if p["jenis"] == "Toko Bangunan":
                            bobot_pid *= 1.30
                        if p["provinsi"] == "Jawa Barat":
                            bobot_pid *= 1.25
                        if p["region"] == "Sumatera":
                            bobot_pid *= 1.35
                        idx_keluar = sales_keluar.get(p["id_sales_awal"])
                        if idx_keluar is not None and 0 <= idx - idx_keluar <= 9:
                            bobot_pid *= 1.25
                    kandidat.append(pid)
                    bobot_kandidat.append(bobot_pid)
                if not kandidat:
                    break
                bobot_kandidat = np.array(bobot_kandidat, dtype=float)
                bobot_kandidat = bobot_kandidat / bobot_kandidat.sum()
                terpilih = str(rng.choice(kandidat, p=bobot_kandidat))
                siklus[terpilih]["churn"] = idx
                catatan.append(
                    {
                        "id_pelanggan": terpilih,
                        "tahun": tahun,
                        "tanggal_churn": bulan_ke_tanggal(idx),
                        "jenis": profil[terpilih]["jenis"],
                        "region": profil[terpilih]["region"],
                        "provinsi": profil[terpilih]["provinsi"],
                    }
                )

    pelanggan = pelanggan.copy()
    return siklus, catatan, pelanggan


def target_bulanan() -> np.ndarray:
    target = np.zeros(N_BULAN, dtype=int)
    for tahun, jumlah in BARIS_PER_TAHUN.items():
        idx_tahun = [i for i in range(N_BULAN) if bulan_ke_tanggal(i).year == tahun]
        bobot = []
        for i in idx_tahun:
            tgl = bulan_ke_tanggal(i)
            w = BOBOT_BULAN[tgl.month - 1]
            if tgl.year == 2020 and 3 <= tgl.month <= 6:
                w *= 0.70
            if tgl.year == 2020 and 7 <= tgl.month <= 8:
                w *= 0.85
            lebaran = LEBARAN[tgl.year]
            selisih = (tgl.year - lebaran.year) * 12 + (tgl.month - lebaran.month)
            if selisih == 0:
                w *= 1.28
            elif selisih == 1:
                w *= 0.90
            bobot.append(w)
        bobot = np.array(bobot)
        bobot = bobot / bobot.sum()
        mentah = bobot * jumlah
        dasar = np.floor(mentah).astype(int)
        sisa = jumlah - dasar.sum()
        urutan = np.argsort(-(mentah - dasar))
        for k in range(sisa):
            dasar[urutan[k % len(urutan)]] += 1
        for pos, i in enumerate(idx_tahun):
            target[i] = dasar[pos]
    return target


def bangun_transaksi(
    rng: np.random.Generator,
    pelanggan: pd.DataFrame,
    produk: pd.DataFrame,
    siklus: dict,
    profil: dict,
) -> pd.DataFrame:
    target = target_bulanan()
    pelanggan_idx = pelanggan.reset_index(drop=True)
    peta_pelanggan = pelanggan_idx.set_index("id_pelanggan")
    produk_idx = produk.reset_index(drop=True)
    peta_kategori = {f"KAT-{i:02d}": nama for i, (nama, _) in enumerate(KATEGORI, start=1)}
    produk_idx["nama_kategori"] = produk_idx["id_kategori"].map(peta_kategori)
    saham_produk = {}
    for kat, grup in produk_idx.groupby("nama_kategori"):
        bobot = 0.62 ** np.arange(len(grup))
        saham_produk[kat] = grup.assign(bobot=bobot).reset_index(drop=True)

    kategori_jenis = {}
    for jenis, peta in AFINITAS.items():
        nama = np.array(list(peta.keys()))
        bobot = np.array(list(peta.values()), dtype=float)
        kategori_jenis[jenis] = (nama, bobot / bobot.sum())

    metode = ["Transfer Bank", "Tunai", "Virtual Account", "Cek Giro", "QRIS"]
    bobot_metode = np.array([0.46, 0.24, 0.16, 0.09, 0.05])
    status_bayar = ["Lunas", "Sebagian", "Belum Lunas"]
    bobot_bayar = np.array([0.76, 0.11, 0.13])

    baris = []
    nomor_transaksi = 0
    for idx_bulan in range(N_BULAN):
        tgl_bulan = bulan_ke_tanggal(idx_bulan)
        tahun = tgl_bulan.year
        jumlah_baris = int(target[idx_bulan])
        if jumlah_baris == 0:
            continue
        aktif = []
        bobot = []
        for pid, s in siklus.items():
            if s["mulai"] <= idx_bulan and (s["churn"] is None or idx_bulan < s["churn"]):
                aktif.append(pid)
                bobot.append(profil[pid]["bobot"])
        if not aktif:
            continue
        bobot = np.array(bobot, dtype=float)
        bobot = bobot / bobot.sum()
        alokasi = rng.multinomial(jumlah_baris, bobot)
        nomor_urut_pesanan = 0
        akhir = min(akhir_bulan(tgl_bulan), SELESAI)
        hari = pd.date_range(tgl_bulan, akhir, freq="D")
        bobot_hari = np.array(
            [1.0 if h.weekday() < 5 else 0.65 if h.weekday() == 5 else 0.30 for h in hari]
        )
        bobot_hari = bobot_hari / bobot_hari.sum()
        for pid, jumlah_pelanggan in zip(aktif, alokasi):
            if jumlah_pelanggan <= 0:
                continue
            info = profil[pid]
            n_pesanan = max(1, int(round(jumlah_pelanggan / rng.uniform(1.6, 2.8))))
            n_pesanan = min(n_pesanan, jumlah_pelanggan)
            tambahan = rng.multinomial(
                jumlah_pelanggan - n_pesanan, np.ones(n_pesanan) / n_pesanan
            )
            isi = tambahan + 1
            for k in range(n_pesanan):
                n_baris = int(isi[k])
                tanggal = hari[rng.choice(len(hari), p=bobot_hari)].date()
                nomor_urut_pesanan += 1
                nomor_pesanan = f"SO-{tanggal:%Y%m}-{nomor_urut_pesanan:05d}"
                metode_bayar = str(rng.choice(metode, p=bobot_metode))
                termin = peta_pelanggan.loc[pid, "termin_pembayaran"]
                nomor_wilayah = int(str(info["id_wilayah"]).split("-")[1])
                sla = WILAYAH[nomor_wilayah - 1][4]
                faktor_wilayah = (
                    1.35 if info["region"] == "Sumatera"
                    else 1.12 if info["region"] == "Kalimantan"
                    else 1.0
                )
                faktor_prioritas = 2.0 - 1.2 * info["prioritas_layanan"]
                sisa_bulan = (
                    siklus[pid]["churn"] - idx_bulan
                    if siklus[pid]["churn"] is not None
                    else 99
                )
                faktor_retensi = 1.8 if 0 <= sisa_bulan <= 6 else 1.5 if 0 <= sisa_bulan <= 12 else 1.0
                if rng.random() < min(
                    PELUANG_TELAT[tahun]
                    * faktor_wilayah
                    * faktor_prioritas
                    * faktor_retensi,
                    0.70,
                ):
                    keterlambatan = int(rng.integers(1, 8))
                else:
                    keterlambatan = 0
                lead_time = sla + keterlambatan
                tanggal_kirim = min(tanggal + timedelta(days=lead_time), SELESAI)
                faktor_retur = 1.8 if 0 <= sisa_bulan <= 6 else 1.0
                if rng.random() < (0.030 + (0.030 if keterlambatan >= 4 else 0.0)) * faktor_retur:
                    status = "Retur"
                elif rng.random() < 0.028:
                    status = "Dibatalkan"
                else:
                    status = "Selesai"
                status_pembayaran = (
                    "Tidak Berlaku"
                    if status == "Dibatalkan"
                    else str(rng.choice(status_bayar, p=bobot_bayar))
                )
                nama_kat, bobot_kat = kategori_jenis[info["jenis"]]
                produk_pakai = []
                for _ in range(n_baris):
                    kat = str(rng.choice(nama_kat, p=bobot_kat))
                    kandidat = saham_produk[kat]
                    tersedia = kandidat[~kandidat["id_produk"].isin(produk_pakai)]
                    if tersedia.empty:
                        tersedia = kandidat
                    bobot_produk = tersedia["bobot"].to_numpy(dtype=float)
                    pilih = tersedia.iloc[
                        rng.choice(len(tersedia), p=bobot_produk / bobot_produk.sum())
                    ]
                    produk_pakai.append(pilih["id_produk"])
                    harga = bulat_ratus(
                        pilih["harga_jual_acuan"] * INDEKS_HARGA[tahun] * rng.uniform(0.985, 1.015)
                    )
                    qty_dasar = 2_300_000 / max(pilih["harga_jual_acuan"], 1)
                    jumlah = int(round(qty_dasar * float(rng.lognormal(0.0, 0.45))))
                    jumlah = max(1, min(jumlah, SATUAN_MAKS[pilih["satuan"]]))
                    diskon = DISKON_TAHUN[tahun] + info["dependensi_diskon"] * 10.0
                    diskon += float(rng.normal(0, 1.2))
                    if tahun >= 2025 and info["dependensi_diskon"] > 0.55:
                        diskon -= 4.0
                    diskon = float(np.clip(diskon, 0.0, 30.0))
                    total = bulat_ratus(jumlah * harga * (1 - diskon / 100.0))
                    ongkir = hitung_ongkir(info["region"], total, rng)
                    keluar_sales_info = info["tanggal_keluar_sales"]
                    id_sales_baris = (
                        info["id_sales_akhir"]
                        if keluar_sales_info is not None and not pd.isna(keluar_sales_info)
                        and tanggal > keluar_sales_info
                        else info["id_sales_awal"]
                    )
                    nomor_transaksi += 1
                    baris.append(
                        {
                            "id_transaksi": f"TRX-{nomor_transaksi:07d}",
                            "nomor_pesanan": nomor_pesanan,
                            "tanggal_transaksi": tanggal,
                            "id_pelanggan": pid,
                            "id_produk": pilih["id_produk"],
                            "id_sales": id_sales_baris,
                            "jumlah": jumlah,
                            "harga_satuan": harga,
                            "diskon_persen": round(diskon, 1),
                            "ongkos_kirim": ongkir,
                            "metode_pembayaran": metode_bayar,
                            "termin_pembayaran": termin,
                            "status_pesanan": status,
                            "tanggal_kirim": tanggal_kirim,
                            "status_pembayaran": status_pembayaran,
                        }
                    )
    return pd.DataFrame(baris)


def hitung_ongkir(region: str, total: int, rng: np.random.Generator) -> int:
    if region == "Jawa" and total >= 12_000_000:
        return 0
    batas = {
        "Jawa": (0, 250_000),
        "Sumatera": (150_000, 600_000),
        "Kalimantan": (250_000, 800_000),
        "Sulawesi": (250_000, 900_000),
    }[region]
    return int(round(rng.integers(batas[0], batas[1] + 1) / 1000.0) * 1000)


def acak_format_tanggal(nilai: pd.Series, rng: np.random.Generator, porsi: float) -> pd.Series:
    hasil = nilai.map(lambda x: x.strftime("%Y-%m-%d") if isinstance(x, date) else str(x))
    jumlah = int(len(hasil) * porsi)
    posisi = rng.choice(len(hasil), size=jumlah, replace=False)
    pola = rng.choice(4, size=jumlah, p=[0.35, 0.25, 0.22, 0.18])
    for i, p in zip(posisi, pola):
        d = date.fromisoformat(str(hasil.iat[i])[:10])
        if p == 0:
            hasil.iat[i] = d.strftime("%d/%m/%Y")
        elif p == 1:
            hasil.iat[i] = d.strftime("%Y/%m/%d")
        elif p == 2:
            hasil.iat[i] = d.strftime("%d-%b-%y")
        else:
            hasil.iat[i] = f"{d.day} {BULAN_ID[d.month - 1]} {d.year}"
    return hasil


def teks_uang(nilai: float, gaya: int) -> str:
    if gaya == 0:
        return rupiah(nilai)
    if gaya == 1:
        return f"{nilai:,.0f}"
    return f"{nilai:.2f}".replace(".", ",")


def noda_nilai(
    df: pd.DataFrame, rng: np.random.Generator, kolom_tanggal: list, kolom_uang: list
) -> pd.DataFrame:
    hasil = df.astype(object).copy()
    n = len(hasil)
    hilang = ["", "N/A", "-", "NULL"]
    for kolom in kolom_tanggal:
        hasil[kolom] = acak_format_tanggal(hasil[kolom], rng, 0.35)
    for kolom in kolom_uang:
        porsi = int(n * 0.14)
        posisi = rng.choice(n, size=porsi, replace=False)
        gaya = rng.choice(3, size=porsi, p=[0.5, 0.3, 0.2])
        nilai_asli = pd.to_numeric(hasil[kolom], errors="coerce")
        kolom_idx = hasil.columns.get_loc(kolom)
        for i, g in zip(posisi, gaya):
            hasil.iat[i, kolom_idx] = teks_uang(float(nilai_asli.iat[i]), int(g))
    for kolom, porsi in {
        "harga_satuan": 0.025,
        "jumlah": 0.015,
        "diskon_persen": 0.020,
        "ongkos_kirim": 0.020,
        "metode_pembayaran": 0.020,
    }.items():
        posisi = rng.choice(n, size=int(n * porsi), replace=False)
        nilai = rng.choice(hilang, size=len(posisi))
        kolom_idx = hasil.columns.get_loc(kolom)
        for i, v in zip(posisi, nilai):
            hasil.iat[i, kolom_idx] = v
    kolom_idx = hasil.columns.get_loc("id_pelanggan")
    for i in rng.choice(n, size=int(n * 0.015), replace=False):
        teks = str(hasil.iat[i, kolom_idx])
        if teks.startswith("CUST-"):
            hasil.iat[i, kolom_idx] = f"CUST-{int(teks.split('-')[1])}"
    kolom_idx = hasil.columns.get_loc("id_produk")
    for i in rng.choice(n, size=int(n * 0.010), replace=False):
        teks = str(hasil.iat[i, kolom_idx])
        if teks.startswith("PRD-"):
            hasil.iat[i, kolom_idx] = f"PRD-{int(teks.split('-')[1])}"
    for i in rng.choice(n, size=int(n * 0.18), replace=False):
        kolom = str(rng.choice(["status_pesanan", "metode_pembayaran", "termin_pembayaran"]))
        kolom_idx = hasil.columns.get_loc(kolom)
        teks = str(hasil.iat[i, kolom_idx])
        varian = [teks.lower(), teks.upper() + " ", " " + teks, teks.replace(" ", "  ")]
        hasil.iat[i, kolom_idx] = varian[int(rng.integers(0, 4))]
    return hasil


def baris_sampah(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    kolom = list(df.columns)
    sampah = []

    def salin(n: int, ubah) -> None:
        for _, row in df.sample(n, random_state=int(rng.integers(0, 10**6))).iterrows():
            baru = row.to_dict()
            ubah(baru)
            sampah.append(baru)

    def apa_apa(_: dict) -> None:
        return None

    def ganti_id(baru: dict) -> None:
        baru["id_transaksi"] = f"TRX-{int(rng.integers(9000000, 9999999))}"

    salin(300, apa_apa)
    salin(120, ganti_id)

    def qty_aneh(baru: dict) -> None:
        ganti_id(baru)
        baru["jumlah"] = 9999

    salin(40, qty_aneh)

    def tanggal_aneh(baru: dict) -> None:
        ganti_id(baru)
        tahun = 2027 if rng.random() < 0.5 else 2018
        baru["tanggal_transaksi"] = date(
            tahun, int(rng.integers(1, 13)), int(rng.integers(1, 28))
        ).strftime("%Y-%m-%d")

    salin(40, tanggal_aneh)

    def pelanggan_aneh(baru: dict) -> None:
        ganti_id(baru)
        baru["id_pelanggan"] = f"CUST-{int(rng.integers(9000, 9999))}"

    salin(50, pelanggan_aneh)

    def produk_aneh(baru: dict) -> None:
        ganti_id(baru)
        baru["id_produk"] = f"PRD-{int(rng.integers(9000, 9999))}"

    salin(40, produk_aneh)

    for _ in range(40):
        sampah.append(
            {
                **{k: "" for k in kolom},
                "id_transaksi": f"TRX-{int(rng.integers(9000000, 9999999))}",
            }
        )

    def tanggal_nol(baru: dict) -> None:
        ganti_id(baru)
        baru["tanggal_transaksi"] = "0000-00-00"

    salin(30, tanggal_nol)

    def qty_nol(baru: dict) -> None:
        ganti_id(baru)
        baru["jumlah"] = 0

    salin(40, qty_nol)
    return pd.DataFrame(sampah, columns=kolom)


def noda_dimensi(
    pelanggan: pd.DataFrame,
    produk: pd.DataFrame,
    kategori: pd.DataFrame,
    wilayah: pd.DataFrame,
    sales: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pel = pelanggan.astype(object).copy()
    pel["tanggal_registrasi"] = acak_format_tanggal(pel["tanggal_registrasi"], rng, 0.30)
    for i in rng.choice(len(pel), size=12, replace=False):
        nilai = str(pel.iat[i, pel.columns.get_loc("nama_pelanggan")])
        pel.iat[i, pel.columns.get_loc("nama_pelanggan")] = nilai.replace(" ", "  ")
    for i in rng.choice(len(pel), size=10, replace=False):
        kode = str(pel.iat[i, pel.columns.get_loc("id_wilayah")])
        nama_kota = WILAYAH[int(kode.split("-")[1]) - 1][0]
        pel.iat[i, pel.columns.get_loc("id_wilayah")] = str(rng.choice(KOTA_ALIAS[nama_kota]))
    for i in rng.choice(len(pel), size=8, replace=False):
        teks = str(pel.iat[i, pel.columns.get_loc("termin_pembayaran")])
        varian = {
            "COD": ["cod", "COD ", "Cash On Delivery"],
            "30 Hari": ["30", "NET 30", "30 hari"],
            "45 Hari": ["45", "NET 45", "45 hari"],
            "60 Hari": ["60", "NET 60", "60 hari"],
        }[teks]
        pel.iat[i, pel.columns.get_loc("termin_pembayaran")] = str(rng.choice(varian))
    for i, gaya in zip(
        rng.choice(len(pel), size=10, replace=False), rng.choice(3, size=10, p=[0.5, 0.3, 0.2])
    ):
        nilai = float(pel.iat[i, pel.columns.get_loc("limit_kredit")])
        pel.iat[i, pel.columns.get_loc("limit_kredit")] = teks_uang(nilai, int(gaya))
    for i in rng.choice(len(pel), size=4, replace=False):
        pel.iat[i, pel.columns.get_loc("kanal_akuisisi")] = ""
    for i in rng.choice(len(pel), size=6, replace=False):
        teks = str(pel.iat[i, pel.columns.get_loc("segmen")])
        pel.iat[i, pel.columns.get_loc("segmen")] = str(
            rng.choice([teks.lower(), teks.upper(), " " + teks])
        )
    pel = pd.concat([pel, pel.sample(2, random_state=int(rng.integers(0, 10**6)))], ignore_index=True)

    prod = produk.astype(object).copy()
    for i, gaya in zip(
        rng.choice(len(prod), size=10, replace=False), rng.choice(3, size=10, p=[0.5, 0.3, 0.2])
    ):
        nilai = float(prod.iat[i, prod.columns.get_loc("harga_modal")])
        prod.iat[i, prod.columns.get_loc("harga_modal")] = teks_uang(nilai, int(gaya))
    for i in rng.choice(len(prod), size=4, replace=False):
        prod.iat[i, prod.columns.get_loc("berat_kg")] = ""
    for i in rng.choice(len(prod), size=12, replace=False):
        teks = str(prod.iat[i, prod.columns.get_loc("status_aktif")])
        prod.iat[i, prod.columns.get_loc("status_aktif")] = str(
            rng.choice([teks.lower(), teks.upper(), " " + teks])
        )
    for i in rng.choice(len(prod), size=8, replace=False):
        nilai = str(prod.iat[i, prod.columns.get_loc("nama_produk")])
        prod.iat[i, prod.columns.get_loc("nama_produk")] = nilai.replace(" ", "  ")
    prod = pd.concat([prod, prod.sample(2, random_state=int(rng.integers(0, 10**6)))], ignore_index=True)

    kat = kategori.astype(object).copy()
    for i in rng.choice(len(kat), size=5, replace=False):
        teks = str(kat.iat[i, kat.columns.get_loc("nama_kategori")])
        kat.iat[i, kat.columns.get_loc("nama_kategori")] = str(
            rng.choice([teks.upper(), teks.lower(), teks + " "])
        )
    for i in rng.choice(len(kat), size=3, replace=False):
        nilai = str(kat.iat[i, kat.columns.get_loc("deskripsi")])
        kat.iat[i, kat.columns.get_loc("deskripsi")] = f"  {nilai} "

    wil = wilayah.astype(object).copy()
    for i in rng.choice(len(wil), size=9, replace=False):
        provinsi = str(wil.iat[i, wil.columns.get_loc("provinsi")])
        wil.iat[i, wil.columns.get_loc("provinsi")] = str(rng.choice(PROVINSI_ALIAS[provinsi]))
    for i in rng.choice(len(wil), size=5, replace=False):
        kota = str(wil.iat[i, wil.columns.get_loc("kota")])
        wil.iat[i, wil.columns.get_loc("kota")] = f" {kota} "
    wil = pd.concat([wil, wil.sample(1, random_state=int(rng.integers(0, 10**6)))], ignore_index=True)

    sal = sales.astype(object).copy()
    for i in rng.choice(len(sal), size=5, replace=False):
        nilai = str(sal.iat[i, sal.columns.get_loc("nama_sales")])
        sal.iat[i, sal.columns.get_loc("nama_sales")] = nilai.replace(" ", "  ")
    posisi_aktif = [i for i in range(len(sal)) if pd.isna(sal.iat[i, sal.columns.get_loc("tanggal_keluar")])]
    for i in posisi_aktif[:8]:
        sal.iat[i, sal.columns.get_loc("tanggal_keluar")] = "0000-00-00"
    for i in rng.choice(len(sal), size=6, replace=False):
        teks = str(sal.iat[i, sal.columns.get_loc("status")])
        sal.iat[i, sal.columns.get_loc("status")] = str(
            rng.choice([teks.lower(), teks.upper(), " " + teks])
        )
    return pel, prod, kat, wil, sal


def cetak_laporan(
    transaksi: pd.DataFrame,
    pelanggan: pd.DataFrame,
    wilayah: pd.DataFrame,
    siklus: dict,
    catatan_churn: list,
) -> None:
    kanonik = transaksi.copy()
    kanonik["tanggal"] = pd.to_datetime(kanonik["tanggal_transaksi"])
    kanonik["tahun"] = kanonik["tanggal"].dt.year
    kanonik["total"] = (
        kanonik["jumlah"]
        * kanonik["harga_satuan"]
        * (1 - kanonik["diskon_persen"] / 100.0)
    )
    waktu_kirim = (pd.to_datetime(kanonik["tanggal_kirim"]) - kanonik["tanggal"]).dt.days
    sla = (
        kanonik["id_pelanggan"]
        .map(pelanggan.set_index("id_pelanggan")["id_wilayah"])
        .map(wilayah.set_index("id_wilayah")["sla_kirim_hari"])
    )
    kanonik["keterlambatan"] = (waktu_kirim - sla).clip(lower=0)
    ringkas = kanonik.groupby("tahun").agg(
        baris=("id_transaksi", "size"),
        pelanggan=("id_pelanggan", "nunique"),
        penjualan=("total", "sum"),
        rata_diskon=("diskon_persen", "mean"),
        persen_telat=("keterlambatan", lambda x: (x > 3).mean() * 100),
        persen_retur=("status_pesanan", lambda x: (x == "Retur").mean() * 100),
    )
    ringkas["penjualan_miliar"] = (ringkas["penjualan"] / 1e9).round(2)
    ringkas["baris_per_pelanggan"] = (ringkas["baris"] / ringkas["pelanggan"]).round(2)
    ringkas["rata_diskon"] = ringkas["rata_diskon"].round(1)
    ringkas["persen_telat"] = ringkas["persen_telat"].round(1)
    ringkas["persen_retur"] = ringkas["persen_retur"].round(1)

    aktif_per_bulan = []
    for idx in range(N_BULAN):
        aktif_per_bulan.append(
            sum(
                1
                for s in siklus.values()
                if s["mulai"] <= idx and (s["churn"] is None or idx < s["churn"])
            )
        )
    ringkas["pelanggan_aktif_akhir_tahun"] = [
        aktif_per_bulan[
            max(i for i in range(N_BULAN) if bulan_ke_tanggal(i).year == tahun)
        ]
        for tahun in ringkas.index
    ]

    print("\n=== RINGKASAN TAHUNAN ===")
    print(
        ringkas[
            [
                "baris",
                "pelanggan",
                "pelanggan_aktif_akhir_tahun",
                "baris_per_pelanggan",
                "penjualan_miliar",
                "rata_diskon",
                "persen_telat",
                "persen_retur",
            ]
        ].to_string()
    )
    print("\n=== CHURN ===")
    print(json.dumps({str(t): sum(1 for c in catatan_churn if c["tahun"] == t) for t in range(2019, 2027)}, indent=2))
    print(f"total event churn: {len(catatan_churn)}")
    per_jenis = {}
    for c in catatan_churn:
        if c["tahun"] >= 2024:
            per_jenis.setdefault(c["jenis"], 0)
            per_jenis[c["jenis"]] += 1
    print("event churn 2024-2026 per jenis:", per_jenis)


def jalankan(konfig: Konfigurasi) -> dict:
    rng = np.random.default_rng(konfig.seed)
    wilayah = bangun_wilayah()
    kategori = bangun_kategori()
    produk = bangun_produk(rng)
    sales = bangun_tenaga_penjualan(rng)
    pelanggan, profil = bangun_pelanggan(rng, wilayah, sales)
    siklus, catatan_churn, pelanggan = simulasi_siklus(pelanggan, profil, sales, rng)
    transaksi = bangun_transaksi(rng, pelanggan, produk, siklus, profil)

    if len(transaksi) != konfig.baris:
        raise RuntimeError(f"baris kanonik {len(transaksi)} tidak sama dengan {konfig.baris}")
    if not transaksi["id_pelanggan"].isin(pelanggan["id_pelanggan"]).all():
        raise RuntimeError("integritas pelanggan kanonik gagal")
    if not transaksi["id_produk"].isin(produk["id_produk"]).all():
        raise RuntimeError("integritas produk kanonik gagal")
    if transaksi["tanggal_transaksi"].min() < MULAI or transaksi["tanggal_transaksi"].max() > SELESAI:
        raise RuntimeError("tanggal kanonik di luar rentang")

    transaksi_kotor = noda_nilai(
        transaksi,
        rng,
        kolom_tanggal=["tanggal_transaksi", "tanggal_kirim"],
        kolom_uang=["harga_satuan", "ongkos_kirim"],
    )
    sampah = baris_sampah(transaksi_kotor, rng)
    transaksi_raw = pd.concat([transaksi_kotor, sampah], ignore_index=True)
    transaksi_raw = transaksi_raw.sample(
        frac=1.0, random_state=int(rng.integers(0, 10**6))
    ).reset_index(drop=True)

    pel, prod, kat, wil, sal = noda_dimensi(
        pelanggan, produk, kategori, wilayah, sales, rng
    )

    konfig.keluaran.mkdir(parents=True, exist_ok=True)
    tabel = {
        "pelanggan": pel,
        "produk": prod,
        "kategori": kat,
        "wilayah": wil,
        "tenaga_penjualan": sal,
        "transaksi": transaksi_raw,
    }
    for nama, tabel_df in tabel.items():
        tabel_df.to_csv(
            konfig.keluaran / f"{nama}.csv", index=False, encoding="utf-8", na_rep=""
        )

    cetak_laporan(transaksi, pelanggan, wilayah, siklus, catatan_churn)
    print("\n=== FILE MENTAH ===")
    for nama, tabel_df in tabel.items():
        print(f"{nama}.csv: {len(tabel_df)} baris")
    print(f"baris sampah ditambahkan: {len(sampah)}")
    print(f"target baris bersih transaksi: {len(transaksi)}")

    return {
        "seed": konfig.seed,
        "baris_kanonik": int(len(transaksi)),
        "baris_raw": int(len(transaksi_raw)),
        "baris_sampah": int(len(sampah)),
        "churn_event": int(len(catatan_churn)),
        "file": {nama: int(len(tabel_df)) for nama, tabel_df in tabel.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator data sintetis PT Graha Material Nusantara")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw")
    parser.add_argument("--rows", type=int, default=15000)
    parser.add_argument("--dirt", type=str, default="medium", choices=["medium"])
    args = parser.parse_args()
    jalankan(Konfigurasi(seed=args.seed, keluaran=Path(args.out), baris=args.rows, dirt=args.dirt))


if __name__ == "__main__":
    main()
