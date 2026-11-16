# Data Sintetis PT Graha Material Nusantara

## Struktur

```
generator.py                              pembuat data mentah
notebooks/01_cleaning_preprocessing.ipynb  EDA, cleaning, dan export
data/raw/*.csv                            6 berkas mentah
data/cleaned/*.csv                        6 berkas bersih, nama dan skema sama
metadata.txt                              ringkasan dataset
```

## Menjalankan

```
pip install -r requirements.txt
python generator.py --seed 42 --out data/raw
jupyter notebook notebooks/01_cleaning_preprocessing.ipynb
```

Generator deterministik: seed sama menghasilkan data identik. Notebook memuat EDA pada data mentah (univariat dan multivariat, numerik dan kategorikal), cleaning, lalu menulis ulang `data/cleaned` dengan nama dan skema yang sama seperti `data/raw`.

## Kamus Data

### pelanggan.csv (100 baris, 10 kolom)
| Kolom | Isi |
|---|---|
| id_pelanggan | kunci, format CUST-0001 |
| nama_pelanggan | nama toko, CV, atau PT |
| jenis_pelanggan | Toko Bangunan, Kontraktor, Developer, Korporat |
| segmen | Platinum, Gold, Silver, Bronze |
| id_wilayah | kunci ke wilayah |
| id_sales | sales penanggung jawab saat ini |
| tanggal_registrasi | tanggal bergabung |
| kanal_akuisisi | Kunjungan Sales, Referral, Pameran, Telemarketing, Website, Tidak Diketahui |
| termin_pembayaran | COD, 30 Hari, 45 Hari, 60 Hari |
| limit_kredit | rupiah |

### produk.csv (80 baris)
id_produk, nama_produk, id_kategori, satuan, harga_modal, harga_jual_acuan (harga 2019), berat_kg, status_aktif.

### kategori.csv (10 baris)
Dua tingkat: lima kategori induk dan lima subkategori melalui `id_kategori_induk`.

### wilayah.csv (24 baris)
id_wilayah, provinsi, kota, region (Jawa, Sumatera, Kalimantan, Sulawesi), zona_kirim, sla_kirim_hari.

### tenaga_penjualan.csv (15 baris)
id_sales, nama_sales, cabang, region, tanggal_masuk, tanggal_keluar, target_bulanan, status. Lima sales resign antara Maret 2025 dan Mei 2026.

### transaksi.csv (15.000 baris bersih, 15 kolom)
| Kolom | Isi |
|---|---|
| id_transaksi | kunci baris |
| nomor_pesanan | pengelompokan item dalam satu pesanan |
| tanggal_transaksi | 2019-01-01 s.d. 2026-10-31 |
| id_pelanggan, id_produk, id_sales | kunci relasi |
| jumlah | kuantitas |
| harga_satuan | harga per satuan saat transaksi |
| diskon_persen | 0 sampai 30 |
| ongkos_kirim | rupiah, 0 untuk pesanan besar di Jawa |
| metode_pembayaran | Transfer Bank, Tunai, Virtual Account, Cek Giro, QRIS, Tidak Diketahui |
| termin_pembayaran | COD, 30 Hari, 45 Hari, 60 Hari (salinan saat pesanan dibuat) |
| status_pesanan | Selesai, Retur, Dibatalkan |
| tanggal_kirim | tanggal barang dikirim |
| status_pembayaran | Lunas, Sebagian, Belum Lunas, Tidak Berlaku |

## Kolom Turunan

Kolom berikut sengaja tidak ada di data mentah maupun bersih.

| Metrik | Rumus |
|---|---|
| omzet baris | `jumlah × harga_satuan × (1 − diskon_persen / 100)` |
| lead time | `tanggal_kirim − tanggal_transaksi` |
| keterlambatan | `max(0, lead time − wilayah.sla_kirim_hari)` |
| status churn | tidak ada transaksi Selesai atau Retur selama 180 hari |
| omzet bulanan | jumlah omzet baris berstatus Selesai per bulan |

## Aturan Analitik

- Omzet dan volume memakai `status_pesanan = Selesai`. Retur dihitung sebagai aktivitas pelanggan, Dibatalkan tidak.
- Churn: tidak ada transaksi Selesai atau Retur selama 180 hari setelah bulan pengamatan, dinilai hanya untuk pelanggan yang terdaftar minimal 90 hari dan punya minimal dua pesanan.
- Churn setara untuk dashboard: tidak ada pesanan dalam 180 hari terakhir.

## Pola yang Bisa Dijadikan Insight

| Gejala di dashboard | Akar masalah |
|---|---|
| Omzet 2026 turun 21% dari puncak 2024, volume turun lebih dalam | Basis pelanggan aktif menyusut 87 ke 67, churn naik sejak 2024 |
| Penurunan terkonsentrasi di Sumatera dan Jawa Barat | Churn wilayah itu 50-67% vs DKI Jakarta 7% |
| Segmen Toko Bangunan jatuh, Kontraktor stabil | Churn Toko Bangunan 49% vs Kontraktor 4% |
| Margin tertekan walau omzet 2024 masih tumbuh | Harga naik 3-7% per tahun menutupi volume yang turun |
| Pelanggan berhenti setelah diskon ditarik | Diskon 6 bulan terakhir churn 11,4% vs 8,3% pelanggan bertahan |
| Keluhan layanan naik | Keterlambatan > 3 hari 22,4% pada pelanggan churn vs 13,5% |
| Retur meningkat | Retur 7,9% pada pelanggan churn vs 5,4% |
