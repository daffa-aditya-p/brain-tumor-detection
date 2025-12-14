# Deteksi Tumor Otak dengan TensorFlow/Keras di Termux

Proyek ini menyediakan pipeline machine learning untuk melatih sebuah Convolutional Neural Network (CNN) untuk deteksi tumor otak. Versi ini telah ditulis ulang menggunakan **TensorFlow/Keras**, sebuah framework yang lebih umum dan stabil.

## Fitur

- **Framework Standar:** Dibangun dengan TensorFlow/Keras untuk kompatibilitas yang lebih baik.
- **Pipeline Efisien:** Menggunakan `tf.data` untuk memuat dan memproses gambar secara efisien.
- **Augmentasi Data:** Menerapkan augmentasi ringan (flip, rotasi) pada data training untuk meningkatkan generalisasi model.
- **Training Modern:** Menggunakan `model.fit()` dengan callbacks `EarlyStopping` (menghentikan training saat tidak ada peningkatan) dan `ModelCheckpoint` (menyimpan hanya model terbaik).
- **Logging Metrik:** Menyimpan riwayat training ke file CSV dan menghasilkan plot kurva loss dan akurasi.

## Prinsip Proyek

Proyek ini mengikuti prinsip-prinsip kunci MLOps untuk training dan evaluasi model yang efektif, seperti yang dicatat oleh bang farel Kurniawan:

- **Pemantauan Loss Komparatif:**
  > “[2/11 08.09] bang farel Kurniawan: Hrs ada 2 perbandingan, loss training sm loss validation”
- **Pemisahan Data yang Benar:**
  > “[2/11 08.13] bang farel Kurniawan: Tp penggunaan datasetnya beda, jd di preprocessing dilakuin dlu data split (biasanya 80/20)”
- **Early Stopping untuk Mencegah Overfitting:**
  > “[2/11 08.19] bang farel Kurniawan: Biasanya pake teknik early stopping, jd kita cari epoch yg optimal. Nnti klo semisal validation lossnya mulai naik, berarti itu tanda2 overfitting”

## Cara Penggunaan

1.  **Siapkan Dataset:**
    Jika belum, jalankan skrip `run.sh` untuk mengunduh dan mengekstrak dataset.
    ```bash
    bash run.sh
    ```

2.  **Latih Model:**
    Jalankan skrip training baru. Skrip ini akan melatih model dan menyimpan file `best_model.keras`.
    ```bash
    python training.py
    ```

3.  **Lakukan Prediksi:**
    Setelah model (`best_model.keras`) tersimpan, gunakan `main.py` untuk melakukan prediksi pada gambar baru.
    ```bash
    # Contoh prediksi
    python main.py dataset/pred/pred10.jpg
    ```

## Kebutuhan Sistem

Pastikan library berikut sudah terinstal di lingkungan Termux Anda:
- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `pillow`
