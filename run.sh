#!/bin/bash
# Skrip untuk mengunduh dan menyiapkan dataset deteksi tumor otak.

echo "Memeriksa apakah 'unzip' terinstal..."
if ! command -v unzip &> /dev/null
then
    echo "'unzip' tidak ditemukan. Silakan install dengan menjalankan 'pkg install unzip'"
    exit 1
fi

# Membuat direktori jika belum ada
mkdir -p ~/Downloads
mkdir -p dataset

ZIP_FILE=~/Downloads/brain-tumor-detection.zip
DATASET_DIR=dataset

echo "Mulai mengunduh dataset... (Ini mungkin memerlukan waktu)"
# Unduh dataset menggunakan curl
curl -L -o $ZIP_FILE https://www.kaggle.com/api/v1/datasets/download/ahmedhamada0/brain-tumor-detection

# Periksa apakah unduhan berhasil
if [ ! -f "$ZIP_FILE" ] || [ $(stat -c%s "$ZIP_FILE") -lt 100000 ]; then
    echo "Gagal mengunduh dataset. Ukuran file tidak valid."
    echo "Ini mungkin karena Anda tidak login ke Kaggle."
    echo "Silakan coba unduh manual dari https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection dan letakkan di folder 'dataset'."
    exit 1
fi

echo "Unduhan selesai. Mengekstrak file ke folder '$DATASET_DIR'..."
# Ekstrak dataset ke direktori yang ditentukan
unzip -q $ZIP_FILE -d $DATASET_DIR

# Hapus file zip setelah diekstrak untuk menghemat ruang
rm $ZIP_FILE

echo "Dataset telah berhasil disiapkan di folder '$DATASET_DIR'."
echo "Struktur direktori internal mungkin bervariasi. Skrip training akan mencoba menemukannya secara otomatis."