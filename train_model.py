import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import nltk # <- Import NLTK
from nltk.corpus import stopwords # <- Import stopwords

print("Mulai proses training...")

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv('src/data/riwayat_transaksi_indonesia_lengkap.csv')
    print(f"Data awal dimuat: {len(df)} baris")
except FileNotFoundError:
    print("ERROR: File 'riwayat_transaksi_indonesia.csv' tidak ditemukan.")
    exit()
    
if 'description' not in df.columns or 'category' not in df.columns:
    exit("ERROR: CSV harus memiliki kolom 'description' dan 'category'")

# --- 2. DATA ASSESSMENT & CLEANING ---

# A. Cek & Hapus Missing Values
if df[['description', 'category']].isnull().any().any():
    print("Menghapus baris dengan nilai (description/category) yang kosong...")
    df.dropna(subset=['description', 'category'], inplace=True)
    print(f"Data setelah hapus null: {len(df)} baris")

# B. Cek & Hapus Duplikat
duplicates_count = df.duplicated().sum()
if duplicates_count > 0:
    print(f"Menghapus {duplicates_count} baris duplikat...")
    df.drop_duplicates(inplace=True)
    print(f"Data setelah hapus duplikat: {len(df)} baris")

# C. (Opsional) Cek Distribusi Kategori (Assessment)
# Ini membantu Anda melihat apakah data Anda 'seimbang' atau tidak
print("\nDistribusi Kategori:")
print(df['category'].value_counts())
print("-" * 30)

# --- 3. FEATURE ENGINEERING & TRAINING ---

# Tentukan Fitur (X) dan Label (y) dari data yang sudah bersih
# X_train = df['description']
# y_train = df['category']

# Siapkan daftar Stopwords Bahasa Indonesia
try:
    indonesian_stopwords = list(stopwords.words('indonesian'))
    # Tambahkan kata-kata umum terkait keuangan yang mungkin tidak membantu
    indonesian_stopwords.extend(['di', 'ke', 'dari', 'atau', 'dan', 'untuk', 'dengan', 'pada', 'baiknya', 'berkali', 'kali', 'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya'])
    print("Berhasil memuat stopwords Bahasa Indonesia.")
except:
    print("Gagal memuat stopwords, akan lanjut tanpa stopwords.")
    indonesian_stopwords = None

# --- 4. MODEL 1: KLASIFIKASI KATEGORI ---
print("\n--- Melatih Model Kategori ---")
X_train_desc = df['description']
y_train_category = df['category']

model_category = make_pipeline(
    TfidfVectorizer(stop_words=indonesian_stopwords),
    MultinomialNB()
)
model_category.fit(X_train_desc, y_train_category)
joblib.dump(model_category, 'model_klasifikasi_kategori.joblib') # Nama file baru
print("Model Kategori Berhasil Dilatih dan Disimpan ke 'model_klasifikasi_kategori.joblib'")

# --- 5. MODEL 2: KLASIFIKASI TIPE (Income/Expense) ---
print("\n--- Melatih Model Tipe ---")
y_train_type = df['type'] # Targetnya adalah kolom 'type'

# Kita bisa gunakan pipeline yang sama
model_type = make_pipeline(
    TfidfVectorizer(stop_words=indonesian_stopwords),
    MultinomialNB()
)
# Latih dengan deskripsi yang sama, tapi target 'type'
model_type.fit(X_train_desc, y_train_type)
joblib.dump(model_type, 'model_klasifikasi_tipe.joblib') # Nama file baru
print("Model Tipe Berhasil Dilatih dan Disimpan ke 'model_klasifikasi_tipe.joblib'")

print("\n--- Semua Proses Training Selesai ---")