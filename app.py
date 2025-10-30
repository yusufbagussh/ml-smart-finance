import pandas as pd
from flask import Flask, request, jsonify
import joblib # <-- TAMBAHKAN IMPORT INI
import statsmodels.api as sm # (Ini untuk fitur /predict)
from io import StringIO # (Ini untuk fitur /predict)
import numpy as np
# HAPUS IMPORT: TfidfVectorizer, MultinomialNB, make_pipeline

# # --- Persiapan Model (Hanya dijalankan sekali saat server start) ---

# # 1. Baca Data Latih dari CSV (File baru Anda)
# try:
#     # Menggunakan file CSV baru Anda
#     df = pd.read_csv('src/data/riwayat_transaksi_indonesia.csv')
# except FileNotFoundError:
#     print("ERROR: File 'riwayat_transaksi_indonesia.csv' tidak ditemukan.")
#     exit("Pastikan 'riwayat_transaksi_indonesia.csv' ada di folder yang sama dengan 'app.py'")

# # 2. Cek apakah kolom yang dibutuhkan ada (sesuai hasil inspeksi)
# if 'description' not in df.columns or 'category' not in df.columns:
#     exit("ERROR: CSV harus memiliki kolom 'description' dan 'category'")

# # 3. Tentukan Fitur (X) dan Label (y)
# X_train = df['description']
# y_train = df['category']

# # 4. Buat & Latih Model Pipeline
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# model.fit(X_train, y_train)

# print("--- Model ML Siap (Dilatih dari riwayat_transaksi_indonesia.csv) ---")
# # -----------------------------------------------------------------

# # 5. Inisialisasi Flask App
# app = Flask(__name__)

# GANTI SEMUA BLOK DI ATAS DENGAN INI:
try:
    # 1. Load Model Kategori
    model_kategori = joblib.load('model_klasifikasi_kategori.joblib') # Nama file baru
    print("--- Model Klasifikasi Kategori Siap (Di-load dari file) ---")
    
    # 2. Load Model Tipe
    model_tipe = joblib.load('model_klasifikasi_tipe.joblib') # Nama file baru
    print("--- Model Klasifikasi Tipe Siap (Di-load dari file) ---")

except FileNotFoundError as e:
    print("="*50)
    print(f"ERROR: File model tidak ditemukan ({e.filename}).")
    print("Silakan jalankan 'python train_model.py' terlebih dahulu untuk membuat file model.")
    print("="*50)
    exit()
# -----------------------------------------------------------------

# 5. Inisialisasi Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/health')
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/classify', methods=['POST'])
def classify_transaction():
    try:
        data = request.get_json()
        description = data.get('description', '')
        if not description:
             return jsonify({'error': 'Missing description'}), 400


        # --- Gunakan Model Kategori ---
        predicted_category = model_kategori.predict([description])[0]
        probabilities_category = model_kategori.predict_proba([description])
        confidence_category = probabilities_category.max()

        # --- Gunakan Model Tipe ---
        predicted_type = model_tipe.predict([description])[0]
        probabilities_type = model_tipe.predict_proba([description])
        confidence_type = probabilities_type.max()

        # --- Kembalikan SEMUA hasil ---
        return jsonify({
            'predicted_category': predicted_category,
            'confidence_category': float(confidence_category),
            'predicted_type': predicted_type, # <-- Hasil Baru
            'confidence_type': float(confidence_type), # <-- Hasil Baru
            'explanation': f"Prediksi: '{predicted_category}' ({predicted_type}) - Keyakinan: Kat {confidence_category*100:.1f}%, Tipe {confidence_type*100:.1f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

@app.route('/predict', methods=['POST'])
def predict_spending():
    try:
        # 1. Terima data riwayat transaksi (sudah diagregasi) dari Laravel
        data = request.get_json()
        
        df = pd.DataFrame(data)
        if df.empty:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # --- PERBAIKAN DI SINI ---
        # Paksa kolom 'amount' menjadi tipe data angka (float)
        # Ini akan mengubah "150000.50" (string) menjadi 150000.50 (angka)
        df['amount'] = pd.to_numeric(df['amount'])
        # --- AKHIR PERBAIKAN ---

        # 2. Proses Data untuk Time Series (Lanjutan)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # --- PERUBAHAN PENTING ---
        # Data kita SUDAH dijumlahkan per hari. TAPI, mungkin ada hari
        # di mana user tidak belanja (datanya bolong/missing).
        # Model time series butuh data kontinu.
        
        # Jadi, kita resample per hari ('D'), dan HANYA mengisi
        # hari yang kosong dengan 0. Kita TIDAK perlu .sum() lagi.
        daily_spending = df['amount'].resample('D').sum()
        # --- AKHIR PERUBAHAN ---
        # console daily spending

        # 3. Latih Model Time Series (SARIMA) - (Ini tetap sama)
        model = sm.tsa.SARIMAX(daily_spending,
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 7))
        
        model_fit = model.fit(disp=False)
        
        # 4. Buat Prediksi untuk 30 hari ke depan - (Ini tetap sama)
        forecast = model_fit.forecast(steps=30)
        
        # 5. Format data untuk dikirim kembali ke Laravel - (Ini tetap sama)
        next_month_total = forecast.sum()
        
        forecast_data = []
        for date, value in forecast.items():
            # --- PERBAIKAN DI SINI ---
            # Mengganti "2025-m-d" dengan kode format tanggal yang benar
            forecast_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'amount': max(0, value) 
            })
            # --- AKHIR PERBAIKAN ---

        return jsonify({
            'next_month_total': f"Rp {next_month_total:,.0f}",
            'forecast_data': forecast_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def get_insights():
    try:
        data = request.get_json()
        
        if 'transactions' not in data or not data['transactions']:
             return jsonify({'insights': [{'type': 'info', 'message': 'Belum ada transaksi bulan ini.'}]})

        df_trans = pd.DataFrame(data['transactions'])
        df_budgets = pd.DataFrame(data['budgets'])
        
        # --- PERBAIKAN UTAMA DI SINI ---
        # 1. Konversi 'amount' dari data transaksi
        df_trans['amount'] = pd.to_numeric(df_trans['amount'])
        
        # 2. (BARU) Konversi 'spent' dan 'budget' dari data anggaran
        df_budgets['spent'] = pd.to_numeric(df_budgets['spent'])
        df_budgets['budget'] = pd.to_numeric(df_budgets['budget'])
        # --- AKHIR PERBAIKAN ---

        insights = []

        # --- FITUR 1: DETEKSI ANOMALI ---
        df_expense = df_trans[df_trans['type'] == 'expense'].copy()
        
        # Cek jika ada data expense
        if not df_expense.empty:
            category_stats = df_expense.groupby('category')['amount'].agg(['mean', 'std']).fillna(0)
            
            for category, stats in category_stats.iterrows():
                mean = stats['mean']
                std = stats['std']
                threshold = max(mean + 2.5 * std, 100000) 

                anomalies = df_expense[
                    (df_expense['category'] == category) &
                    (df_expense['amount'] > threshold)
                ]
                
                for _, row in anomalies.iterrows():
                    insight = {
                        'type': 'anomaly',
                        'message': f"Transaksi '{row['description']}' (Rp {row['amount']:,.0f}) terdeteksi tidak biasa untuk kategori '{category}'. Rata-rata Anda di kategori ini hanya Rp {mean:,.0f}."
                    }
                    insights.append(insight)

        # --- FITUR 2: WAWASAN KATEGORI & BUDGET ---
        # Cek jika ada data budget
        if not df_budgets.empty:
            for _, budget in df_budgets.iterrows():
                category = budget['category']
                spent = budget['spent']
                budget_amount = budget['budget']

                if budget_amount > 0 and spent > budget_amount:
                    over_amount = spent - budget_amount
                    insight = {
                        'type': 'budget_warning',
                        'message': f"Anda MELEBIHI anggaran '{category}' sebesar Rp {over_amount:,.0f}!"
                    }
                    insights.append(insight)

                    # --- Deep Dive ---
                    if not df_expense.empty:
                        descriptions = df_expense[df_expense['category'] == category]['description'].str.lower()
                        
                        if descriptions.empty:
                            continue

                        if category == 'Food & Dining':
                            kopi_count = descriptions.str.contains('kopi|starbucks|kenangan|jiwa').sum()
                            gofood_count = descriptions.str.contains('gofood|grabfood').sum()
                            
                            if kopi_count > 5:
                                insights.append({'type': 'category_insight', 'message': f"💡 Tips: Anda memiliki {kopi_count} transaksi 'Kopi' bulan ini. Mengurangi jajan kopi bisa sangat membantu anggaran 'Food & Dining'."})
                            if gofood_count > 5:
                                insights.append({'type': 'category_insight', 'message': f"💡 Tips: Anda memesan {gofood_count} kali via 'Gofood/Grabfood'. Memasak di rumah bisa menghemat banyak."})

                        if category == 'Shopping':
                            shopee_count = descriptions.str.contains('shopee').sum()
                            tokopedia_count = descriptions.str.contains('tokopedia').sum()
                            if (shopee_count + tokopedia_count) > 5:
                                insights.append({'type': 'category_insight', 'message': f"💡 Tips: Anda memiliki {shopee_count + tokopedia_count} transaksi di e-commerce bulan ini. Cek kembali keranjang Anda untuk barang yang tidak perlu."})

        if not insights:
            insights.append({'type': 'info', 'message': 'Data keuangan Anda bulan ini terlihat sehat. Pertahankan!'})

        return jsonify({'insights': insights})

    except Exception as e:
        # Ini adalah error yang dikirim ke Laravel (Status 500)
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

if __name__ == '__main__':
    # Jalankan server API di port 5000
    print("=== Memulai Server Flask di http://localhost:5000 ===")
    app.run(port=5000, debug=True)
    
