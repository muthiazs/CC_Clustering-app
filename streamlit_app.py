import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random

# Judul dan Deskripsi Proyek
st.title("CC Clustering 💳")
st.write(
    "Tirza Aurellia Wijaya (24060122130047) /" 
    "Muthia Zhafira Sahnah (24060122130071) /"
    "Nadiva Aulia Inaya (24060122130093) /"
    "Alya Safina (24060122140123)"
)
st.write(
    "Tujuan dari proyek ini adalah untuk melakukan segmentasi terhadap pengguna kartu kredit berdasarkan perilaku belanja, pola pembayaran, dan penggunaan kredit mereka. Dengan menggunakan teknik clustering, bank atau lembaga keuangan dapat memahami lebih baik pelanggan mereka, menargetkan mereka dengan promosi yang lebih relevan, serta mengidentifikasi risiko kredit yang lebih baik."
)

# Membaca Dataset dari Folder
DATA_PATH = "data/CC GENERAL.csv"
try:
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    st.success("Dataset berhasil dimuat dari folder!")

    # Menampilkan Preview Dataset
    st.write("📊 **Preview Dataset**")
    st.dataframe(df.head())

    # Deskripsi Fitur Dataset
    st.header("📋 Deskripsi Fitur dalam Dataset")
    feature_description = pd.DataFrame({
        "Fitur": [
            "CUST_ID", "BALANCE", "BALANCE_FREQUENCY", "PURCHASES",
            "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
            "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
            "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY",
            "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT",
            "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
        ],
        "Tipe Data": [
            "Kategori", "Numerik", "Numerik", "Numerik", "Numerik", "Numerik",
            "Numerik", "Numerik", "Numerik", "Numerik", "Numerik", "Numerik",
            "Numerik", "Numerik", "Numerik", "Numerik", "Numerik", "Numerik"
        ],
        "Deskripsi": [
            "ID unik untuk mengidentifikasi setiap pelanggan",
            "Total saldo yang tersisa di akun kartu kredit pelanggan",
            "Frekuensi rata-rata pelanggan memperbarui saldo mereka, berkisar antara 0-1",
            "Total nilai pembelian yang dilakukan oleh pelanggan",
            "Total pembelian satu kali yang dilakukan pelanggan",
            "Total pembelian yang dilakukan secara angsuran",
            "Total uang tunai yang ditarik oleh pelanggan menggunakan kartu kredit",
            "Frekuensi rata-rata pelanggan melakukan pembelian, berkisar antara 0-1",
            "Frekuensi rata-rata pelanggan melakukan pembelian satu kali, berkisar 0-1",
            "Frekuensi rata-rata pelanggan melakukan pembelian angsuran, berkisar 0-1",
            "Frekuensi rata-rata pelanggan menarik uang tunai dengan kartu kredit, 0-1",
            "Jumlah transaksi uang tunai yang dilakukan oleh pelanggan",
            "Jumlah total transaksi pembelian yang dilakukan oleh pelanggan",
            "Batas kredit maksimum yang diberikan kepada pelanggan",
            "Total jumlah pembayaran yang dilakukan oleh pelanggan",
            "Jumlah pembayaran minimum yang diperlukan untuk pelanggan",
            "Persentase pembayaran penuh yang dilakukan oleh pelanggan, berkisar antara 0-1",
            "Jumlah bulan pelanggan telah menggunakan kartu kredit"
        ]
    })
    st.table(feature_description)

     # Plot histogram untuk semua kolom numerik
    st.header("📊 Histogram Distribusi Data")
    st.write(
        "Histogram di bawah menunjukkan distribusi data untuk setiap kolom numerik. "
        "Interpretasi histogram disertakan di bawah setiap plot."
    )
    
    # Loop untuk membuat histogram untuk setiap kolom numerik
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].hist(bins=30, color="blue", alpha=0.7, ax=ax)
        ax.set_title(f"Distribusi {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)
        
        # Penjelasan distribusi
        if col == "BALANCE":
            st.write("💡 **Interpretasi**: Skew ke kanan (positive skew). Kebanyakan pengguna memiliki jumlah saldo yang sedikit di akun CC mereka.")
        elif col == "BALANCE_FREQUENCY":
            st.write("💡 **Interpretasi**: Skew ke kiri. Kebanyakan pengguna sering memperbarui saldo mereka (a.k.a. membayar tagihan cc).")
        elif col == "PURCHASES":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna jarang melakukan pembelian.")
        elif col == "ONEOFF_PURCHASES":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna jarang menggunakan cc untuk melakukan pembelian sekali beli.")
        elif col == "INSTALLMENTS_PURCHASES":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna jarang menggunakan cc untuk melakukan pembelian secara angsuran (kredit).")
        elif col == "CASH_ADVANCE":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna jarang melakukan tarik tunai menggunakan cc.")
        elif col == "PURCHASES_FREQUENCY":
            st.write("💡 **Interpretasi**: Distribusi bimodal. Ada dua perilaku utama: jarang banget dan sering banget.")
        elif col == "ONEOFF_PURCHASES_FREQUENCY":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna jarang melakukan pembelian satu kali, namun beberapa sering.")
        elif col == "PURCHASES_INSTALLMENTS_FREQUENCY":
            st.write("💡 **Interpretasi**: Distribusi relatif normal dibandingkan yang lain.")
        elif col == "CASH_ADVANCE_FREQUENCY":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna jarang melakukan tarik tunai.")
        elif col == "CASH_ADVANCE_TRX":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna jarang melakukan transaksi tarik tunai.")
        elif col == "PURCHASES_TRX":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna jarang melakukan transaksi pembelian.")
        elif col == "CREDIT_LIMIT":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna memiliki batas kredit yang cenderung rendah, sekitar 800-1000, namun beberapa memiliki limit tinggi.")
        elif col == "PAYMENTS":
            st.write("💡 **Interpretasi**: Skew ke kanan. Kebanyakan pengguna jarang menggunakan cc untuk pembayaran.")
        elif col == "MINIMUM_PAYMENTS":
            st.write("💡 **Interpretasi**: Beberapa pelanggan memiliki pembayaran minimum sangat tinggi, namun mayoritas kecil.")
        elif col == "PRC_FULL_PAYMENT":
            st.write("💡 **Interpretasi**: Sebagian besar pelanggan jarang atau tidak pernah membayar penuh saldo mereka.")
        elif col == "TENURE":
            st.write("💡 **Interpretasi**: Skew ke kiri. Kebanyakan pelanggan sudah menggunakan cc minimal 12 bulan.")

    # Analisis Awal Dataset
    st.header("🔍 Analisis Awal Dataset")
    rows, cols = df.shape
    st.write(f"Dataset memiliki **{rows} baris** dan **{cols} kolom**.")

    # Info Dataset dalam Bentuk Tabel
    st.subheader("📌 Info Dataset")
    info_table = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": [str(df[col].dtype) for col in df.columns],
        "Jumlah Non-Null": [df[col].notnull().sum() for col in df.columns],
        "Jumlah Null": [df[col].isnull().sum() for col in df.columns]
    })
    st.write(info_table)

    st.subheader("📾 Statistik Dataset")
    st.write(df.describe())

    st.subheader("❓ Missing Values")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        "Kolom": missing_values.index,
        "Jumlah Missing": missing_values.values,
        "Persentase Missing (%)": missing_percent.values
    })
    st.write(missing_df)

    # Penanganan Missing Values
    df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
    df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())

    # Pilih fitur numerik untuk clustering
    numeric_features = [
        'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 
        'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 
        'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT'
    ]

    # Fungsi capping berdasarkan IQR
    def apply_capping(df):
        capped_df = df.copy()  # Salin DataFrame untuk capping
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)  # Kuartil pertama
            Q3 = df[col].quantile(0.75)  # Kuartil ketiga
            IQR = Q3 - Q1  # Rentang antar kuartil
            
            lower_bound = Q1 - 1.5 * IQR  # Batas bawah
            upper_bound = Q3 + 1.5 * IQR  # Batas atas
            
            # Terapkan capping
            capped_df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            capped_df[col] = np.where(df[col] > upper_bound, upper_bound, capped_df[col])
        
        return capped_df

    # Terapkan capping pada dataset
    df = apply_capping(df)

    # Standarisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_features])  # Standarisasi hanya untuk fitur numerik
    df_scaled = pd.DataFrame(X_scaled, columns=numeric_features, index=df.index)

    # Copy untuk PCA3
    df_copy = df_scaled.copy()

    # PCA
    st.header("📊 Principal Component Analysis (PCA)")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    # Buat DataFrame komponen utama
    components_df = pd.DataFrame(pca.components_, columns=numeric_features, index=['PC1', 'PC2'])

    # Tampilkan kontribusi fitur untuk setiap komponen utama
    for i, row in components_df.iterrows():
        st.subheader(f"Komponen penyusun {i}:")
        contributing_features = row.index[row.abs() > 0.1]
        st.write(", ".join(contributing_features))

    # Elbow Method & Silhouette Score
    st.header("🔍 Cluster Optimalisasi")
        
    # Elbow Method
    st.subheader("Elbow Method")
    inertia = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, inertia, '-o', color='blue')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.grid(True)
    st.pyplot(fig)

    # Silhouette Score
    st.subheader("Silhouette Score")
    silhouette_scores = []
    k_values = range(2, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, silhouette_scores, '-o', color='green')
    ax.set_title('Silhouette Scores for Different k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.grid(True)
    st.pyplot(fig)

    # Pilih jumlah cluster optimal berdasarkan nilai Silhouette Score terbaik
    optimal_k =3

    st.header("📌 Clustering")
    st.write(f"Jumlah cluster optimal adalah {optimal_k} berdasarkan Silhouette Score.")

    # K-Means Clustering dengan jumlah cluster optimal
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Visualisasi hasil clustering
    st.header("Visualisasi Hasil Clustering")

    # Lakukan PCA untuk proyeksi 2D
    X_pca = pca.transform(X_scaled)

    # Buat plot menggunakan Seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set2', s=50, ax=ax)
    ax.set_title('K-Means Clustering Visualization')
    ax.set_xlabel('Purchase and Payment Behavior')
    ax.set_ylabel('CC Limit and Cash Usage Behavior')
    ax.legend(title='Cluster')
    ax.grid(True)

    # Tampilkan plot di Streamlit
    st.pyplot(fig)

    
    # Cluster Profiling dan definisi sifat setiap cluster
    st.header("📋 Profil Cluster")

    # Menampilkan profil rata-rata fitur per cluster
    cluster_profiles = df.groupby('Cluster')[numeric_features].mean()
    st.dataframe(cluster_profiles)

    # Menambahkan definisi sifat tiap cluster
    st.write("Definisi setiap Cluster:")

    # Deskripsi untuk setiap cluster berdasarkan profiling
    for cluster_num in range(optimal_k):
        st.write(f"Cluster {cluster_num}:")
        if cluster_num == 0:
            st.write("Pengguna dengan aktivitas transaksi rendah, namun penggunaan kartu kredit dan penarikan tunai tinggi. Kemungkinan besar menggunakan kartu kredit untuk menarik tunai dan membayar transaksi dengan uang tunai.")
        elif cluster_num == 1:
            st.write("Pengguna dengan aktivitas transaksi rendah dan pembayaran kartu kredit yang cenderung minimum atau sebagian kecil. Cenderung tidak memanfaatkan limit kartu kredit secara maksimal.")
        elif cluster_num == 2:
            st.write("Pengguna dengan aktivitas transaksi dan pembayaran yang tinggi, sering memanfaatkan cash advance dan menggunakan limit kartu kredit secara maksimal.")
        else:
            st.write("Deskripsi untuk cluster ini belum tersedia.")

    # Prediksi Cluster untuk Input Baru
    st.header("🔮 Prediksi Cluster Baru")
    st.write("Masukkan nilai fitur untuk prediksi cluster:")
        
    # Input fitur untuk prediksi
    input_features = {}
    for feature in numeric_features:
        input_features[feature] = st.number_input(
            f"{feature}", 
            value=float(df[feature].mean()), 
            step=0.01
        )
        
    # Tombol prediksi
    if st.button("Prediksi Cluster"):
        # Siapkan input untuk prediksi
        input_data = [input_features[feature] for feature in numeric_features]
            
        # Standarisasi input
        input_scaled = scaler.transform([input_data])
            
        # Prediksi cluster
        predicted_cluster = kmeans.predict(input_scaled)[0]
            
        st.success(f"Pelanggan diprediksi masuk ke Cluster {predicted_cluster}")
            
         # Tampilkan profil cluster terdekat
        st.write("Profil Cluster Terdekat:")
        st.dataframe(cluster_profiles.loc[predicted_cluster])
        
        # Keterangan sifat customer per cluster
        st.write(f"Deskripsi Sifat Customer untuk Cluster {predicted_cluster}:")
        
        if predicted_cluster == 0:
            st.write("Pengguna dengan aktivitas transaksi rendah, namun penggunaan kartu kredit dan penarikan tunai tinggi. Kemungkinan besar menggunakan kartu kredit untuk menarik tunai dan membayar transaksi dengan uang tunai.")
        elif predicted_cluster == 1:
            st.write("Pengguna dengan aktivitas transaksi rendah dan pembayaran kartu kredit yang cenderung minimum atau sebagian kecil. Cenderung tidak memanfaatkan limit kartu kredit secara maksimal.")
        elif predicted_cluster == 2:
            st.write("Pengguna dengan aktivitas transaksi dan pembayaran yang tinggi, sering memanfaatkan cash advance dan menggunakan limit kartu kredit secara maksimal.")
        else:
            st.write("Deskripsi cluster ini belum tersedia.")



except FileNotFoundError:
    st.error(f"File dataset tidak ditemukan di path: {DATA_PATH}. Pastikan file sudah ada di folder data.")
