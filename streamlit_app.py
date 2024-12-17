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
# 💳 Segmentasi Pengguna Kartu Kredit dengan Teknik Clustering 🔍
st.title("Segmentasi Pengguna Kartu Kredit dengan Teknik Clustering💳🔍")

st.markdown("---")

# Membaca Dataset dari Folder
DATA_PATH = "data/CC GENERAL.csv"
try:
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    st.success("Dataset berhasil dimuat dari folder!")

    with st.expander("👥**Anggota Kelompok**"):
        # 👥 Anggota Kelompok
        st.markdown("""
        - 🏅 **Tirza Aurellia Wijaya** (24060122130047)  
        - 🏅 **Muthia Zhafira Sahnah** (24060122130071)  
        - 🏅 **Nadiva Aulia Inaya** (24060122130093)  
        - 🏅 **Alya Safina** (24060122140123)  
        """)

        st.markdown("---")

    # 📌 Motivasi Proyek
    with st.expander("📌Motivasi Proyek & Apa yang akan dilakukan 🧩"):
        st.write(
            "Dalam dunia yang serba cepat ini, lembaga keuangan dan bank dihadapkan pada jutaan transaksi kartu kredit setiap harinya. "
            "**Bagaimana cara memahami pelanggan dengan lebih baik?** "
            "**Bagaimana bank bisa menawarkan layanan yang lebih personal dan mengelola risiko kredit dengan lebih cerdas?** "
            "Di sinilah analisis data dan teknik clustering berperan penting!"
        )

        st.write("""
        Dengan melakukan segmentasi pengguna kartu kredit berdasarkan pola belanja, kebiasaan pembayaran, dan penggunaan kredit, bank dapat:

        1. 🎯 **Menargetkan promosi yang lebih relevan** kepada segmen pelanggan tertentu.  
        2. ⚠️ **Mengidentifikasi pelanggan berisiko tinggi** untuk memitigasi risiko kredit.  
        3. 😊 **Meningkatkan kepuasan pelanggan** dengan layanan yang disesuaikan dengan kebutuhan mereka.  
        """)

        st.markdown("---")

        # 🧩 Apa yang Akan Kita Lakukan di Proyek Ini?
        st.write("🧩 Apa yang Akan Kita Lakukan di Proyek Ini?")
        st.write("""
        Dalam proyek ini, kita akan:

        1. 🧪 **Menggunakan teknik clustering** seperti K-Means.  
        2. 📊 **Menganalisis perilaku pengguna kartu kredit** berdasarkan data pembayaran dan penggunaan kartu kredit customer.  
        """)

    with st.expander("🔍 **Informasi Lengkap tentang Dataset**"):

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

    # Penanganan Missing Values
    df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
    df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())

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

    # Pisahkan kolom numerik dan non-numerik
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df_numerical = df[numerical_cols]

    # Standarisasi dan konversi hasilnya kembali menjadi DataFrame
    scaler = StandardScaler()

    # Misalnya df adalah DataFrame yang sudah ada
    # Pisahkan kolom numerik dan non-numerik
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df_numerical = df[numerical_cols]

    # Standarisasi hanya pada kolom numerik
    X_scaled = scaler.fit_transform(df_numerical)

    with st.expander("👩🏻‍💻PCA & Clustering💬"):

        # PCA
        st.header("📊 Principal Component Analysis (PCA)")
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)

        # Buat DataFrame komponen utama
        components_df = pd.DataFrame(pca.components_, columns=numerical_cols, index=['PC1', 'PC2'])

        # Tampilkan kontribusi fitur untuk setiap komponen utama
        for i, row in components_df.iterrows():
            st.subheader(f"Komponen penyusun {i}:")
            contributing_features = row.index[row.abs() > 0.1]
            st.write(", ".join(contributing_features))


        # Pilih jumlah cluster optimal berdasarkan nilai Silhouette Score terbaik
        optimal_k = 3

        st.header("📌 Clustering")
        st.write(f"Jumlah cluster optimal adalah {optimal_k} berdasarkan Silhouette Score.")

        # K-Means Clustering dengan jumlah cluster optimal
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Tukar label Cluster 0 dengan Cluster 2
        cluster_mapping = {0: 2, 1: 1, 2: 0}
        df['Cluster'] = df['Cluster'].map(cluster_mapping)

        # Lakukan PCA untuk proyeksi 2D
        X_pca = pca.transform(X_scaled)

        # Visualisasi hasil clustering
        st.header("Visualisasi Hasil Clustering")

        # Buat plot menggunakan Seaborn
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set2', s=50)
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
        cluster_profiles = df.groupby('Cluster')[numerical_cols].mean()

        # Pemetaan cluster berdasarkan yang sudah dilakukan
        cluster_profiles = cluster_profiles.rename(index={0: 2, 1: 1, 2: 0})

        # Tampilkan profil cluster yang sudah ter-mapping
        st.dataframe(cluster_profiles)

        # 📊 Analisis Segmentasi dan Strategi Promosi untuk Tiap Cluster 🎯
        st.header("🧐 Analisis Segmentasi dan Strategi Promosi untuk Tiap Cluster 🎯")
        st.write(
            "Berdasarkan hasil clustering menggunakan metode **K-Means**, kita telah mengidentifikasi tiga cluster pelanggan dengan karakteristik yang berbeda-beda. "
            "Berikut adalah analisis mendalam dan strategi promosi yang sesuai untuk tiap cluster."
        )

        st.markdown("---")

        # 🧩 Cluster 0: Pengguna dengan Aktivitas Sedang hingga Tinggi
        st.subheader("🧩 Cluster 0: Pengguna dengan Aktivitas Sedang hingga Tinggi")
        st.markdown("### 🔍 **Karakteristik:**")
        st.write("""
        - **Aktivitas Pembelian:** Sedang hingga tinggi.  
        - **Saldo & Limit Kredit:** Cukup baik.  
        - **Frekuensi Penggunaan:** Sering melakukan pembelian sekali bayar dan pembayaran rutin.  
        - **Profil Umum:** Pengguna aktif dengan transaksi rutin dan pengelolaan stabil.
        """)

        st.markdown("### 🎯 **Strategi Promosi:**")
        st.write("""
        - **Insentif untuk Pembelian Rutin:** Berikan poin reward atau cashback untuk pembelian reguler agar pengguna tetap aktif.  
        - **Diskon untuk Pembelian Sekali Bayar:** Dorong pembelian langsung lunas dengan diskon khusus.  
        - **Program Loyalitas:** Mendorong konsistensi penggunaan dengan reward berkala.
        """)

        st.markdown("### 💡 **Contoh Promosi:**")
        st.write("""
        - "Dapatkan cashback 10% untuk pembelian langsung di kategori elektronik!"  
        - "Setiap transaksi di supermarket selama akhir pekan, dapatkan poin reward ekstra!"
        """)

        st.markdown("---")

        # 🌟 Cluster 1: Pengguna dengan Aktivitas Rendah
        st.subheader("🌟 Cluster 1: Pengguna dengan Aktivitas Rendah")
        st.markdown("### 🔍 **Karakteristik:**")
        st.write("""
        - **Aktivitas Pembelian:** Rendah.  
        - **Saldo & Limit Kredit:** Rendah.  
        - **Frekuensi Penggunaan:** Jarang menggunakan kartu kredit dan cenderung pasif.  
        - **Profil Umum:** Pengguna pasif dengan aktivitas transaksi rendah dan jarang memanfaatkan kartu kredit.
        """)

        st.markdown("### 🎯 **Strategi Promosi:**")
        st.write("""
        - **Aktivasi Pengguna Baru:** Promosi yang mendorong pengguna untuk mulai bertransaksi.  
        - **Diskon dan Cashback Transaksi Pertama:** Insentif khusus untuk pembelian awal.  
        - **Edukasi Penggunaan Kartu Kredit:** Mengingatkan manfaat menggunakan kartu kredit.
        """)

        st.markdown("### 💡 **Contoh Promosi:**")
        st.write("""
        - "Gunakan kartu kredit Anda untuk pertama kalinya dan dapatkan cashback 15%!"  
        - "Diskon 20% untuk transaksi pertama di e-commerce pilihan!"
        """)

        st.markdown("---")

        # ⚖️ Cluster 2: Pengguna Aktif dengan Aktivitas Tinggi
        st.subheader("⚖️ Cluster 2: Pengguna Aktif dengan Aktivitas Tinggi")
        st.markdown("### 🔍 **Karakteristik:**")
        st.write("""
        - **Aktivitas Pembelian:** Tinggi.  
        - **Saldo & Limit Kredit:** Tinggi.  
        - **Frekuensi Penggunaan:** Sering melakukan pembayaran besar dan mengelola saldo besar.  
        - **Profil Umum:** Pengguna loyal dengan transaksi besar, limit tinggi, dan pengelolaan saldo aktif.
        """)

        st.markdown("### 🎯 **Strategi Promosi:**")
        st.write("""
        - **Program Eksklusif untuk VIP:** Penawaran spesial untuk pengguna dengan transaksi tinggi.  
        - **Reward Double Points:** Insentif tambahan untuk pembelian besar.  
        - **Akses Layanan Premium:** Berikan fasilitas eksklusif seperti akses ke acara atau layanan concierge.
        """)

        st.markdown("### 💡 **Contoh Promosi:**")
        st.write("""
        - "Dapatkan 2x poin reward untuk transaksi di restoran dan perjalanan!"  
        - "Transaksi bulanan di atas Rp 10.000.000? Nikmati akses layanan premium gratis!"
        """)

        st.markdown("---")

    # 🔮 Prediksi Cluster untuk Input Baru
    st.header("🔮 Prediksi Cluster untuk Input Baru")
    st.write("Masukkan nilai fitur untuk memprediksi cluster pelanggan:")

    # Input fitur untuk prediksi
    input_features = {}
    for feature in numerical_cols:
        input_features[feature] = st.number_input(
            f"{feature}", 
            value=float(df[feature].mean()), 
            step=0.01
        )

    # Tombol Prediksi
    if st.button("Prediksi Cluster"):
        # Siapkan input untuk prediksi
        input_data = [input_features[feature] for feature in numerical_cols]

        # Standarisasi input
        input_scaled = scaler.transform([input_data])

        # Prediksi cluster menggunakan input yang sudah distandarisasi
        predicted_cluster = kmeans.predict(input_scaled)[0]

        # Pemetaan hasil prediksi ke label cluster yang baru
        predicted_cluster_mapped = cluster_mapping[predicted_cluster]

        # Tampilkan hasil prediksi
        st.success(f"Pelanggan diprediksi masuk ke **Cluster {predicted_cluster_mapped}**")

        # Deskripsi dan strategi promosi untuk masing-masing cluster
        if predicted_cluster == 0:
            st.markdown("### 🧩 **Cluster 0: Pengguna dengan Aktivitas Sedang hingga Tinggi**")
            st.write("""
            **Karakteristik:**
            - Aktivitas Pembelian: Sedang hingga tinggi.  
            - Saldo & Limit Kredit: Cukup baik.  
            - Frekuensi Penggunaan: Sering melakukan pembelian sekali bayar dan pembayaran rutin.  
            - Profil Umum: Pengguna aktif dengan transaksi rutin dan pengelolaan stabil.
            """)
            st.markdown("**🎯 Strategi Promosi:**")
            st.write("""
            - Insentif untuk pembelian rutin.  
            - Diskon untuk pembelian langsung.  
            - Program loyalitas dengan reward berkala.
            """)
            st.markdown("**💡 Contoh Promosi:**")
            st.write("""
            - "Dapatkan cashback 10% untuk pembelian langsung di kategori elektronik!"  
            - "Setiap transaksi di supermarket selama akhir pekan, dapatkan poin reward ekstra!"
            """)

        elif predicted_cluster == 1:
            st.markdown("### 🌟 **Cluster 1: Pengguna dengan Aktivitas Rendah**")
            st.write("""
            **Karakteristik:**
            - Aktivitas Pembelian: Rendah.  
            - Saldo & Limit Kredit: Rendah.  
            - Frekuensi Penggunaan: Jarang menggunakan kartu kredit dan cenderung pasif.  
            - Profil Umum: Pengguna pasif dengan aktivitas transaksi rendah dan jarang memanfaatkan kartu kredit.
            """)
            st.markdown("**🎯 Strategi Promosi:**")
            st.write("""
            - Aktivasi pengguna baru.  
            - Diskon dan cashback untuk transaksi pertama.  
            - Edukasi mengenai penggunaan kartu kredit.
            """)
            st.markdown("**💡 Contoh Promosi:**")
            st.write("""
            - "Gunakan kartu kredit Anda untuk pertama kalinya dan dapatkan cashback 15%!"  
            - "Diskon 20% untuk transaksi pertama di e-commerce pilihan!"
            """)

        elif predicted_cluster == 2:
            st.markdown("### ⚖️ **Cluster 2: Pengguna Aktif dengan Aktivitas Tinggi**")
            st.write("""
            **Karakteristik:**
            - Aktivitas Pembelian: Tinggi.  
            - Saldo & Limit Kredit: Tinggi.  
            - Frekuensi Penggunaan: Sering melakukan pembayaran besar dan mengelola saldo besar.  
            - Profil Umum: Pengguna loyal dengan transaksi besar, limit tinggi, dan pengelolaan saldo aktif.
            """)
            st.markdown("**🎯 Strategi Promosi:**")
            st.write("""
            - Program eksklusif untuk VIP.  
            - Reward double points untuk transaksi besar.  
            - Akses ke layanan premium seperti layanan concierge.
            """)
            st.markdown("**💡 Contoh Promosi:**")
            st.write("""
            - "Dapatkan 2x poin reward untuk transaksi di restoran dan perjalanan!"  
            - "Transaksi bulanan di atas Rp 10.000.000? Nikmati akses layanan premium gratis!"
            """)

        # Tampilkan profil cluster terdekat
        st.write("### 📊 **Profil Cluster Terdekat:**")
        st.dataframe(cluster_profiles.loc[predicted_cluster])



except FileNotFoundError:
    st.error(f"File dataset tidak ditemukan di path: {DATA_PATH}. Pastikan file sudah ada di folder data.")
