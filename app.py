import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter(action="ignore", category=PerformanceWarning)

st.set_page_config(page_title="Uji Korelasi", layout="wide")

# --- UI PILIH DATA ---
st.title("📊 Aplikasi Uji Korelasi IPK Mahasiswa")

selected_data = st.radio("Pilih jenis data yang ingin diuji korelasinya:", 
                         ["Data Survey Sosial Ekonomi", "Data Nilai Transkrip Mahasiswa"])

st.write("---")

# === FUNGSI UNTUK DATA 1 ===
def korelasi_data_survey():
    st.subheader("📁 Hasil Uji Korelasi - Data Survey Sosial Ekonomi")

    survey_df = pd.read_excel("data_survey.xlsx")

    survey_df.rename(columns={
        'IPK - (Skala 0.00 - 4.00)': 'IPK',
        'Pendidikan terakhir Ayah': 'pendidikan_ayah',
        'Pendidikan terakhir Ibu': 'pendidikan_ibu',
        'Pendapatan orang tua per bulan': 'pendapatan',
        'Seberapa sering Anda mengalami kendala finansial dalam memenuhi kebutuhan akademik? (Seperti: buku, biaya praktikum, internet, atau alat pendukung lainnya)': 'kendala_finansial',
        'Seberapa sering Anda mengakses sumber belajar tambahan? (Misalnya: jurnal akademik, kelas tambahan, atau bimbingan belajar)': 'akses_sumber_belajar',
        'Selama kuliah, apakah kebutuhan finansial Anda lebih banyak ditanggung oleh?': 'penanggung_biaya'
    }, inplace=True)

    pendidikan_mapping = {'SD': 1, 'SMP': 2, 'SMA/SMK': 3, 'D3': 4, 'S1': 5, 'S2': 6, 'S3': 7, 'Tidak Ada': 0}
    pendapatan_mapping = {'< Rp 3.000.000': 1, 'Rp 3.000.000 – Rp 6.000.000': 2, 'Rp 6.000.000 – Rp 10.000.000': 3, '> Rp 10.000.000': 4}
    frekuensi_mapping = {'Sangat jarang': 1, 'Jarang': 2, 'Kadang-kadang': 3, 'Sering': 4, 'Sangat sering': 5, 'Tidak pernah': 0}
    penanggung_biaya_mapping = {
        'Orang tua/wali sepenuhnya': 1,
        'Orang tua/wali sebagian, sisanya dari beasiswa atau kerja': 2,
        'Beasiswa sepenuhnya': 3,
        'Hasil kerja sendiri sepenuhnya': 4
    }

    survey_df['pendidikan_ayah'] = survey_df['pendidikan_ayah'].map(pendidikan_mapping)
    survey_df['pendidikan_ibu'] = survey_df['pendidikan_ibu'].map(pendidikan_mapping)
    survey_df['pendapatan'] = survey_df['pendapatan'].map(pendapatan_mapping)
    survey_df['kendala_finansial'] = survey_df['kendala_finansial'].map(frekuensi_mapping)
    survey_df['akses_sumber_belajar'] = survey_df['akses_sumber_belajar'].map(frekuensi_mapping)
    survey_df['penanggung_biaya'] = survey_df['penanggung_biaya'].map(penanggung_biaya_mapping)

    # Korelasi Spearman
    numerical_cols = ['pendidikan_ayah', 'pendidikan_ibu', 'pendapatan',
                      'kendala_finansial', 'akses_sumber_belajar', 'penanggung_biaya']
    
    results = []
    for col in numerical_cols:
        corr, p = spearmanr(survey_df[col], survey_df['IPK'])
        results.append((col, corr, p))

    # Tampilkan hasil
    st.write("### Tabel Hasil Uji Korelasi Spearman")
    st.table(pd.DataFrame(results, columns=["Variabel", "Spearman ρ", "p-value"]).round(3))

    # Heatmap
    st.write("### 🔥 Visualisasi Korelasi (Heatmap)")
    corr_df = survey_df[['IPK'] + numerical_cols]
    corr_matrix = corr_df.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# === FUNGSI UNTUK DATA 2 ===
def korelasi_data_nilai():
    st.subheader("📁 Hasil Uji Korelasi - Data Nilai Transkrip Mahasiswa")

    # Upload file dari user
    uploaded_transkrip = st.file_uploader("Unggah file data transkrip mahasiswa (.xlsx)", type=["xlsx"])
    uploaded_sks = st.file_uploader("Unggah file mapping SKS mata kuliah (.csv)", type=["csv"])

    # Validasi file
if uploaded_transkrip is None or uploaded_sks is None:
    st.warning("Mohon unggah kedua file untuk melanjutkan analisis.")
    return

# Load data dari file upload
analysis_df = pd.read_excel(uploaded_transkrip)
sks_df = pd.read_csv(uploaded_sks)

    # Identifikasi kolom nilai dan kehadiran
    nilai_cols = [col for col in analysis_df.columns if col.endswith("(nilai)")]
    hadir_cols = [col.replace("(nilai)", "(hadir)") for col in nilai_cols]
    mk_names = [col.replace(" (nilai)", "") for col in nilai_cols]

    # Pastikan kolom hadir bertipe numerik
    for col in hadir_cols:
        if col in analysis_df.columns:
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')

    # Definisikan bobot nilai huruf
    bobot_ipk = {
        "A": 4.0, "A-": 3.75, "B+": 3.5, "B": 3.0, "B-": 2.75,
        "C+": 2.5, "C": 2.0, "D": 1.0, "E": 0.0, "F": 0.0
    }

    # Fungsi konversi nilai numerik ke nilai huruf
    def konversi_nilai(nilai, hadir):
        if pd.isna(hadir) or hadir < 11:
            return pd.NA
        if pd.isna(nilai):
            return pd.NA
        nilai = float(nilai)
        if nilai >= 85: return "A"
        elif nilai >= 80: return "A-"
        elif nilai >= 75: return "B+"
        elif nilai >= 70: return "B"
        elif nilai >= 65: return "B-"
        elif nilai >= 60: return "C+"
        elif nilai >= 55: return "C"
        elif nilai >= 45: return "D"
        else: return "E"

    # Buat kolom nilai huruf
    for mk, n_col, h_col in zip(mk_names, nilai_cols, hadir_cols):
        if n_col in analysis_df.columns and h_col in analysis_df.columns:
            analysis_df[f"{mk} (nilai_huruf)"] = analysis_df.apply(
                lambda row: konversi_nilai(row[n_col], row[h_col]), axis=1
            )

    # Tambahkan kolom SKS sesuai mata kuliah
    analysis_df_merged = analysis_df.copy()
    for mk in mk_names:
        if mk in sks_df["mata_kuliah"].values:
            sks_value = sks_df[sks_df["mata_kuliah"] == mk]["sks"].values[0]
            analysis_df_merged[f"{mk}_sks"] = sks_value

    # Hitung IPK tiap mahasiswa
    ipk_list = []
    for idx, row in analysis_df_merged.iterrows():
        total_bobot = 0
        total_sks = 0
        for mk in mk_names:
            nilai_huruf_col = f"{mk} (nilai_huruf)"
            sks_col = f"{mk}_sks"
            if nilai_huruf_col in analysis_df_merged.columns and sks_col in analysis_df_merged.columns:
                nilai_huruf = row[nilai_huruf_col]
                sks = row[sks_col]
                if pd.notna(nilai_huruf) and pd.notna(sks):
                    total_bobot += bobot_ipk.get(nilai_huruf, 0) * sks
                    total_sks += sks
        ipk = round(total_bobot / total_sks, 2) if total_sks > 0 else np.nan
        ipk_list.append(ipk)
    analysis_df_merged["IPK"] = ipk_list

    # Kolom yang dibutuhkan
    required_cols = ['jenis_kelamin', 'pendidikan_ayah', 'pendidikan_ibu', 'IPK']

    # Filter data yang lengkap dan valid
    pendidikan_valid = ['Tidak Sekolah', 'SD', 'SMP', 'SMA', 'D1', 'D2', 'D3', 'D4', 'S1', 'S2', 'S3']
    final_df = analysis_df_merged[required_cols].copy()
    final_df = final_df[
        final_df['pendidikan_ayah'].isin(pendidikan_valid) &
        final_df['pendidikan_ibu'].isin(pendidikan_valid)
    ].copy()
    final_df.dropna(inplace=True)

    # Encoding kategorikal ke numerik
    pendidikan_map = {
        "Tidak Sekolah": 0, "SD": 1, "SMP": 2, "SMA": 3, "D1": 4, "D2": 5,
        "D3": 6, "D4": 7, "S1": 8, "S2": 9, "S3": 10
    }
    final_df['pendidikan_ayah'] = final_df['pendidikan_ayah'].map(pendidikan_map)
    final_df['pendidikan_ibu'] = final_df['pendidikan_ibu'].map(pendidikan_map)
    final_df['jenis_kelamin'] = final_df['jenis_kelamin'].map({'Laki-laki': 1, 'Perempuan': 0})

    # Uji Korelasi Spearman
    cols_for_corr = ['pendidikan_ayah', 'pendidikan_ibu']
    results = []
    for col in cols_for_corr:
        rho, p_val = spearmanr(final_df[col], final_df['IPK'])
        results.append((col, rho, p_val))

    # Tampilkan hasil korelasi
    st.write("### Tabel Hasil Uji Korelasi Spearman")
    st.table(pd.DataFrame(results, columns=["Variabel", "Spearman ρ", "p-value"]).round(3))

    # Visualisasi Heatmap
    st.write("### 🔥 Visualisasi Korelasi (Heatmap)")
    corr_df = final_df[['IPK'] + cols_for_corr]
    corr_matrix = corr_df.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)


# === HANDLE PILIHAN ===
if selected_data == "Data Survey Sosial Ekonomi":
    korelasi_data_survey()
elif selected_data == "Data Nilai Transkrip Mahasiswa":
    korelasi_data_nilai()
