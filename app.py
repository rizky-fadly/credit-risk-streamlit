import streamlit as st
import pandas as pd
import joblib

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Risiko Kartu Kredit",
    layout="centered"
)

st.title("Prediksi Risiko Gagal Bayar Kartu Kredit")
st.write(
    "Aplikasi ini memprediksi risiko gagal bayar kartu kredit "
    "menggunakan model **XGBoost**."
)

# =========================
# LOAD MODEL & FITUR
# =========================
model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# DEMO DATA
# =========================
demo_data = {
    "LIMIT_BAL": 50000000,
    "SEX": 1,
    "EDUCATION": 4,
    "MARRIAGE": 2,
    "AGE": 30,

    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,

    "BILL_AMT1": 8000000,
    "BILL_AMT2": 7500000,
    "BILL_AMT3": 7000000,
    "BILL_AMT4": 6800000,
    "BILL_AMT5": 6500000,
    "BILL_AMT6": 6000000,

    "PAY_AMT1": 3000000,
    "PAY_AMT2": 3000000,
    "PAY_AMT3": 3000000,
    "PAY_AMT4": 3000000,
    "PAY_AMT5": 3000000,
    "PAY_AMT6": 3000000,
}

if "demo" not in st.session_state:
    st.session_state.demo = False

if st.button("🎯 Isi Contoh Data (Demo)"):
    st.session_state.demo = True

use_demo = st.session_state.demo

# =========================
# BAGIAN 1: DATA NASABAH
# =========================
st.header("📋 Data Nasabah")

input_data = {}

# ---- INPUT KATEGORIK ----
sex_map = {"Laki-laki": 1, "Perempuan": 2}
edu_map = {
    "Sekolah Dasar": 1,
    "SMA": 2,
    "Diploma": 3,
    "Sarjana": 4,
    "Pascasarjana": 5,
    "Lainnya": 6
}
marriage_map = {
    "Menikah": 1,
    "Belum Menikah": 2,
    "Lainnya": 3
}

sex_label = st.selectbox("Jenis Kelamin", list(sex_map.keys()))
edu_label = st.selectbox("Pendidikan Terakhir", list(edu_map.keys()))
marriage_label = st.selectbox("Status Pernikahan", list(marriage_map.keys()))
age_val = st.text_input(
    "Usia (tahun)",
    value=str(demo_data["AGE"]) if use_demo else "30"
)

input_data["SEX"] = sex_map[sex_label]
input_data["EDUCATION"] = edu_map[edu_label]
input_data["MARRIAGE"] = marriage_map[marriage_label]
input_data["AGE"] = int(age_val) if age_val.isdigit() else 30

st.divider()

# =========================
# RIWAYAT KREDIT
# =========================
st.subheader("Riwayat Kredit")

# ---- STATUS KETERLAMBATAN ----
pay_options = {
    "Tepat waktu": 0,
    "Terlambat 1 bulan": 1,
    "Terlambat 2 bulan": 2,
    "Terlambat ≥3 bulan": 3
}

for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
    default_val = demo_data.get(col, 0) if use_demo else 0
    selected = st.selectbox(
        col,
        list(pay_options.keys()),
        index=min(default_val, 3)
    )
    input_data[col] = pay_options[selected]

st.divider()

# ---- NOMINAL (TANPA FORMAT, AMAN CLOUD) ----
for col in feature_names:
    if col.startswith("BILL_AMT") or col.startswith("PAY_AMT") or col == "LIMIT_BAL":
        default_val = demo_data.get(col, 0) if use_demo else 0
        input_data[col] = st.number_input(
            col,
            value=float(default_val),
            min_value=0.0
        )

# =========================
# BAGIAN 2: HASIL
# =========================
st.header("📊 Hasil Prediksi")

if st.button("Prediksi Risiko"):
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_names]

    prob = model.predict_proba(df_input)[0][1]

    st.metric("Probabilitas Gagal Bayar", f"{prob * 100:.2f} %")

    st.write(f"**Limit Kredit:** Rp {input_data['LIMIT_BAL']:,.0f}")

    if prob >= 0.5:
        st.error("⚠️ Risiko Tinggi Gagal Bayar")
    else:
        st.success("✅ Risiko Rendah Gagal Bayar")
