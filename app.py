import streamlit as st
import pandas as pd
import joblib

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Risiko Kredit",
    layout="centered"
)

st.title("Prediksi Risiko Gagal Bayar Kartu Kredit")
st.write(
    "Aplikasi ini digunakan untuk memprediksi kemungkinan nasabah "
    "mengalami gagal bayar kartu kredit menggunakan model **XGBoost**."
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# DEMO DATA
# =========================
demo = {
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

if st.button("🎯 Gunakan Data Contoh"):
    st.session_state.demo = True

use_demo = st.session_state.demo

# =========================
# BAGIAN 1: DATA NASABAH
# =========================
st.header("📋 Data Nasabah")
input_data = {}

# --- DATA PRIBADI ---
sex_map = {"Laki-laki": 1, "Perempuan": 2}
edu_map = {
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

input_data["SEX"] = sex_map[
    st.selectbox("Jenis Kelamin", sex_map.keys())
]

input_data["EDUCATION"] = edu_map[
    st.selectbox("Pendidikan Terakhir", edu_map.keys())
]

input_data["MARRIAGE"] = marriage_map[
    st.selectbox("Status Pernikahan", marriage_map.keys())
]

input_data["AGE"] = st.number_input(
    "Usia Nasabah (tahun)",
    min_value=18,
    max_value=100,
    value=demo["AGE"] if use_demo else 30
)

input_data["LIMIT_BAL"] = st.number_input(
    "Limit Kartu Kredit (Rp)",
    min_value=0.0,
    value=float(demo["LIMIT_BAL"]) if use_demo else 10000000.0
)

st.divider()

# =========================
# RIWAYAT PEMBAYARAN
# =========================
st.subheader("📆 Riwayat Pembayaran Kredit")

status_bayar = {
    "Tepat waktu": 0,
    "Terlambat 1 bulan": 1,
    "Terlambat 2 bulan": 2,
    "Terlambat lebih dari 3 bulan": 3
}

pay_labels = {
    "PAY_0": "Status Pembayaran Bulan Ini",
    "PAY_2": "Status Pembayaran 2 Bulan Lalu",
    "PAY_3": "Status Pembayaran 3 Bulan Lalu",
    "PAY_4": "Status Pembayaran 4 Bulan Lalu",
    "PAY_5": "Status Pembayaran 5 Bulan Lalu",
    "PAY_6": "Status Pembayaran 6 Bulan Lalu",
}

for col, label in pay_labels.items():
    default = demo[col] if use_demo else 0
    input_data[col] = status_bayar[
        st.selectbox(label, status_bayar.keys(), index=min(default, 3))
    ]

st.divider()

# =========================
# TAGIHAN & PEMBAYARAN
# =========================
st.subheader("💳 Tagihan & Pembayaran (Rp)")

for i in range(1, 7):
    input_data[f"BILL_AMT{i}"] = st.number_input(
        f"Jumlah Tagihan Bulan ke-{i}",
        min_value=0.0,
        value=float(demo[f"BILL_AMT{i}"]) if use_demo else 0.0
    )

for i in range(1, 7):
    input_data[f"PAY_AMT{i}"] = st.number_input(
        f"Jumlah Pembayaran Bulan ke-{i}",
        min_value=0.0,
        value=float(demo[f"PAY_AMT{i}"]) if use_demo else 0.0
    )

# =========================
# HASIL PREDIKSI
# =========================
st.header("📊 Hasil Prediksi")

if st.button("🔍 Prediksi Risiko"):
    df = pd.DataFrame([input_data])[feature_names]
    prob = model.predict_proba(df)[0][1]

    st.metric(
        "Probabilitas Gagal Bayar",
        f"{prob * 100:.2f}%"
    )

    st.write(f"**Limit Kredit:** Rp {input_data['LIMIT_BAL']:,.0f}")

    if prob >= 0.5:
        st.error("⚠️ Risiko Tinggi Gagal Bayar")
    else:
        st.success("✅ Risiko Rendah Gagal Bayar")
