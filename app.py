import streamlit as st
import pandas as pd
import joblib

# =========================
# KONFIGURASI
# =========================
st.set_page_config(page_title="Prediksi Risiko Kredit", layout="centered")

st.title("Prediksi Risiko Gagal Bayar Kartu Kredit")
st.write(
    "Aplikasi ini memprediksi risiko gagal bayar kartu kredit "
    "menggunakan model **XGBoost**."
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
    "AGE": 30,
    "SEX": 1,
    "EDUCATION": 4,
    "MARRIAGE": 2,
    **{f"PAY_{i}": 0 for i in [0,2,3,4,5,6]},
    **{f"BILL_AMT{i}": 8000000 - (i-1)*500000 for i in range(1,7)},
    **{f"PAY_AMT{i}": 3000000 for i in range(1,7)},
}

if "demo" not in st.session_state:
    st.session_state.demo = False

if st.button("🎯 Gunakan Data Contoh"):
    st.session_state.demo = True

use_demo = st.session_state.demo
input_data = {}

# =========================
# DATA NASABAH
# =========================
st.header("📋 Data Nasabah")

sex_map = {"Laki-laki": 1, "Perempuan": 2}
edu_map = {"SMA": 2, "Diploma": 3, "Sarjana": 4, "Pascasarjana": 5, "Lainnya": 6}
marriage_map = {"Menikah": 1, "Belum Menikah": 2, "Lainnya": 3}

input_data["SEX"] = sex_map[st.selectbox("Jenis Kelamin", sex_map)]
input_data["EDUCATION"] = edu_map[st.selectbox("Pendidikan Terakhir", edu_map)]
input_data["MARRIAGE"] = marriage_map[st.selectbox("Status Pernikahan", marriage_map)]

input_data["AGE"] = st.number_input(
    "Usia Nasabah (tahun)",
    min_value=18,
    max_value=100,
    value=demo["AGE"] if use_demo else 30
)

# ---- LIMIT ----
limit_bal = st.number_input(
    "Limit Kartu Kredit (Rp)",
    min_value=0.0,
    value=float(demo["LIMIT_BAL"]) if use_demo else 10000000.0
)
st.caption(f"≈ Rp {limit_bal:,.0f}")
input_data["LIMIT_BAL"] = limit_bal

st.divider()

# =========================
# RIWAYAT PEMBAYARAN
# =========================
st.subheader("📆 Riwayat Pembayaran Kredit")

status_bayar = {
    "Tepat waktu": 0,
    "Terlambat 1 bulan": 1,
    "Terlambat 2 bulan": 2,
    "Terlambat > 3 bulan": 3
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
# TAGIHAN (LIVE PREVIEW)
# =========================
st.subheader("💳 Jumlah Tagihan per Bulan")

for i in range(1, 7):
    val = st.number_input(
        f"Tagihan Bulan ke-{i} (Rp)",
        min_value=0.0,
        value=float(demo[f"BILL_AMT{i}"]) if use_demo else 0.0
    )
    st.caption(f"≈ Rp {val:,.0f}")
    input_data[f"BILL_AMT{i}"] = val

st.divider()

# =========================
# PEMBAYARAN (LIVE PREVIEW)
# =========================
st.subheader("💰 Jumlah Pembayaran per Bulan")

for i in range(1, 7):
    val = st.number_input(
        f"Pembayaran Bulan ke-{i} (Rp)",
        min_value=0.0,
        value=float(demo[f"PAY_AMT{i}"]) if use_demo else 0.0
    )
    st.caption(f"≈ Rp {val:,.0f}")
    input_data[f"PAY_AMT{i}"] = val

# =========================
# HASIL
# =========================
st.header("📊 Hasil Prediksi")

if st.button("🔍 Prediksi Risiko"):
    df = pd.DataFrame([input_data])[feature_names]
    prob = model.predict_proba(df)[0][1]

    st.metric("Probabilitas Gagal Bayar", f"{prob*100:.2f}%")
    st.write(f"**Limit Kredit:** Rp {input_data['LIMIT_BAL']:,.0f}")

    if prob >= 0.5:
        st.error("⚠️ Risiko Tinggi Gagal Bayar")
    else:
        st.success("✅ Risiko Rendah Gagal Bayar")
