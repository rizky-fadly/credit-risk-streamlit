import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Prediksi Risiko Kredit", layout="centered")

st.title("Prediksi Risiko Gagal Bayar Kartu Kredit")
st.write(
    "Aplikasi ini memprediksi risiko gagal bayar kartu kredit "
    "menggunakan model **XGBoost** dan menjelaskan hasilnya dengan **SHAP**."
)

# =========================
# LOAD MODEL & FITUR
# =========================
model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")
explainer = shap.TreeExplainer(model)

# =========================
# DATA CONTOH
# =========================
contoh_rendah = {
    "LIMIT_BAL": 50000000, "AGE": 30, "SEX": 1, "EDUCATION": 4, "MARRIAGE": 2,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 8000000, "BILL_AMT2": 7500000, "BILL_AMT3": 7000000,
    "BILL_AMT4": 6800000, "BILL_AMT5": 6500000, "BILL_AMT6": 6000000,
    "PAY_AMT1": 3000000, "PAY_AMT2": 3000000, "PAY_AMT3": 3000000,
    "PAY_AMT4": 3000000, "PAY_AMT5": 3000000, "PAY_AMT6": 3000000,
}

contoh_tinggi = {
    "LIMIT_BAL": 20000000, "AGE": 25, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 1, "PAY_4": 3, "PAY_5": 2, "PAY_6": 1,
    "BILL_AMT1": 15000000, "BILL_AMT2": 14000000, "BILL_AMT3": 13500000,
    "BILL_AMT4": 13000000, "BILL_AMT5": 12500000, "BILL_AMT6": 12000000,
    "PAY_AMT1": 500000, "PAY_AMT2": 500000, "PAY_AMT3": 500000,
    "PAY_AMT4": 500000, "PAY_AMT5": 500000, "PAY_AMT6": 500000,
}

if "contoh_data" not in st.session_state:
    st.session_state.contoh_data = None

c1, c2 = st.columns(2)
with c1:
    if st.button("🟢 Contoh Risiko Rendah"):
        st.session_state.contoh_data = contoh_rendah
with c2:
    if st.button("🔴 Contoh Risiko Tinggi"):
        st.session_state.contoh_data = contoh_tinggi

use_data = st.session_state.contoh_data
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
    "Usia Nasabah (tahun)", 18, 100,
    value=use_data["AGE"] if use_data else 30
)

limit_bal = st.number_input(
    "Limit Kartu Kredit (Rp)", min_value=0.0,
    value=float(use_data["LIMIT_BAL"]) if use_data else 10000000.0
)
st.caption(f"≈ Rp {limit_bal:,.0f}")
input_data["LIMIT_BAL"] = limit_bal

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
    "PAY_0": "Pembayaran Bulan Ini",
    "PAY_2": "2 Bulan Lalu",
    "PAY_3": "3 Bulan Lalu",
    "PAY_4": "4 Bulan Lalu",
    "PAY_5": "5 Bulan Lalu",
    "PAY_6": "6 Bulan Lalu",
}

for col, label in pay_labels.items():
    default = use_data[col] if use_data else 0
    input_data[col] = status_bayar[
        st.selectbox(label, status_bayar, index=min(default, 3))
    ]

# =========================
# TAGIHAN & PEMBAYARAN
# =========================
st.subheader("💳 Tagihan & Pembayaran (Rp)")

for i in range(1, 7):
    bill = st.number_input(
        f"Tagihan Bulan ke-{i}", min_value=0.0,
        value=float(use_data[f"BILL_AMT{i}"]) if use_data else 0.0
    )
    st.caption(f"≈ Rp {bill:,.0f}")
    input_data[f"BILL_AMT{i}"] = bill

for i in range(1, 7):
    pay = st.number_input(
        f"Pembayaran Bulan ke-{i}", min_value=0.0,
        value=float(use_data[f"PAY_AMT{i}"]) if use_data else 0.0
    )
    st.caption(f"≈ Rp {pay:,.0f}")
    input_data[f"PAY_AMT{i}"] = pay

# =========================
# HASIL PREDIKSI
# =========================
st.header("📊 Hasil Prediksi")

if st.button("🔍 Prediksi Risiko"):
    df = pd.DataFrame([input_data])[feature_names]
    prob = model.predict_proba(df)[0][1]

    st.metric("Probabilitas Gagal Bayar", f"{prob*100:.2f}%")

    if prob >= 0.5:
        st.error("⚠️ Risiko Tinggi Gagal Bayar")
    else:
        st.success("✅ Risiko Rendah Gagal Bayar")

    # =========================
    # SHAP EXPLANATION
    # =========================
    st.subheader("🔎 Penjelasan Prediksi (SHAP)")

    shap_values = explainer.shap_values(df)
    shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    shap_df = pd.DataFrame({
        "Fitur": feature_names,
        "Pengaruh": shap_vals
    })

    shap_df["|Pengaruh|"] = shap_df["Pengaruh"].abs()
    shap_df = shap_df.sort_values("|Pengaruh|", ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(shap_df["Fitur"], shap_df["Pengaruh"])
    ax.axvline(0)
    ax.set_title("10 Fitur Paling Berpengaruh")
    ax.set_xlabel("Pengaruh terhadap Risiko Gagal Bayar")
    plt.gca().invert_yaxis()

    st.pyplot(fig)

    top_feature = shap_df.iloc[0]["Fitur"]
    st.info(
        f"Fitur paling berpengaruh pada prediksi ini adalah **{top_feature}**. "
        "Nilai positif meningkatkan risiko gagal bayar, sedangkan nilai negatif menurunkannya."
    )
