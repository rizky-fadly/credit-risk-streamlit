import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# OPTIONAL SHAP
# ======================
try:
    import shap
    SHAP_OK = True
except:
    SHAP_OK = False

st.set_page_config(page_title="Prediksi Risiko Kredit", layout="wide")
st.title("📊 Prediksi Credit Default Menggunakan XGBoost & SHAP")

# ======================
# LOAD MODEL
# ======================
model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ======================
# DEMO DATA
# ======================
LOW_RISK = {
    "LIMIT_BAL": 120_000_000,
    "AGE": 35,
    "SEX": 1,
    "EDUCATION": 3,
    "MARRIAGE": 2,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0,
    "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
}

HIGH_RISK = {
    "LIMIT_BAL": 15_000_000,
    "AGE": 28,
    "SEX": 1,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 2,
    "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
}

for i in range(1, 7):
    LOW_RISK[f"BILL_AMT{i}"] = 5_000_000
    LOW_RISK[f"PAY_AMT{i}"] = 5_000_000

    HIGH_RISK[f"BILL_AMT{i}"] = 18_000_000
    HIGH_RISK[f"PAY_AMT{i}"] = 500_000

# ======================
# SESSION STATE
# ======================
if "input_data" not in st.session_state:
    st.session_state.input_data = LOW_RISK.copy()

# ======================
# BUTTON DEMO
# ======================
c1, c2 = st.columns(2)
with c1:
    if st.button("🟢 Contoh Risiko Rendah"):
        st.session_state.input_data = LOW_RISK.copy()

with c2:
    if st.button("🔴 Contoh Risiko Tinggi"):
        st.session_state.input_data = HIGH_RISK.copy()

# ======================
# INPUT FORM
# ======================
st.header("1️⃣ Data Nasabah")

input_data = {}

col1, col2 = st.columns(2)

with col1:
    input_data["LIMIT_BAL"] = st.number_input(
        "Limit Kredit (Rp)",
        min_value=0,
        value=int(st.session_state.input_data["LIMIT_BAL"]),
        step=1_000_000,
        key="limit"
    )

    input_data["AGE"] = st.number_input(
        "Usia",
        min_value=17,
        max_value=100,
        value=int(st.session_state.input_data["AGE"]),
        key="age"
    )

    input_data["SEX"] = st.selectbox(
        "Jenis Kelamin",
        options=[1, 2],
        format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan",
        index=0 if st.session_state.input_data["SEX"] == 1 else 1
    )

    input_data["EDUCATION"] = st.selectbox(
        "Pendidikan Terakhir",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Pascasarjana",
            2: "Sarjana",
            3: "SMA",
            4: "Lainnya"
        }[x],
        index=st.session_state.input_data["EDUCATION"] - 1
    )

    input_data["MARRIAGE"] = st.selectbox(
        "Status Pernikahan",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Belum Menikah",
            2: "Menikah",
            3: "Lainnya"
        }[x],
        index=st.session_state.input_data["MARRIAGE"] - 1
    )

with col2:
    st.subheader("Riwayat Keterlambatan Pembayaran")

    for p in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        input_data[p] = st.selectbox(
            p,
            options=[-1, 0, 1, 2, 3],
            index=2 if st.session_state.input_data[p] >= 1 else 1,
            help="0 = Lancar, 1–3 = Terlambat"
        )

# ======================
# TAGIHAN & PEMBAYARAN
# ======================
st.subheader("Tagihan & Pembayaran (6 Bulan Terakhir)")
for i in range(1, 7):
    c1, c2 = st.columns(2)
    with c1:
        input_data[f"BILL_AMT{i}"] = st.number_input(
            f"Tagihan Bulan ke-{i} (Rp)",
            min_value=0,
            value=int(st.session_state.input_data[f"BILL_AMT{i}"]),
            step=500_000,
            key=f"bill{i}"
        )
    with c2:
        input_data[f"PAY_AMT{i}"] = st.number_input(
            f"Pembayaran Bulan ke-{i} (Rp)",
            min_value=0,
            value=int(st.session_state.input_data[f"PAY_AMT{i}"]),
            step=500_000,
            key=f"pay{i}"
        )

# ======================
# PREDICTION
# ======================
st.header("2️⃣ Hasil Prediksi")

if st.button("🔍 Prediksi Risiko"):
    df_input = pd.DataFrame([input_data])[feature_names]
    prob = model.predict_proba(df_input)[0][1]

    st.metric("Probabilitas Gagal Bayar", f"{prob*100:.2f}%")

    if prob >= 0.5:
        st.error("🔴 Risiko Tinggi Gagal Bayar")
    else:
        st.success("🟢 Risiko Rendah Gagal Bayar")

    # ======================
    # SHAP
    # ======================
    st.header("3️⃣ Penjelasan Model (SHAP)")

    if SHAP_OK:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)

        if isinstance(shap_values, list):
            shap_single = shap_values[1][0]
        else:
            shap_single = shap_values[0]

        shap_df = pd.DataFrame({
            "Fitur": feature_names,
            "SHAP": shap_single
        }).sort_values("SHAP")

        naik = shap_df[shap_df["SHAP"] > 0].tail(5)
        turun = shap_df[shap_df["SHAP"] < 0].head(5)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].barh(naik["Fitur"], naik["SHAP"])
        ax[0].set_title("🔴 Faktor Peningkat Risiko")

        ax[1].barh(turun["Fitur"], turun["SHAP"])
        ax[1].set_title("🟢 Faktor Penurun Risiko")

        st.pyplot(fig)

        st.subheader("🧠 Interpretasi Otomatis")
        for f in naik["Fitur"]:
            st.markdown(f"- **{f}** meningkatkan risiko gagal bayar.")
        for f in turun["Fitur"]:
            st.markdown(f"- **{f}** menurunkan risiko gagal bayar.")
