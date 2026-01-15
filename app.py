import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# OPTIONAL SHAP (AMAN)
# ======================
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Prediksi Risiko Gagal Bayar",
    layout="wide"
)

st.title("📊 Prediksi Risiko Gagal Bayar Kredit")
st.caption(
    "Aplikasi ini memprediksi kemungkinan gagal bayar nasabah "
    "menggunakan model XGBoost dan Explainable AI (SHAP)."
)

# ======================
# LOAD MODEL & FEATURE
# ======================
model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ======================
# LABEL AWAM
# ======================
label_map = {
    "LIMIT_BAL": "Limit Kredit",
    "SEX": "Jenis Kelamin",
    "EDUCATION": "Pendidikan",
    "MARRIAGE": "Status Pernikahan",
    "AGE": "Usia",
    "PAY_0": "Status Pembayaran Terakhir",
    "PAY_2": "Pembayaran 2 Bulan Lalu",
    "PAY_3": "Pembayaran 3 Bulan Lalu",
    "PAY_4": "Pembayaran 4 Bulan Lalu",
    "PAY_5": "Pembayaran 5 Bulan Lalu",
    "PAY_6": "Pembayaran 6 Bulan Lalu",
    "BILL_AMT1": "Tagihan Bulan Terakhir",
    "BILL_AMT2": "Tagihan 2 Bulan Lalu",
    "BILL_AMT3": "Tagihan 3 Bulan Lalu",
    "BILL_AMT4": "Tagihan 4 Bulan Lalu",
    "BILL_AMT5": "Tagihan 5 Bulan Lalu",
    "BILL_AMT6": "Tagihan 6 Bulan Lalu",
    "PAY_AMT1": "Pembayaran Bulan Terakhir",
    "PAY_AMT2": "Pembayaran 2 Bulan Lalu",
    "PAY_AMT3": "Pembayaran 3 Bulan Lalu",
    "PAY_AMT4": "Pembayaran 4 Bulan Lalu",
    "PAY_AMT5": "Pembayaran 5 Bulan Lalu",
    "PAY_AMT6": "Pembayaran 6 Bulan Lalu",
}

# ======================
# SESSION STATE
# ======================
if "demo" not in st.session_state:
    st.session_state.demo = "normal"

# ======================
# DEMO PRESET (FIX UTAMA)
# ======================
demo_low = {
    "PAY_0": 0,
    "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 3_000_000, "BILL_AMT2": 3_500_000,
    "BILL_AMT3": 4_000_000, "BILL_AMT4": 4_500_000,
    "BILL_AMT5": 5_000_000, "BILL_AMT6": 5_500_000,
    "PAY_AMT1": 3_000_000, "PAY_AMT2": 3_500_000,
    "PAY_AMT3": 4_000_000, "PAY_AMT4": 4_500_000,
    "PAY_AMT5": 5_000_000, "PAY_AMT6": 5_500_000,
}

demo_high = {
    "PAY_0": 2,
    "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
    "BILL_AMT1": 20_000_000, "BILL_AMT2": 18_000_000,
    "BILL_AMT3": 17_000_000, "BILL_AMT4": 16_000_000,
    "BILL_AMT5": 15_000_000, "BILL_AMT6": 14_000_000,
    "PAY_AMT1": 500_000, "PAY_AMT2": 500_000,
    "PAY_AMT3": 500_000, "PAY_AMT4": 500_000,
    "PAY_AMT5": 500_000, "PAY_AMT6": 500_000,
}

# ======================
# LAYOUT
# ======================
col_input, col_output = st.columns([1, 1.2])

# ======================
# INPUT
# ======================
with col_input:
    st.subheader("📋 Data Nasabah")

    st.markdown("### 🎯 Contoh Cepat")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🟢 Contoh Risiko Rendah"):
            st.session_state.demo = "low"
    with c2:
        if st.button("🔴 Contoh Risiko Tinggi"):
            st.session_state.demo = "high"

    with st.expander("Identitas Dasar", expanded=True):
        sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        education = st.selectbox(
            "Pendidikan Terakhir",
            ["SMA", "Diploma", "Sarjana", "Pascasarjana"]
        )
        marriage = st.selectbox(
            "Status Pernikahan",
            ["Belum Menikah", "Menikah", "Lainnya"]
        )
        age = st.number_input("Usia (tahun)", 18, 100, 30)

    with st.expander("Limit Kredit", expanded=True):
        if st.session_state.demo == "low":
            limit_bal = 100_000_000
        elif st.session_state.demo == "high":
            limit_bal = 15_000_000
        else:
            limit_bal = 50_000_000

        limit_bal = st.number_input(
            "Limit Kredit",
            value=int(limit_bal),
            key="limit_bal"
        )
        st.caption(f"💰 Pratinjau: Rp {limit_bal:,.0f}")

    with st.expander("Riwayat Pembayaran", expanded=True):
        pay_status = st.selectbox(
            "Status Pembayaran Terakhir",
            ["Tepat Waktu", "Terlambat 1 Bulan", "Terlambat >1 Bulan"]
        )

    st.markdown("### Detail Tagihan & Pembayaran")

    input_data = {}

    if st.session_state.demo == "low":
        demo_values = demo_low
    elif st.session_state.demo == "high":
        demo_values = demo_high
    else:
        demo_values = {}

    for col in feature_names:
        if col == "LIMIT_BAL":
            input_data[col] = limit_bal
        elif col == "SEX":
            input_data[col] = 1 if sex == "Laki-laki" else 2
        elif col == "EDUCATION":
            input_data[col] = {
                "SMA": 1, "Diploma": 2, "Sarjana": 3, "Pascasarjana": 4
            }[education]
        elif col == "MARRIAGE":
            input_data[col] = {
                "Belum Menikah": 1, "Menikah": 2, "Lainnya": 3
            }[marriage]
        elif col == "AGE":
            input_data[col] = age
        elif col == "PAY_0":
            input_data[col] = {
                "Tepat Waktu": 0,
                "Terlambat 1 Bulan": 1,
                "Terlambat >1 Bulan": 2
            }[pay_status]
        else:
            default_val = demo_values.get(col, 0)
            input_data[col] = st.number_input(
                label_map.get(col, col),
                value=float(default_val),
                key=f"input_{col}"
            )

# ======================
# PREDICTION
# ======================
df_input = pd.DataFrame([input_data])[feature_names]
prob = model.predict_proba(df_input)[0][1]

# ======================
# OUTPUT
# ======================
with col_output:
    st.subheader("🔍 Hasil Analisis Risiko")

    st.metric(
        "Probabilitas Gagal Bayar",
        f"{prob*100:.2f} %"
    )

    if prob >= 0.5:
        st.error("🔴 Risiko Tinggi Gagal Bayar")
    else:
        st.success("🟢 Risiko Rendah Gagal Bayar")

    st.subheader("📊 Penjelasan Model (Explainable AI)")

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)

        if isinstance(shap_values, list):
            shap_single = shap_values[1][0]
        else:
            shap_single = shap_values[0]

        shap_df = pd.DataFrame({
            "Fitur": feature_names,
            "SHAP": shap_single
        })

        shap_df["Fitur"] = shap_df["Fitur"].map(label_map).fillna(shap_df["Fitur"])

        risk_up = shap_df[shap_df["SHAP"] > 0].sort_values("SHAP", ascending=False).head(5)
        risk_down = shap_df[shap_df["SHAP"] < 0].sort_values("SHAP").head(5)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].barh(risk_up["Fitur"], risk_up["SHAP"])
        axes[0].set_title("🔴 Faktor Peningkat Risiko")
        axes[0].invert_yaxis()

        axes[1].barh(risk_down["Fitur"], risk_down["SHAP"])
        axes[1].set_title("🟢 Faktor Penurun Risiko")
        axes[1].invert_yaxis()

        st.pyplot(fig)

        st.subheader("🧠 Penjelasan Otomatis")
        for _, r in risk_up.iterrows():
            st.markdown(f"- **{r['Fitur']}** meningkatkan risiko gagal bayar.")
        for _, r in risk_down.iterrows():
            st.markdown(f"- **{r['Fitur']}** membantu menurunkan risiko gagal bayar.")

    else:
        st.warning("SHAP tidak tersedia di environment deployment.")

st.caption(
    "Catatan: Hasil prediksi bersifat pendukung keputusan "
    "dan tidak menggantikan analisis kredit oleh pihak bank."
)
