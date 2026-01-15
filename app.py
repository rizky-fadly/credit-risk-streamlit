import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================
# SAFE SHAP
# ======================
try:
    import shap
    SHAP_OK = True
except:
    SHAP_OK = False

st.set_page_config(page_title="Prediksi Risiko Kredit", layout="wide")
st.title("📊 Prediksi Risiko Gagal Bayar Kredit")

model = joblib.load("model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ======================
# DEMO DATA (FIX UTAMA)
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
    LOW_RISK[f"BILL_AMT{i}"] = 5_000_000 + i * 500_000
    LOW_RISK[f"PAY_AMT{i}"] = 5_000_000 + i * 500_000

    HIGH_RISK[f"BILL_AMT{i}"] = 18_000_000 - i * 1_000_000
    HIGH_RISK[f"PAY_AMT{i}"] = 500_000

# ======================
# STATE
# ======================
if "demo" not in st.session_state:
    st.session_state.demo = "low"

# ======================
# UI BUTTON
# ======================
c1, c2 = st.columns(2)
with c1:
    if st.button("🟢 Contoh Risiko Rendah"):
        st.session_state.demo = "low"
with c2:
    if st.button("🔴 Contoh Risiko Tinggi"):
        st.session_state.demo = "high"

# ======================
# BUILD INPUT (FIX)
# ======================
if st.session_state.demo == "low":
    input_data = LOW_RISK.copy()
else:
    input_data = HIGH_RISK.copy()

df_input = pd.DataFrame([input_data])[feature_names]

# ======================
# PREDICTION
# ======================
prob = model.predict_proba(df_input)[0][1]

st.subheader("🔍 Hasil Prediksi")
st.metric("Probabilitas Gagal Bayar", f"{prob*100:.2f}%")

if prob >= 0.5:
    st.error("🔴 Risiko Tinggi Gagal Bayar")
else:
    st.success("🟢 Risiko Rendah Gagal Bayar")

# ======================
# SHAP
# ======================
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

    st.subheader("🧠 Penjelasan Otomatis")
    for f in naik["Fitur"]:
        st.markdown(f"- **{f}** meningkatkan risiko gagal bayar.")
    for f in turun["Fitur"]:
        st.markdown(f"- **{f}** menurunkan risiko gagal bayar.")
