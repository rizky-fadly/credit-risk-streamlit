import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Default Prediction")

st.title("Credit Card Default Prediction")
st.write("Prediksi risiko gagal bayar kartu kredit")

# load model
model = joblib.load("xgb_credit_default_model.pkl")

st.subheader("Input Data Nasabah")

input_data = {
    "LIMIT_BAL": st.number_input("Limit Kredit", value=50000),
    "AGE": st.number_input("Usia", value=30),
    "PAY_0": st.number_input("Status Telat Bulan Terakhir", value=0),
    "BILL_AMT1": st.number_input("Tagihan Bulan Terakhir", value=0),
    "PAY_AMT1": st.number_input("Pembayaran Bulan Terakhir", value=0),
}

if st.button("Prediksi"):
    df_input = pd.DataFrame([input_data])
    prob = model.predict_proba(df_input)[0][1]

    st.write(f"Probabilitas Default: **{prob:.2f}**")

    if prob >= 0.5:
        st.error("⚠️ Berisiko Default")
    else:
        st.success("✅ Tidak Berisiko Default")
