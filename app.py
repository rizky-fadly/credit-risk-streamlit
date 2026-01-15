import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Default Prediction")

st.title("Credit Card Default Prediction")
st.write("Aplikasi prediksi risiko gagal bayar kartu kredit menggunakan XGBoost")

# load model & feature list
model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.subheader("Input Data Nasabah")

input_data = {}

for col in feature_names:
    input_data[col] = st.number_input(col, value=0.0)

if st.button("Prediksi"):
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_names]

    prob = model.predict_proba(df_input)[0][1]

    st.write(f"**Probabilitas Default:** {prob:.3f}")

    if prob >= 0.5:
        st.error("⚠️ Berisiko Default")
    else:
        st.success("✅ Tidak Berisiko Default")
