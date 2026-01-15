import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Risiko Kartu Kredit")

st.title("Prediksi Risiko Gagal Bayar Kartu Kredit")
st.write("Isi data nasabah untuk melihat risiko gagal bayar")

model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# mapping nama teknis → nama ramah pengguna
label_map = {
    "PAY_0": "Keterlambatan pembayaran bulan terakhir",
    "PAY_2": "Keterlambatan pembayaran 2 bulan lalu",
    "PAY_3": "Keterlambatan pembayaran 3 bulan lalu",
    "PAY_4": "Keterlambatan pembayaran 4 bulan lalu",
    "PAY_5": "Keterlambatan pembayaran 5 bulan lalu",
    "PAY_6": "Keterlambatan pembayaran 6 bulan lalu",

    "BILL_AMT1": "Tagihan bulan terakhir (Rp)",
    "BILL_AMT2": "Tagihan 2 bulan lalu (Rp)",
    "BILL_AMT3": "Tagihan 3 bulan lalu (Rp)",
    "BILL_AMT4": "Tagihan 4 bulan lalu (Rp)",
    "BILL_AMT5": "Tagihan 5 bulan lalu (Rp)",
    "BILL_AMT6": "Tagihan 6 bulan lalu (Rp)",

    "PAY_AMT1": "Pembayaran bulan terakhir (Rp)",
    "PAY_AMT2": "Pembayaran 2 bulan lalu (Rp)",
    "PAY_AMT3": "Pembayaran 3 bulan lalu (Rp)",
    "PAY_AMT4": "Pembayaran 4 bulan lalu (Rp)",
    "PAY_AMT5": "Pembayaran 5 bulan lalu (Rp)",
    "PAY_AMT6": "Pembayaran 6 bulan lalu (Rp)",
}

st.subheader("Data Nasabah")

input_data = {}
for col in feature_names:
    label = label_map.get(col, col)
    input_data[col] = st.number_input(label, value=0.0)

st.caption("Catatan: nilai keterlambatan → 0 = tepat waktu, 1 = telat 1 bulan, dst.")

if st.button("Prediksi Risiko"):
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_names]

    prob = model.predict_proba(df_input)[0][1]

    st.write(f"**Probabilitas gagal bayar:** {prob:.2f}")

    if prob >= 0.5:
        st.error("⚠️ Risiko Tinggi Gagal Bayar")
    else:
        st.success("✅ Risiko Rendah Gagal Bayar")
