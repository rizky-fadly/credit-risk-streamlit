import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Risiko Kartu Kredit")

st.title("Prediksi Risiko Gagal Bayar Kartu Kredit")

# load model & fitur
model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# BAGIAN 1: DATA NASABAH
# =========================
st.header("📋 Data Nasabah")

input_data = {}

# --- INPUT KATEGORIK (lebih manusiawi) ---
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
education_label = st.selectbox("Pendidikan Terakhir", list(edu_map.keys()))
marriage_label = st.selectbox("Status Pernikahan", list(marriage_map.keys()))
age_value = st.text_input("Usia (tahun)", value="30")

input_data["SEX"] = sex_map[sex_label]
input_data["EDUCATION"] = edu_map[education_label]
input_data["MARRIAGE"] = marriage_map[marriage_label]
input_data["AGE"] = int(age_value) if age_value.isdigit() else 30

st.divider()

# --- INPUT NUMERIK LAIN ---
st.subheader("Riwayat Pembayaran & Tagihan")

label_map = {
    "PAY_0": "Keterlambatan pembayaran bulan terakhir",
    "PAY_2": "Keterlambatan 2 bulan lalu",
    "PAY_3": "Keterlambatan 3 bulan lalu",
    "PAY_4": "Keterlambatan 4 bulan lalu",
    "PAY_5": "Keterlambatan 5 bulan lalu",
    "PAY_6": "Keterlambatan 6 bulan lalu",

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

for col in feature_names:
    if col not in input_data:
        label = label_map.get(col, col)
        input_data[col] = st.number_input(label, value=0.0)

st.caption("Catatan: keterlambatan → 0 = tepat waktu, 1 = telat 1 bulan, dst.")

# =========================
# BAGIAN 2: HASIL
# =========================
st.header("📊 Hasil Prediksi")

if st.button("Prediksi Risiko"):
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_names]

    prob = model.predict_proba(df_input)[0][1]

    st.metric("Probabilitas Gagal Bayar", f"{prob:.2f}")

    if prob >= 0.5:
        st.error("⚠️ Risiko Tinggi Gagal Bayar")
    else:
        st.success("✅ Risiko Rendah Gagal Bayar")
