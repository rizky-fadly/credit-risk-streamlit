import streamlit as st
import pandas as pd
import joblib

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Risiko Kartu Kredit",
    layout="centered"
)

st.title("Prediksi Risiko Gagal Bayar Kartu Kredit")
st.write(
    "Aplikasi ini memprediksi risiko gagal bayar kartu kredit "
    "menggunakan model **XGBoost**."
)

# =========================
# LOAD MODEL & FITUR
# =========================
model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# DEMO DATA (1 KLIK)
# =========================
demo_data = {
    "LIMIT_BAL": 50_000_000,
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

    "BILL_AMT1": 8_000_000,
    "BILL_AMT2": 7_500_000,
    "BILL_AMT3": 7_000_000,
    "BILL_AMT4": 6_800_000,
    "BILL_AMT5": 6_500_000,
    "BILL_AMT6": 6_000_000,

    "PAY_AMT1": 3_000_000,
    "PAY_AMT2": 3_000_000,
    "PAY_AMT3": 3_000_000,
    "PAY_AMT4": 3_000_000,
    "PAY_AMT5": 3_000_000,
    "PAY_AMT6": 3_000_000,
}

if "demo" not in st.session_state:
    st.session_state["demo"] = False

if st.button("🎯 Isi Contoh Data (Demo)"):
    st.session_state["demo"] = True

use_demo = st.session_state["demo"]

# =========================
# BAGIAN 1: DATA NASABAH
# =========================
st.header("📋 Data Nasabah")

input_data = {}

# --- Mapping kategori ---
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

# --- Input kategori ---
sex_label = st.selectbox("Jenis Kelamin", list(sex_map.keys()))
education_label = st.selectbox("Pendidikan Terakhir", list(edu_map.keys()))
marriage_label = st.selectbox("Status Pernikahan", list(marriage_map.keys()))
age_value = st.text_input(
    "Usia (tahun)",
    value=str(demo_data["AGE"]) if use_demo else "30"
)

input_data["SEX"] = sex_map[sex_label]
input_data["EDUCATION"] = edu_map[education_label]
input_data["MARRIAGE"] = marriage_map[marriage_label]
input_data["AGE"] = int(age_value) if age_value.isdigit() else 30

st.divider()

# --- Label ramah pengguna ---
label_map = {
    "LIMIT_BAL": "Limit Kredit Kartu (Rp)",

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

st.subheader("Riwayat Kredit")

for col in feature_names:
    if col not in input_data:
        label = label_map.get(col, col)
        default_val = demo_data.get(col, 0) if use_demo else 0

        # Jika variabel nominal uang
        if col.startswith("BILL_AMT") or col.startswith("PAY_AMT") or col == "LIMIT_BAL":
            input_data[col] = st.number_input(
                label,
                value=float(default_val),
                format="%,.0f"
            )
        else:
            # Variabel status keterlambatan
            input_data[col] = st.number_input(
                label,
                value=int(default_val),
                step=1
            )

st.caption("Keterlambatan: 0 = tepat waktu, 1 = telat 1 bulan, dst.")

# =========================
# BAGIAN 2: HASIL
# =========================
st.header("📊 Hasil Prediksi")

if st.button("Prediksi Risiko"):
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_names]

    prob = model.predict_proba(df_input)[0][1]

    st.metric(
        "Probabilitas Gagal Bayar",
        f"{prob * 100:.2f} %"
    )

    st.write(
        f"**Limit Kredit:** Rp {input_data['LIMIT_BAL']:,.0f}"
    )

    if prob >= 0.5:
        st.error("⚠️ Risiko Tinggi Gagal Bayar")
    else:
        st.success("✅ Risiko Rendah Gagal Bayar")

