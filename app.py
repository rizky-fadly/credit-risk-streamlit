import streamlit as st
import pandas as pd
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

st.set_page_config(page_title="Prediksi Risiko Kredit", layout="centered")
st.title("📊 Prediksi Credit Default Menggunakan XGBoost & SHAP")

# ======================
# LOAD MODEL
# ======================
model = joblib.load("xgb_credit_default_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ======================
# FORMAT RUPIAH
# ======================
def rupiah(x):
    return f"Rp {int(x):,}".replace(",", ".")

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
# DEMO LOADER (CRITICAL FIX)
# ======================
def load_demo(demo):
    for k, v in demo.items():
        st.session_state[k] = v

# ======================
# DEMO BUTTON
# ======================
st.subheader("🔎 Contoh Cepat")

if st.button("🟢 Contoh Risiko Rendah"):
    load_demo(LOW_RISK)

if st.button("🔴 Contoh Risiko Tinggi"):
    load_demo(HIGH_RISK)

# ======================
# INPUT DATA
# ======================
st.header("1️⃣ Data Nasabah")

input_data = {}

input_data["LIMIT_BAL"] = st.number_input(
    "Limit Kredit",
    min_value=0,
    step=1_000_000,
    key="LIMIT_BAL"
)
st.caption(f"💰 {rupiah(input_data['LIMIT_BAL'])}")

input_data["AGE"] = st.number_input(
    "Usia",
    min_value=17,
    max_value=100,
    key="AGE"
)

input_data["SEX"] = st.selectbox(
    "Jenis Kelamin",
    options=[1, 2],
    format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan",
    key="SEX"
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
    key="EDUCATION"
)

input_data["MARRIAGE"] = st.selectbox(
    "Status Pernikahan",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Belum Menikah",
        2: "Menikah",
        3: "Lainnya"
    }[x],
    key="MARRIAGE"
)

# ======================
# PAY STATUS
# ======================
st.subheader("Riwayat Keterlambatan Pembayaran")

pay_labels = {
    "PAY_0": "Keterlambatan Bulan Terakhir",
    "PAY_2": "Keterlambatan 2 Bulan Lalu",
    "PAY_3": "Keterlambatan 3 Bulan Lalu",
    "PAY_4": "Keterlambatan 4 Bulan Lalu",
    "PAY_5": "Keterlambatan 5 Bulan Lalu",
    "PAY_6": "Keterlambatan 6 Bulan Lalu",
}

for p, label in pay_labels.items():
    input_data[p] = st.selectbox(
        label,
        options=[-1, 0, 1, 2, 3],
        format_func=lambda x: {
            -1: "Tidak ada tagihan",
            0: "Lancar",
            1: "Terlambat 1 bulan",
            2: "Terlambat 2 bulan",
            3: "Terlambat ≥3 bulan"
        }[x],
        key=p
    )

# ======================
# BILL & PAYMENT
# ======================
st.subheader("Tagihan & Pembayaran 6 Bulan Terakhir")

for i in range(1, 7):
    bill_key = f"BILL_AMT{i}"
    pay_key = f"PAY_AMT{i}"

    input_data[bill_key] = st.number_input(
        f"Tagihan Bulan ke-{i}",
        min_value=0,
        step=500_000,
        key=bill_key
    )
    st.caption(f"📄 {rupiah(input_data[bill_key])}")

    input_data[pay_key] = st.number_input(
        f"Pembayaran Bulan ke-{i}",
        min_value=0,
        step=500_000,
        key=pay_key
    )
    st.caption(f"💸 {rupiah(input_data[pay_key])}")

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

        shap_single = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

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

