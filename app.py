import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Credit Risk Model", layout="centered")

st.title("Credit Risk Prediction – Grid Search AUC")

# =========================
# Upload Dataset
# =========================
uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Preview dataset")
    st.dataframe(df.head())

    # pilih kolom target
    target_col = st.selectbox("Pilih kolom target (label)", df.columns)

    # fitur = semua kolom numerik selain target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # hanya numerik agar simple
    X = X.select_dtypes(include="number")

    st.write("Total fitur numerik:", X.shape[1])

    if st.button("Train Model + GridSearch"):
        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # pipeline scaler + model
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42))
        ])

        # parameter grid (aman untuk skripsi)
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5]
        }

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        # best model
        best_model = grid.best_estimator_

        # pred proba untuk AUC
        y_prob = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        st.success(f"AUC terbaik: {auc:.4f}")
        st.write("Best params:", grid.best_params_)

        st.subheader("Prediksi 1 Nasabah")

        # buat form input untuk tiap fitur
        input_data = {}
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(X[col].mean()))
            input_data[col] = val

        if st.button("Prediksi Risiko"):
            single_df = pd.DataFrame([input_data])
            prob = best_model.predict_proba(single_df)[0][1]

            st.write(f"Probabilitas gagal bayar: {prob:.3f}")

            if prob >= 0.5:
                st.error("Berisiko Tinggi")
            else:
                st.success("Risiko Rendah")
