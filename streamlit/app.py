import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")
st.title("🫀 Heart Disease Risk Predictor")

@st.cache_resource
def get_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["Age","Sex","CP","BP","Cholesterol","FBS","EKG","MaxHR","ExAng","ST","Slope","Vessels","Thal","Target"]
    df = pd.read_csv(url, header=None, names=cols, na_values="?").dropna()
    df["Target"] = (df["Target"] > 0).astype(int)

    df["HR_ratio"]       = df["MaxHR"] / (220 - df["Age"])
    df["ST_angina"]      = df["ST"] * df["ExAng"]
    df["vessel_thal"]    = df["Vessels"] * df["Thal"]
    df["metabolic_risk"] = (df["Cholesterol"] > 240).astype(int) + (df["BP"] > 140).astype(int) + df["FBS"]
    df["chol_per_age"]   = df["Cholesterol"] / (df["Age"] + 1)

    FEATURES = ["Age","Sex","CP","BP","Cholesterol","FBS","EKG","MaxHR","ExAng","ST","Slope",
                "HR_ratio","ST_angina","vessel_thal","metabolic_risk","chol_per_age"]

    X, y = df[FEATURES], df["Target"]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.7, random_state=42,
                          eval_metric="logloss", early_stopping_rounds=30)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model, FEATURES

with st.spinner("Loading model…"):
    model, FEATURES = get_model()

st.markdown("---")
st.subheader("Enter Patient Details")

c1, c2 = st.columns(2)

with c1:
    age     = st.number_input("Age", 20, 80, 54)
    bp      = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 130)
    chol    = st.number_input("Cholesterol (mg/dL)", 100, 400, 220)
    max_hr  = st.number_input("Max Heart Rate Achieved", 60, 210, 150)
    st_dep  = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    fbs     = st.selectbox("Fasting Blood Sugar > 120?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with c2:
    sex     = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp      = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                           format_func=lambda x: {1:"Typical Angina",2:"Atypical Angina",3:"Non-anginal",4:"Asymptomatic"}[x])
    ekg     = st.selectbox("Resting EKG Results", [0, 1, 2],
                           format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
    ex_ang  = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    slope   = st.selectbox("Slope of Peak ST Segment", [1, 2, 3],
                           format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x])
    vessels = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3])
    thal    = st.selectbox("Thallium Stress Test", [3, 6, 7],
                           format_func=lambda x: {3:"Normal",6:"Fixed Defect",7:"Reversible Defect"}[x])

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    row = pd.DataFrame([[
        age, sex, cp, bp, chol, fbs, ekg, max_hr, ex_ang, st_dep, slope,
        max_hr / (220 - age),
        st_dep * ex_ang,
        vessels * thal,
        int(chol > 240) + int(bp > 140) + fbs,
        chol / (age + 1)
    ]], columns=FEATURES)

    prob = model.predict_proba(row)[0][1]
    pred = model.predict(row)[0]

    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Risk Probability", f"{prob*100:.1f}%")
    m2.metric("Prediction", "❤️‍🩹 Presence" if pred else "✅ Absence")

    if pred:
        st.error("**High Risk** — This profile suggests a significant likelihood of heart disease. Please consult a cardiologist.")
    else:
        st.success("**Low Risk** — This profile does not strongly indicate heart disease.")

    st.progress(float(prob), text=f"Risk Score: {prob*100:.1f}%")
