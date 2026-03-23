import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import shap
import os
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF

# -------------------
# PAGE SETUP
# -------------------
st.set_page_config(page_title="Smart Diabetes Prediction", layout="centered")

# --- CUSTOM UI STYLING (Added for the visual cards) ---
st.markdown("""
<style>
    .risk-factors-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #6c757d;
        height: 100%;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e1e4e8;
        background-color: #ffffff;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);
    }
    .status-header {
        margin-bottom: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Smart Healthcare System")
st.subheader("Early Diabetes Risk Prediction using Machine Learning")

# -------------------
# LOAD MODEL
# -------------------

@st.cache_resource
def load_models():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
        scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

        model = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))
        
        return model, scaler

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ✅ CALL FUNCTION HERE (OUTSIDE)
model, scaler = load_models()

# -------------------
# DATABASE
# -------------------
@st.cache_resource
def init_db():
    conn = sqlite3.connect("patients.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pregnancies INT,
    glucose REAL,
    bp REAL,
    skin REAL,
    insulin REAL,
    bmi REAL,
    dpf REAL,
    age INT,
    prediction INT,
    probability REAL,
    risk TEXT,
    timestamp TEXT
    )
    """)
    conn.commit()
    return conn

conn = init_db()
c = conn.cursor()

# -------------------
# ROLE
# -------------------
role = st.sidebar.selectbox("Login as", ["Patient", "Doctor"])

# ==========================================================
# PATIENT SIDE
# ==========================================================
if role == "Patient":

    st.header("👤 Patient Self-Assessment")

    age = st.number_input("Age", 1, 120, 30)
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    height = st.number_input("Height (cm)", 120.0, 220.0, 170.0)

    activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    thirst = st.selectbox("Excessive Thirst", ["No", "Yes"])
    urination = st.selectbox("Frequent Urination", ["No", "Yes"])
    weight_loss = st.selectbox("Sudden Weight Loss", ["No", "Yes"])
    family = st.selectbox("Family History", ["No", "Yes"])
    weakness = st.selectbox("Weakness", ["No", "Yes"])

    if st.button("Check Preliminary Risk"):

        bmi = weight / ((height / 100) ** 2)
        
        risk_score = sum([
            thirst == "Yes",
            urination == "Yes",
            weight_loss == "Yes",
            weakness == "Yes",
            family == "Yes",
            activity == "Low",
            bmi > 30
        ])

        if risk_score <= 2:
            risk = "Low"
            risk_color = "#28a745"
        elif risk_score <= 4:
            risk = "Medium"
            risk_color = "#ffc107"
        else:
            risk = "High"
            risk_color = "#dc3545"

        # --- REFINED PATIENT DISPLAY (MIRRORING IMAGE) ---
        st.subheader("Assessment Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="risk-factors-box">
                <b>Key Risk Factors</b><br><br>
                • Age: {age}<br>
                • BMI: {round(bmi, 2)}<br>
                • Family History: {family}<br>
                • Symptoms: {risk_score} detected
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="status-header" style="color: {risk_color}; font-size: 24px;">
                    ✅ {risk} Risk of Diabetes
                </div>
                <p>Based on your self-assessment, you have a <b>{risk}</b> risk level.</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(risk_score / 7)

        st.subheader("🧠 Clinical Insight")

        if risk == "Low":
            st.info("Maintain your healthy lifestyle.")
        elif risk == "Medium":
            st.warning("Monitor your health and improve diet/exercise.")
        else:
            st.error("Seek medical advice immediately.")

        result_data = pd.DataFrame([{
            "Age": age,
            "BMI": round(bmi, 2),
            "Risk": risk,
            "Timestamp": datetime.now()
        }])

        csv = result_data.to_csv(index=False)
        st.download_button("Download Result (CSV)", csv, "patient_result.csv")

        def generate_pdf(data):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, "Patient Risk Report", ln=True)

            for col in data.columns:
                pdf.cell(200, 10, f"{col}: {data[col].iloc[0]}", ln=True)

            return pdf.output(dest='S').encode('latin-1')

        pdf_data = generate_pdf(result_data)
        st.download_button("Download Result (PDF)", pdf_data, "patient_result.pdf")

# ==========================================================
# DOCTOR SIDE
# ==========================================================
elif role == "Doctor":

    st.header("🩺 Clinical Diabetes Prediction")

    inputs = [
        st.number_input("Pregnancies", 0, 20, 1),
        st.number_input("Glucose", 0, 300, 120),
        st.number_input("Blood Pressure", 0, 200, 70),
        st.number_input("Skin Thickness", 0, 100, 20),
        st.number_input("Insulin", 0, 900, 80),
        st.number_input("BMI", 0.0, 70.0, 25.0),
        st.number_input("DPF", 0.0, 3.0, 0.5),
        st.number_input("Age", 1, 120, 30)
    ]

    if st.button("Predict Diabetes Risk"):

        try:
            data = np.array([inputs])
            scaled = scaler.transform(data)

            prediction = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1]

            risk = "Low" if prob < 0.33 else "Medium" if prob < 0.66 else "High"
            risk_color = "#28a745" if risk == "Low" else "#ffc107" if risk == "Medium" else "#dc3545"

            c.execute("""
            INSERT INTO patients VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (*inputs, prediction, prob, risk, datetime.now()))
            conn.commit()

            # --- REFINED DOCTOR DISPLAY (MIRRORING IMAGE) ---
            st.subheader("Prediction Result")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="risk-factors-box">
                    <b>Key Risk Factors</b><br><br>
                    • Age: {inputs[7]}<br>
                    • Glucose: {inputs[1]}<br>
                    • BP: {inputs[2]}<br>
                    • BMI: {inputs[5]}
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                status_text = "Negative" if prediction == 0 else "Positive"
                st.markdown(f"""
                <div class="result-card">
                    <div class="status-header" style="color: {risk_color}; font-size: 24px;">
                        {'✅' if prediction == 0 else '⚠️'} {risk} Risk - Diabetes {status_text}
                    </div>
                    <p style="margin-bottom:0;">Risk Probability: <b>{prob*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob))

            # Clinical Insight
            st.subheader("🧠 Clinical Insight")
            if risk == "Low":
                st.info("Patient is stable.")
            elif risk == "Medium":
                st.warning("Patient needs monitoring.")
            else:
                st.error("Immediate medical attention required.")

            # Export
            result_data = pd.DataFrame([{
                "Prediction": prediction,
                "Probability": prob,
                "Risk": risk,
                "Timestamp": datetime.now()
            }])

            st.subheader("⬇ Export Result")

            csv = result_data.to_csv(index=False)
            st.download_button("Download CSV", csv, "doctor_result.csv")

            def generate_pdf(data):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, "Doctor Prediction Report", ln=True)

                for col in data.columns:
                    pdf.cell(200, 10, f"{col}: {data[col].iloc[0]}", ln=True)

                return pdf.output(dest='S').encode('latin-1')

            pdf_data = generate_pdf(result_data)
            st.download_button("Download PDF", pdf_data, "doctor_report.pdf")

            # -------------------
            #  SHAP
            # -------------------
            st.subheader("🔍 Explainable AI (SHAP)")

            explainer = shap.Explainer(model)
            shap_values = explainer(scaled)

            vals = shap_values.values
            base = shap_values.base_values

            if len(vals.shape) == 3:
                class_index = 1 if vals.shape[2] > 1 else 0
                vals = vals[0, :, class_index]
                base = base[0, class_index] if len(base.shape) > 1 else base[0]
            else:
                vals = vals[0]
                base = base[0]

            fig, ax = plt.subplots()
            shap.plots.waterfall(
                shap.Explanation(
                    values=vals,
                    base_values=base,
                    data=scaled[0]
                ),
                show=False
            )
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Records
    st.subheader("📋 Patient Records")
    history = pd.read_sql("SELECT * FROM patients", conn)

    if len(history) > 0:
        st.dataframe(history)
        csv = history.to_csv(index=False)
        st.download_button("Download CSV", csv, "patient_records.csv")
    else:
        st.warning("No patient records found yet.")

# --- FOOTER DISCLAIMER ---
st.markdown("---")
st.caption("Disclaimer: This tool provides a risk assessment and is not a substitute for professional medical advice.")