import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Predictive Hospital Resource Intelligence",
    layout="wide"
)

st.title("🏥 Predictive Hospital Resource & Emergency Load System")
st.markdown("""
This system predicts **daily emergency admissions**, estimates **ICU demand**,  
and suggests **staff allocation** to support proactive hospital planning.
""")

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("emergency_load_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------------------------------
# Select Input Mode
# --------------------------------------------------
st.header("🧾 Choose Input Method")
mode = st.radio("Select how you want to provide data:", ["Manual Input", "Upload CSV"])

# Model feature list (same as training)
MODEL_FEATURES = [
    "DayOfWeek",
    "Month",
    "Is_Weekend",
    "Recent_Emergencies_24h",
    "Bed_Occupancy_Rate",
    "ICU_Occupancy_Rate"
]

# --------------------------------------------------
# MANUAL INPUT MODE
# --------------------------------------------------
if mode == "Manual Input":
    st.subheader("✍️ Enter Daily Hospital Details")

    admission_date = st.date_input("Admission Date")
    recent_emergencies = st.number_input("Recent Emergencies (last 24h)", min_value=0, value=50)
    bed_occupancy = st.slider("Bed Occupancy Rate", 0.0, 1.0, 0.6)
    icu_occupancy = st.slider("ICU Occupancy Rate", 0.0, 1.0, 0.7)

    # Feature engineering (same as training)
    admission_date = pd.to_datetime(admission_date)
    day_of_week = admission_date.dayofweek
    month = admission_date.month
    is_weekend = 1 if day_of_week in [5, 6] else 0

    input_df = pd.DataFrame([[
        day_of_week,
        month,
        is_weekend,
        recent_emergencies,
        bed_occupancy,
        icu_occupancy
    ]], columns=MODEL_FEATURES)

    if st.button("🔮 Predict"):
        emergency_pred = int(round(model.predict(input_df)[0]))

        # ICU demand estimation
        ICU_RATIO = 0.75
        icu_demand = int(round(emergency_pred * ICU_RATIO))

        # Staff workload logic
        if emergency_pred >5 or icu_demand >3:
            staff_level = "HIGH"
            staff_rec = "Add 2 nurses + 1 emergency doctor per shift"
        elif emergency_pred > 3 or icu_demand > 2:
            staff_level = "MEDIUM"
            staff_rec = "Add 1 nurse per shift"
        else:
            staff_level = "LOW"
            staff_rec = "No additional staff required"

        st.success("### ✅ Prediction Results")
        st.metric("🚑 Predicted Emergency Admissions (Daily)", emergency_pred)
        st.metric("🏥 Predicted ICU Beds Needed", icu_demand)
        st.metric("👩‍⚕️ Staff Workload Level", staff_level)
        st.info(f"**Staff Recommendation:** {staff_rec}")

# --------------------------------------------------
# CSV UPLOAD MODE
# --------------------------------------------------
if mode == "Upload CSV":
    st.subheader("📂 Upload Daily Hospital CSV")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        REQUIRED_COLUMNS = [
            "Admission_Date",
            "Recent_Emergencies_24h",
            "Bed_Occupancy_Rate",
            "ICU_Occupancy_Rate"
        ]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # Feature engineering (same as training)
        df["Admission_Date"] = pd.to_datetime(df["Admission_Date"])
        df["DayOfWeek"] = df["Admission_Date"].dt.dayofweek
        df["Month"] = df["Admission_Date"].dt.month
        df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

        X = df[MODEL_FEATURES]

        # Predictions
        emergency_pred = model.predict(X)
        df["Predicted_Emergency_Admissions"] = emergency_pred.round(0).astype(int)
        import math

        ICU_RATIO = 1
        icu_demand = math.ceil(emergency_pred * ICU_RATIO)


                # -----------------------------
        # Dynamic Staff Workload Logic
        # -----------------------------
        low_thresh = df["Predicted_Emergency_Admissions"].quantile(0.50)
        high_thresh = df["Predicted_Emergency_Admissions"].quantile(0.80)

        def staff_workload_dynamic(emergency):
            if emergency >= high_thresh:
                return "HIGH"
            elif emergency >= low_thresh:
                return "MEDIUM"
            else:
                return "LOW"

        df["Predicted_Staff_Workload"] = df[
            "Predicted_Emergency_Admissions"
        ].apply(staff_workload_dynamic)

        st.success("### ✅ Prediction Results (Daily)")
        st.dataframe(
            df[
                [
                    "Admission_Date",
                    "Predicted_Emergency_Admissions",
                    "Predicted_ICU_Demand",
                    "Predicted_Staff_Workload"
                ]
            ].head(10)
        )

        st.subheader("📈 Daily Trends")
        st.line_chart(df["Predicted_Emergency_Admissions"])
        st.line_chart(df["Predicted_ICU_Demand"])

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
---
🔹 *This system is a decision-support tool for hospital preparedness and  
does not replace clinical judgment or medical diagnosis.*
""")
