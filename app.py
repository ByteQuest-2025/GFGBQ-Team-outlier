import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Hospital Intelligence Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

st.markdown("""
<style>
:root {
    --bg-main: #0b1020;
    --bg-panel: #121833;
    --bg-card: #161d3f;
    --border-soft: rgba(255,255,255,0.08);

    --accent-cyan: #22d3ee;
    --accent-violet: #a78bfa;
    --accent-green: #34d399;
    --accent-yellow: #fbbf24;
    --accent-red: #fb7185;

    --text-main: #e5e7eb;
    --text-muted: #9ca3af;
}

/* APP BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #111827, #020617);
    color: var(--text-main);
}

/* HEADER */
.main-header {
    background: linear-gradient(135deg, #111827, #1e293b);
    border: 1px solid var(--border-soft);
    border-left: 5px solid var(--accent-cyan);
    border-radius: 18px;
    padding: 2.8rem;
    text-align: center;
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
}

.header-glow {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
    padding: 0.8rem 2.2rem;
    border-radius: 999px;
    display: inline-block;
    color: #020617;
    font-weight: 800;
    margin-top: 1rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #020617);
    border-right: 1px solid var(--border-soft);
}

/* CARDS */
.content-card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1rem;
    box-shadow: 0 12px 30px rgba(0,0,0,0.45);
}

/* METRICS */
.metric-card {
    background: linear-gradient(145deg, #161d3f, #0f172a);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    transition: all 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 25px 50px rgba(34,211,238,0.15);
}
.big-number {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ALERTS */
.alert-high {
    background: linear-gradient(90deg, #fb7185, #be123c);
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    font-weight: 800;
}
.alert-moderate {
    background: linear-gradient(90deg, #fbbf24, #d97706);
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    font-weight: 800;
}
.alert-low {
    background: linear-gradient(90deg, #34d399, #059669);
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    font-weight: 800;
}

/* TABS */
div[data-testid="stTabs"] button {
    background: transparent;
    color: var(--text-muted);
    font-weight: 600;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-cyan);
    border-bottom: 3px solid var(--accent-cyan);
}

/* FOOTER */
.footer {
    background: linear-gradient(135deg, #111827, #020617);
    border: 1px solid var(--border-soft);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-top: 3rem;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    icu_model = pickle.load(open("icu_model.pkl","rb"))
    emergency_model = pickle.load(open("emergency_model.pkl","rb"))
    staff_model = pickle.load(open("staff_model.pkl","rb"))
    try:
        staff_label = pickle.load(open("staff_label_encoder.pkl","rb"))
    except:
        staff_label = None
    return icu_model, emergency_model, staff_model, staff_label

icu_model, emergency_model, staff_model, staff_label = load_models()

st.markdown("""
<div class='main-header'>
    <h1 style='font-size: 3.2rem;'>🏥 Hospital Intelligence Platform</h1>
    <div class='header-glow'>
        <h2>AI-Driven Predictive Healthcare Analytics</h2>
    </div>
    <p>Real-time ICU Risk • Emergency Forecasting • Staff Optimization</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🎛 Control Center")

    age = st.slider("Patient Age",1,100,40)
    cost = st.number_input("Treatment Cost ₹",1000,500000,25000)

    bed_load = st.slider("Bed Occupancy %",0.0,1.0,0.75,0.01)
    icu_load = st.slider("ICU Occupancy %",0.0,1.0,0.60,0.01)

    staff_score = st.slider("Staff Workload %",0.0,100.0,45.0)
    recent_emg = st.slider("24H Emergencies",0,200,30)

    hour = st.slider("Current Hour",0,23,10)
    dow = st.slider("Day (0=Mon)",0,6,2)

    if st.button("🚀 RUN PREDICTIONS", use_container_width=True):
        st.session_state.run_prediction = True

input_features = pd.DataFrame([[age,cost,bed_load,icu_load,
staff_score,hour,dow,1,recent_emg]],columns=[
"Age","Cost","Bed_Occupancy_Rate","ICU_Occupancy_Rate",
"Staff_Workload_Score","Hour","DayOfWeek",
"Emergency_Flag","Recent_Emergencies_24h"])

tab1, tab2, tab3 = st.tabs([
    "🏥 ICU Risk Intelligence",
    "🚨 Emergency Forecast",
    "👥 Staff Optimization"
])

with tab1:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.header("🎯 ICU Admission Risk Analysis")

    if not st.session_state.run_prediction:
        st.info("Click **RUN PREDICTIONS** to analyze ICU Risk")
    else:
        icu_prob = icu_model.predict_proba(input_features)[0][1]
        risk_level = "HIGH" if icu_prob >= 0.7 else "MODERATE" if icu_prob >= 0.4 else "LOW"

        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🎲 ICU Risk Probability</h3>
                <div class='big-number'>{icu_prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🚦 Risk Level</h3>
                <div class='big-number'>{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>⚡ Emergency Influence</h3>
                <div class='big-number' style="color:#34d399">{recent_emg}</div>
            </div>
            """, unsafe_allow_html=True)

        if risk_level=="HIGH":
            st.markdown("<div class='alert-high'>🚨 CRITICAL ICU RISK - PREPARE NOW</div>", unsafe_allow_html=True)
        elif risk_level=="MODERATE":
            st.markdown("<div class='alert-moderate'>⚠ MODERATE RISK - MONITOR CLOSELY</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-low'>✅ LOW RISK - SYSTEM STABLE</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.header("📈 Emergency Load Prediction")

    if not st.session_state.run_prediction:
        st.info("Click **RUN PREDICTIONS** to forecast emergency admissions")
    else:
        emergency_pred = int(emergency_model.predict(input_features)[0])
        st.metric("Predicted Next Cycle Admissions", emergency_pred, delta=recent_emg)

        hours = np.arange(12)
        trend = np.maximum(20, emergency_pred + np.random.normal(0,15,12).cumsum())
        df_trend = pd.DataFrame({'Hour': hours, 'Predicted Emergencies': trend})

        st.line_chart(df_trend.set_index('Hour'), use_container_width=True)
        st.success("AI Emergency Prediction Active")

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.header("👥 Staff Workload Intelligence")

    if not st.session_state.run_prediction:
        st.info("Click **RUN PREDICTIONS** to analyze staff workload")
    else:
        staff_pred = staff_model.predict(input_features)[0]
        staff_level = staff_label.inverse_transform([staff_pred])[0] if staff_label else str(staff_pred)

        c1,c2 = st.columns(2)
        c1.metric("Current Workload Score", f"{staff_score:.0f}/100")
        c2.metric("Predicted Level", staff_level)

        if "high" in staff_level.lower():
            st.markdown("<div class='alert-high'>🚨 HIGH BURNOUT RISK</div>", unsafe_allow_html=True)
        elif "moderate" in staff_level.lower():
            st.markdown("<div class='alert-moderate'>⚠ MODERATE LOAD</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-low'>✅ OPTIMAL STAFFING</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

