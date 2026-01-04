# 🏥 Predictive Hospital Resource & Emergency Load Intelligence System

## PS 06 – Hackathon Problem Statement

### 🚨 Problem Overview

Hospitals often face **unexpected surges in emergency cases**, leading to:

* Sudden emergency influx
* ICU bed shortages
* Staff burnout
* Poor preparedness during outbreaks or peak seasons

Most hospitals react **after overload occurs**, rather than planning ahead.

---

## 🎯 Proposed Solution

We built an **AI-powered predictive intelligence system** that helps hospitals proactively plan resources by:

* 📈 Predicting **daily emergency admissions**
* 🏥 Estimating **ICU bed demand**
* 👩‍⚕️ Forecasting **staff workload surges**
* 📊 Supporting **staff allocation, bed management & emergency preparedness**

The system leverages:

* Historical hospital admission data
* Bed & ICU occupancy rates
* Temporal features (day, month, weekends)
* Seasonal and trend-based patterns

---

## 🧠 How It Works

1. Historical data is used to train a machine learning model
2. Features such as recent emergencies, occupancy rates, and calendar data are processed
3. The trained model predicts emergency load
4. ICU demand and staff workload are derived using intelligent rules
5. Results are presented through an interactive **Streamlit dashboard**

---

## 🖥️ Application Features

* 🔹 Manual input for daily hospital conditions
* 🔹 CSV upload for batch predictions
* 🔹 Real-time emergency admission prediction
* 🔹 ICU demand estimation
* 🔹 Staff workload categorization (LOW / MEDIUM / HIGH)
* 🔹 Visual trend analysis

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Jupyter Notebook

---

## 📂 Project Structure

```
GFGBQ-Team-outlier/
│── app.py                    # Streamlit application
│── emergency_load_model.pkl  # Trained ML model
│── dataset.csv               # Historical hospital dataset
│── task.ipynb                # Model training & experimentation
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
```

---

## 🌐 Live Streamlit Application

🚀 **Live Demo:** http://localhost:8501/#ai-driven-predictive-healthcare-analytics 

The application is deployed on **Streamlit Community Cloud** and provides an interactive interface to:

* Predict daily emergency admissions
* Estimate ICU bed demand
* Analyze staff workload levels

Judges and users can test predictions using manual input or CSV upload without any local setup.



---

## 👥 Hackathon Team Members

* **Kanika AB**
* **Dhinesh S**
* **Sreevibu S**
* **Selvaganapathy K**

---

## 🏁 Conclusion

This project demonstrates how predictive analytics and machine learning can transform hospital operations from **reactive to proactive**, reducing overload risks and improving patient care readiness.

---

⚠️ *Disclaimer: This system is a decision-support tool and does not replace medical judgment.*
