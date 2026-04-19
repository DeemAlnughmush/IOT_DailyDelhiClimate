# 🌦 Climate Forecasting Dashboard

An interactive web-based dashboard for analyzing historical climate data and predicting future values using machine learning models.

---

## 🚀 Live Demo

👉 **Open the app here:**
[https://iotdailydelhiclimate.streamlit.app](https://iotdailydelhiclimate-6zcjwo5d9pqdfyzvamk8xg.streamlit.app/)
> This app is deployed using **Streamlit Community Cloud** and can be accessed from any device (mobile, tablet, or desktop).

---

## 📊 Project Overview

This project uses a real-world climate dataset from Delhi, India, covering the period:

**1 January 2013 → 24 April 2017**

The dashboard allows users to:

* Explore historical climate trends
* Apply interactive filters
* Compare machine learning models
* Predict future climate values

---

## 🤖 Machine Learning Models

We trained and compared:

* Random Forest
* Linear Regression

### Best Models per Target:

| Target      | Best Model           |
| ----------- | -------------------- |
| Temperature | Random Forest ✅      |
| Humidity    | Random Forest ✅      |
| Wind Speed  | Linear Regression ⚠️ |
| Pressure    | Random Forest ❌      |

> ⚠️ Pressure prediction was excluded from the dashboard due to very low performance (R² ≈ 0).

---

## 📈 Features

* 📅 Date range filtering (within dataset limits)
* 📊 Interactive visualizations (Plotly)
* 🔍 Dynamic variable selection
* 📉 Model performance comparison
* 🔮 Future prediction using trained models

---

## 🗂 Project Structure

```
📦 climate-project
 ┣ 📜 app.py
 ┣ 📜 notebook.ipynb
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┣ 📂 data/
 ┃ ┣ climate_dashboard_data.csv
 ┃ ┗ comparison_metrics.csv
 ┣ 📂 models/
 ┃ ┣ temp_model.pkl
 ┃ ┣ humidity_model.pkl
 ┃ ┣ wind_model.pkl
 ┃ ┗ pressure_model.pkl
```

---

## ⚙️ Installation & Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

The dashboard is deployed using:

* **Streamlit Community Cloud**
* Connected directly to this GitHub repository

---

## 📌 Notes

* Dataset is limited to historical records (2013–2017)
* Future predictions are based on learned patterns, not real-time data
* Pressure prediction was excluded due to weak model performance

---

## ✨ Future Improvements

* Add real-time weather API integration
* Improve model performance (especially wind speed)
* Add more interactive analytics

---

## 📄 License

This project is for educational purposes.
