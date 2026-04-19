import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Climate Forecasting Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("climate_dashboard_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv("comparison_metrics.csv")

@st.cache_resource
def load_models():
    return {
        "Temperature": joblib.load("temp_model.pkl"),
        "Humidity": joblib.load("humidity_model.pkl"),
        "Wind Speed": joblib.load("wind_model.pkl"),
        # Pressure model موجود عندك، لكن مستبعد من التوقع النهائي لضعف الأداء
        # "Pressure": joblib.load("pressure_model.pkl"),
    }

df = load_data()
metrics_df = load_metrics()
models = load_models()

# ---------------------------
# Helper mappings
# ---------------------------
column_map = {
    "Temperature": "meantemp",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "Pressure": "meanpressure"
}

description_map = {
    "Temperature": "Average daily air temperature.",
    "Humidity": "Average daily humidity level.",
    "Wind Speed": "Average daily wind speed.",
    "Pressure": "Average daily atmospheric pressure."
}

# ---------------------------
# Title
# ---------------------------
st.title("Climate Forecasting Dashboard")
st.markdown(
    "Interactive dashboard for analyzing Delhi climate trends and predicting future values "
    "using the best-performing machine learning models."
)

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Dashboard Filters")
st.sidebar.markdown(
    "Use these filters to explore the historical climate data."
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    help="Choose the start and end dates to filter the displayed climate records."
)

selected_variable = st.sidebar.selectbox(
    "Choose Variable",
    ["Temperature", "Humidity", "Wind Speed", "Pressure"],
    help="Select which climate variable you want to visualize in the dynamic trend chart."
)

st.sidebar.caption(f"Variable info: {description_map[selected_variable]}")

# Team members under filters
st.sidebar.markdown("---")
st.sidebar.subheader("Team Members")
st.sidebar.markdown("""
- Deem Ali – 44XXXXXXX  
- Student 2 – 44XXXXXXX  
- Student 3 – 44XXXXXXX  
""")

# ---------------------------
# Apply date filter
# ---------------------------
filtered = df.copy()

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["date"].dt.date >= start_date) &
        (filtered["date"].dt.date <= end_date)
    ]

selected_column = column_map[selected_variable]

# ---------------------------
# Top KPI cards
# ---------------------------
best_overall_row = metrics_df.loc[metrics_df["R2"].idxmax()]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Records", len(filtered))
c2.metric("Average Temperature", round(filtered["meantemp"].mean(), 2))
c3.metric("Average Humidity", round(filtered["humidity"].mean(), 2))
c4.metric("Average Wind Speed", round(filtered["wind_speed"].mean(), 2))
c5.metric("Best Overall Target", best_overall_row["Target"])

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Future Prediction", "Model Performance"])

# ===========================
# TAB 1: OVERVIEW
# ===========================
with tab1:
    st.subheader("Historical Climate Trends")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_temp = px.line(
            filtered,
            x="date",
            y="meantemp",
            title="Temperature Over Time"
        )
        st.plotly_chart(fig_temp, use_container_width=True, key="temp_chart")

    with col_right:
        fig_hum = px.line(
            filtered,
            x="date",
            y="humidity",
            title="Humidity Over Time"
        )
        st.plotly_chart(fig_hum, use_container_width=True, key="humidity_chart")

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        fig_wind = px.line(
            filtered,
            x="date",
            y="wind_speed",
            title="Wind Speed Over Time"
        )
        st.plotly_chart(fig_wind, use_container_width=True, key="wind_chart")

    with col_right2:
        fig_pressure = px.line(
            filtered,
            x="date",
            y="meanpressure",
            title="Pressure Over Time"
        )
        st.plotly_chart(fig_pressure, use_container_width=True, key="pressure_chart")

    st.subheader(f"{selected_variable} Trend")
    fig_dynamic = px.line(
        filtered,
        x="date",
        y=selected_column,
        title=f"{selected_variable} Over Time"
    )
    st.plotly_chart(
        fig_dynamic,
        use_container_width=True,
        key=f"dynamic_chart_{selected_variable}"
    )

    st.subheader("Filtered Climate Data")
    st.dataframe(filtered.tail(100), use_container_width=True)

# ===========================
# TAB 2: FUTURE PREDICTION
# ===========================
with tab2:
    st.subheader("Future Climate Prediction")
    st.markdown(
        "Select a future date to predict the climate variable using the best saved model."
    )

    # استبعدنا Pressure من التوقع النهائي لضعف الأداء
    prediction_target = st.selectbox(
        "Choose Prediction Target",
        ["Temperature", "Humidity", "Wind Speed"],
        help="Pressure is excluded from future prediction because its model performance is too weak."
    )

    future_date = st.date_input(
        "Select Future Date",
        value=max_date,
        help="Choose a date for which you want to estimate the future climate value."
    )

    def prepare_future_input(date_value):
        dt = pd.to_datetime(date_value)
        return pd.DataFrame([{
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "dayofweek": dt.dayofweek,
            "is_weekend": 1 if dt.dayofweek in [5, 6] else 0
        }])

    if st.button("Predict", use_container_width=True):
        input_data = prepare_future_input(future_date)
        prediction = models[prediction_target].predict(input_data)[0]

        st.success(f"Predicted {prediction_target}: {prediction:.2f}")

        st.info(
            "This is a pattern-based estimate learned from historical daily climate data."
        )

# ===========================
# TAB 3: MODEL PERFORMANCE
# ===========================
with tab3:
    st.subheader("Model Comparison")
    st.dataframe(metrics_df, use_container_width=True)

    fig_models = px.bar(
        metrics_df,
        x="Target",
        y="R2",
        color="Best Model",
        text="R2",
        title="Best Model by Target (R² Score)"
    )
    st.plotly_chart(fig_models, use_container_width=True, key="model_chart")

    st.markdown("### Notes")
    st.markdown(
        """
- **Temperature**: strongest target in the project.
- **Humidity**: acceptable but weaker than temperature.
- **Wind Speed**: weak performance, interpret with caution.
- **Pressure**: excluded from future prediction because its performance is too poor.
        """
    )

# ---------------------------
# Footer
# ---------------------------
st.caption(
    "Built with Streamlit for climate trend analysis and future prediction."
)
