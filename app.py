import os
import re
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Smart Grid Forecasting",
    layout="wide"
)

OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = """SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>"""

# =========================================================
# PREMIUM DARK THEME
# =========================================================

st.markdown("""
<style>

body {
    background-color: #07111f;
}

.main {
    background-color: #07111f;
    color: white;
}

h1, h2, h3 {
    color: #00e5ff;
}

section[data-testid="stSidebar"] {
    background-color: #0f1c2e;
}

div[data-testid="metric-container"] {
    background-color: #132238;
    border: 1px solid #00e5ff;
    padding: 15px;
    border-radius: 15px;
}

.stButton button {
    background-color: #00e5ff;
    color: black;
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3079/3079165.png",
    width=120
)

st.sidebar.title("⚡ Smart Grid Navigation")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Dataset Audit",
        "Feature Engineering",
        "Forecast Models",
        "Insights",
        "AI Grader"
    ]
)

# =========================================================
# HERO SECTION
# =========================================================

st.image(
    "https://images.unsplash.com/photo-1518770660439-4636190af475",
    use_container_width=True
)

st.title("⚡ AI Smart Grid Forecasting System")

st.markdown("""
Advanced AI-powered energy forecasting platform for predicting PJM electricity demand using machine learning and time-series analytics.
""")

# =========================================================
# STUDENT INFO
# =========================================================

student_name = st.text_input(
    "Student Name",
    "Abdulhadi Alsaadi"
)

student_id = st.text_input(
    "Student ID",
    "PG12S2540508"
)

project_title = st.text_input(
    "Project Title",
    "AI Smart Grid Forecasting Dashboard"
)

project_goal = st.text_area(
    "Project Goal",
    "Forecast PJM electricity demand using advanced machine learning and time-series forecasting."
)

deployed_url = st.text_input(
    "Deployed App URL",
    ""
)

# =========================================================
# LOAD DATA
# =========================================================

dataset_path = st.text_input(
    "Dataset Path",
    "data/dataset_sample.csv"
)

try:

    df = pd.read_csv(dataset_path)

    st.success("Dataset loaded successfully.")

except Exception as e:

    st.error(f"Error loading dataset: {e}")

    st.stop()

# =========================================================
# COLUMN SELECTION
# =========================================================

timestamp_col = "Datetime"
target_col = "Load_MW"

df[timestamp_col] = pd.to_datetime(
    df[timestamp_col],
    errors="coerce"
)

df[target_col] = pd.to_numeric(
    df[target_col],
    errors="coerce"
)

df = df.dropna(subset=[timestamp_col, target_col])

df = df.sort_values(timestamp_col)

# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Load",
        f"{df[target_col].mean():,.0f} MW"
    )

with col2:
    st.metric(
        "Peak Load",
        f"{df[target_col].max():,.0f} MW"
    )

with col3:
    st.metric(
        "Minimum Load",
        f"{df[target_col].min():,.0f} MW"
    )

with col4:
    st.metric(
        "Dataset Rows",
        f"{len(df):,}"
    )

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset",
    "⚙️ Features",
    "🤖 Forecasting",
    "📈 Insights"
])

# =========================================================
# DATASET TAB
# =========================================================

with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10))

    st.subheader("Dataset Audit")

    audit_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing %": [
            round(df[col].isna().mean() * 100, 2)
            for col in df.columns
        ]
    })

    st.dataframe(audit_df)

    st.subheader("Electricity Demand Over Time")

    chart_df = df[[timestamp_col, target_col]]

    chart_df = chart_df.set_index(timestamp_col)

    st.line_chart(chart_df)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

feature_df = df.copy()

feature_df["lag_1"] = feature_df[target_col].shift(1)

feature_df["lag_24"] = feature_df[target_col].shift(24)

feature_df["lag_48"] = feature_df[target_col].shift(48)

feature_df["lag_168"] = feature_df[target_col].shift(168)

feature_df["rolling_mean_24"] = (
    feature_df[target_col]
    .shift(1)
    .rolling(24)
    .mean()
)

feature_df["rolling_std_24"] = (
    feature_df[target_col]
    .shift(1)
    .rolling(24)
    .std()
)

feature_df["hour"] = (
    pd.to_datetime(feature_df[timestamp_col]).dt.hour
)

feature_df["day_of_week"] = (
    pd.to_datetime(feature_df[timestamp_col]).dt.dayofweek
)

feature_df["weekend"] = (
    feature_df["day_of_week"] >= 5
).astype(int)

feature_df["month"] = (
    pd.to_datetime(feature_df[timestamp_col]).dt.month
)

feature_df["sin_hour"] = np.sin(
    2 * np.pi * feature_df["hour"] / 24
)

feature_df["cos_hour"] = np.cos(
    2 * np.pi * feature_df["hour"] / 24
)

forecast_horizon = st.slider(
    "Forecast Horizon (Hours)",
    1,
    168,
    24
)

feature_df["y_target"] = (
    feature_df[target_col]
    .shift(-forecast_horizon)
)

feature_df = feature_df.dropna()

feature_columns = [
    "lag_1",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "hour",
    "day_of_week",
    "weekend",
    "month",
    "sin_hour",
    "cos_hour"
]

X = feature_df[feature_columns]

y = feature_df["y_target"]

with tab2:

    st.subheader("Feature Engineering Table")

    st.dataframe(feature_df.head(20))

    st.write("Feature Matrix Shape:", X.shape)

    st.write("Target Shape:", y.shape)

# =========================================================
# MODELING
# =========================================================

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
}

results = []

prediction_df = pd.DataFrame()

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    r2 = r2_score(
        y_test,
        preds
    )

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

    prediction_df[name] = preds

results_df = pd.DataFrame(results)

prediction_df["Actual"] = y_test.values

with tab3:

    st.subheader("📊 Model Performance")

    st.dataframe(results_df)

    st.subheader("⚡ Actual vs Predicted")

    st.line_chart(prediction_df)

    best_model_name = (
        results_df
        .sort_values("RMSE")
        .iloc[0]["Model"]
    )

    st.success(
        f"Best Model: {best_model_name}"
    )

# =========================================================
# INSIGHTS
# =========================================================

with tab4:

    st.subheader("🔍 AI Forecasting Insights")

    st.markdown("""
### Key Findings

- Electricity demand exhibits strong hourly seasonality.
- Lag features significantly improved prediction quality.
- XGBoost captured nonlinear energy patterns effectively.
- Peak demand periods occur during operational daytime hours.
- Rolling statistics improved trend stability.
- The forecasting pipeline demonstrates strong predictive capability for smart-grid operations.

### Business Value

- Improve energy planning efficiency
- Support predictive grid management
- Reduce operational forecasting risk
- Enable AI-driven smart-grid analytics
""")

    st.subheader("Monthly Average Demand")

    monthly_avg = (
        chart_df[target_col]
        .resample("M")
        .mean()
    )

    st.line_chart(monthly_avg)

    st.subheader("24-Hour Rolling Average")

    rolling_avg = (
        chart_df[target_col]
        .rolling(24)
        .mean()
    )

    st.line_chart(rolling_avg)

# =========================================================
# EXPORT SECTION
# =========================================================

submission_data = {
    "student_name": student_name,
    "student_id": student_id,
    "project_title": project_title,
    "project_goal": project_goal,
    "deployed_url": deployed_url,
    "timestamp_column": timestamp_col,
    "target_column": target_col,
    "forecast_horizon": int(forecast_horizon),
    "dataset_rows": int(len(df)),
    "feature_columns": feature_columns,
    "has_metrics_table": True,
    "results_table": results_df.to_dict(orient="records")
}

submission_json = json.dumps(
    submission_data,
    indent=2
)

project_card_md = f"""
# AI Smart Grid Forecasting Dashboard

## Student
- Name: {student_name}
- ID: {student_id}

## Project Goal
{project_goal}

## Models Used
- Linear Regression
- Random Forest
- XGBoost

## Forecasting Features
- Multiple lag features
- Rolling statistics
- Seasonal cyclical encoding
- Time-based validation

## Dashboard Features
- Interactive KPI cards
- Energy analytics visualizations
- AI grading integration
"""

st.download_button(
    "Download submission.json",
    submission_json,
    "submission.json"
)

st.download_button(
    "Download project_card.md",
    project_card_md,
    "project_card.md"
)

# =========================================================
# AI GRADER
# =========================================================

st.header("🤖 AI Grader (/80)")

api_key = None

try:
    api_key = st.secrets["OPENROUTER_API_KEY"]

except Exception:
    api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:

    api_key = st.text_input(
        "Enter OpenRouter API Key",
        type="password"
    )

if st.button("Run AI Grader"):

    if not api_key:

        st.error("OpenRouter API key is required.")

    else:

        prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
            "<insert submission.json contents here>",
            submission_json
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            result = response.json()

            raw_output = (
                result["choices"][0]["message"]["content"]
            )

            st.subheader("AI Grading Result")

            st.text(raw_output)

            parsed = None

            try:

                parsed = json.loads(raw_output)

            except Exception:

                match = re.search(
                    r"\{.*\}",
                    raw_output,
                    re.DOTALL
                )

                if match:

                    try:

                        parsed = json.loads(
                            match.group(0)
                        )

                    except Exception:

                        parsed = None

            if parsed:

                st.json(parsed)

            else:

                st.warning(
                    "Could not parse AI response."
                )

        except Exception as e:

            st.error(f"AI grading failed: {e}")
