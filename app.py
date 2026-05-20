import os
import re
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

st.set_page_config(page_title="EDA Project B", layout="wide")

st.title("Mini Project B - Time-Series Forecasting Starter")

st.header("Student Information")

student_name = st.text_input("Student Name", "Abdulhadi Alsaadi")
student_id = st.text_input("Student ID", "PG12S2540508")
project_title = st.text_input("Project Title", "PJM Grid Load Forecasting")
project_goal = st.text_area(
    "Project Goal",
    "Forecast electrical load demand using historical PJM grid data."
)
deployed_url = st.text_input("Deployed App URL", "")

st.header("Dataset Loader")

default_dataset_path = "data/dataset_sample.csv"
dataset_path = st.text_input("Dataset Path", default_dataset_path)

try:
    df = pd.read_csv(dataset_path)
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.subheader("First 10 Rows")
st.dataframe(df.head(10))

st.subheader("Dataset Audit")

dtype_df = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[col].dtype) for col in df.columns],
    "missing_percent": [round(df[col].isna().mean() * 100, 2) for col in df.columns]
})
st.dataframe(dtype_df)

timestamp_candidates = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()

st.header("Column Selection")

timestamp_col = st.selectbox(
    "Choose Timestamp Column",
    options=df.columns,
    index=df.columns.get_loc("Datetime") if "Datetime" in df.columns else 0
)

target_col = st.selectbox(
    "Choose Target Column",
    options=numeric_candidates,
    index=numeric_candidates.index("Load_MW") if "Load_MW" in numeric_candidates else 0
)

st.header("Time-Series Cleaning & Resampling")

df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

df = df.dropna(subset=[timestamp_col, target_col])
df = df.sort_values(timestamp_col)

resample_option = st.selectbox(
    "Optional Resampling Frequency",
    options=["None", "H", "D", "W", "M"],
    index=0
)

if resample_option != "None":
    temp_df = df.set_index(timestamp_col)
    df = temp_df.resample(resample_option)[target_col].mean().reset_index()

forecast_horizon = st.number_input(
    "Forecast Horizon",
    min_value=1,
    max_value=168,
    value=24
)

st.subheader("Cleaned Dataset Preview")
st.dataframe(df.head(10))

st.header("Baseline Feature Engineering")

feature_df = df.copy()

feature_df["lag_1"] = feature_df[target_col].shift(1)
feature_df["lag_24"] = feature_df[target_col].shift(24)

feature_df["rolling_mean_24"] = (
    feature_df[target_col]
    .shift(1)
    .rolling(window=24)
    .mean()
)

feature_df["hour"] = pd.to_datetime(feature_df[timestamp_col]).dt.hour
feature_df["weekend"] = (
    pd.to_datetime(feature_df[timestamp_col]).dt.dayofweek >= 5
).astype(int)

feature_df["month"] = pd.to_datetime(feature_df[timestamp_col]).dt.month

feature_df["y_target"] = feature_df[target_col].shift(-forecast_horizon)

feature_df = feature_df.dropna()

feature_columns = [
    "lag_1",
    "lag_24",
    "rolling_mean_24",
    "hour",
    "weekend",
    "month"
]

X = feature_df[feature_columns]
y = feature_df["y_target"]

st.subheader("Feature Table Preview")
st.dataframe(feature_df.head(10))

st.write("Feature Matrix Shape:", X.shape)
st.write("Target Vector Shape:", y.shape)

results_df = None

st.header("STUDENT ADDITIONS - MODELING")

st.code(
    """
# Add your forecasting models here

# Example:
# model_predictions = ...
# metrics = ...

# Create a metrics/results dataframe:
# results_df = pd.DataFrame({
#     "Model": [...],
#     "MAE": [...],
#     "RMSE": [...]
# })
"""
)

st.header("STUDENT ADDITIONS - DASHBOARD")

st.code(
    """
# Add extra visualizations and KPIs here

# Example:
# st.line_chart(...)
# st.metric(...)
"""
)

st.header("Export Section")

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
    "has_metrics_table": isinstance(results_df, pd.DataFrame),
    "results_table": [] if results_df is None else results_df.to_dict(orient="records")
}

submission_json = json.dumps(submission_data, indent=2)

project_card_md = f"""
# Project Card

## Student
- Name: {student_name}
- ID: {student_id}

## Project
- Title: {project_title}
- Goal: {project_goal}

## Dataset
- Timestamp Column: {timestamp_col}
- Target Column: {target_col}
- Rows: {len(df)}

## Features
- lag_1
- lag_24
- rolling_mean_24
- hour
- weekend
- month

## Forecast Horizon
{forecast_horizon}
"""

st.download_button(
    label="Download submission.json",
    data=submission_json,
    file_name="submission.json",
    mime="application/json"
)

st.download_button(
    label="Download project_card.md",
    data=project_card_md,
    file_name="project_card.md",
    mime="text/markdown"
)

st.header("AI Grader (/80)")

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

            raw_output = result["choices"][0]["message"]["content"]

            st.subheader("Raw AI Output")
            st.text(raw_output)

            parsed = None

            try:
                parsed = json.loads(raw_output)
            except Exception:
                match = re.search(r"\{.*\}", raw_output, re.DOTALL)

                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except Exception:
                        parsed = None

            if parsed:
                st.subheader("Parsed JSON")
                st.json(parsed)
            else:
                st.warning("Could not parse JSON response.")

        except Exception as e:
            st.error(f"AI grading failed: {e}")
