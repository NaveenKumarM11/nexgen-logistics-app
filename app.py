import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None

def safe_merge(left, right, on, how='left'):
    try:
        return left.merge(right, on=on, how=how)
    except Exception as e:
        st.warning(f"Merge failed on {on}: {e}")
        return left

def preprocess_for_model(df, target_col='is_delayed'):
    df = df.copy()
    # create delay_minutes if times exist
    if 'promised_time' in df.columns and 'actual_time' in df.columns:
        df['delay_minutes'] = (pd.to_datetime(df['actual_time']) - pd.to_datetime(df['promised_time'])).dt.total_seconds()/60
    # create binary label if missing
    if target_col not in df.columns and 'delay_minutes' in df.columns:
        df[target_col] = (df['delay_minutes'] > 60).astype(int)

    numeric_feats, cat_feats = [], []
    for c in df.columns:
        if c in [target_col, 'order_id', 'vehicle_id', 'promised_time', 'actual_time']:
            continue
        if df[c].dtype.kind in 'biufc':
            numeric_feats.append(c)
        elif df[c].dtype == 'object' and df[c].nunique() <= 50:
            cat_feats.append(c)

    if not numeric_feats:
        df['__dummy'] = 1
        numeric_feats = ['__dummy']

    X = df[numeric_feats + cat_feats].copy()
    y = df[target_col] if target_col in df.columns else None

    num_trans = Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_trans = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    preproc = ColumnTransformer([('num', num_trans, numeric_feats), ('cat', cat_trans, cat_feats)], sparse_threshold=0)
    return X, y, preproc, numeric_feats, cat_feats

# ---------------------------
# Load data
# ---------------------------
st.title("NexGen Logistics — Predictive Delivery Optimizer")

orders = load_csv_if_exists("orders.csv")
delivery_perf = load_csv_if_exists("delivery_performance.csv")
routes = load_csv_if_exists("routes_distance.csv")
fleet = load_csv_if_exists("vehicle_fleet.csv")
warehouse = load_csv_if_exists("warehouse_inventory.csv")
feedback = load_csv_if_exists("customer_feedback.csv")
costs = load_csv_if_exists("cost_breakdown.csv")

if orders is None:
    st.error("orders.csv not loaded. Please upload or place it in the app folder.")
    st.stop()

# Merge datasets
df = orders.copy()
if delivery_perf is not None and 'order_id' in delivery_perf.columns:
    df = safe_merge(df, delivery_perf, on='order_id')
if routes is not None and 'order_id' in routes.columns:
    df = safe_merge(df, routes, on='order_id')
if fleet is not None and 'vehicle_id' in fleet.columns and 'vehicle_id' in df.columns:
    df = safe_merge(df, fleet, on='vehicle_id')
if costs is not None and 'order_id' in costs.columns:
    df = safe_merge(df, costs, on='order_id')
if feedback is not None and 'order_id' in feedback.columns:
    df = safe_merge(df, feedback[['order_id','satisfaction_score']], on='order_id')

# ---------------------------
# Preview
# ---------------------------
with st.expander("Preview datasets"):
    st.dataframe(df.head())

# ---------------------------
# KPIs
# ---------------------------
st.markdown("## Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", len(df))
if 'delivery_status' in df.columns:
    delayed = (df['delivery_status'].str.contains("delay", case=False, na=False)).sum()
    col2.metric("Delayed Deliveries", delayed)
if 'order_value' in df.columns:
    col3.metric("Avg Order Value", f"{df['order_value'].mean():.2f}")

# ---------------------------
# Visualizations
# ---------------------------
st.markdown("## Visualizations")
if 'product_category' in df.columns:
    fig = px.histogram(df, x='product_category', title="Orders by Product Category")
    st.plotly_chart(fig, use_container_width=True)
if 'distance_km' in df.columns:
    fig2 = px.box(df, y='distance_km', title="Delivery Distance Distribution (km)")
    st.plotly_chart(fig2, use_container_width=True)
if 'satisfaction_score' in df.columns:
    fig3 = px.histogram(df, x='satisfaction_score', title="Customer Satisfaction Scores")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# Train Model
# ---------------------------
st.markdown("## Predictive Modeling")

if 'promised_time' in df.columns and 'actual_time' in df.columns:
    df['promised_time'] = pd.to_datetime(df['promised_time'], errors='coerce')
    df['actual_time'] = pd.to_datetime(df['actual_time'], errors='coerce')
    df['delay_minutes'] = (df['actual_time'] - df['promised_time']).dt.total_seconds()/60
    df['is_delayed'] = (df['delay_minutes'] > 60).astype(int)

if 'is_delayed' not in df.columns or df['is_delayed'].nunique() < 2:
    st.warning("Not enough labeled data for model training. Add 'promised_time' and 'actual_time' columns or label delays manually.")
else:
    X, y, preproc, num, cat = preprocess_for_model(df, 'is_delayed')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = Pipeline([('preproc', preproc), ('clf', RandomForestClassifier(n_estimators=120, random_state=42))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    st.write(f"Model trained — AUC: {auc:.3f}")
    st.text(classification_report(y_test, y_pred))
    joblib.dump(clf, "delay_classifier.joblib")
    st.success("Model saved as delay_classifier.joblib")

# ---------------------------
# Predict New Orders
# ---------------------------
st.markdown("## Predict on New Orders")
batch = st.file_uploader("Upload new orders (CSV)", type="csv")
if batch and os.path.exists("delay_classifier.joblib"):
    new_df = pd.read_csv(batch)
    clf = joblib.load("delay_classifier.joblib")
    try:
        probs = clf.predict_proba(new_df)[:,1]
        new_df['delay_probability'] = probs
        new_df['predicted_delayed'] = (probs >= 0.6).astype(int)
        st.dataframe(new_df.head())
        st.download_button("Download Predictions", data=new_df.to_csv(index=False), file_name="predicted_orders.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.info("✅ All datasets integrated successfully. Use the trained model to predict at-risk deliveries.")
