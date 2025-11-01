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

# ----------------------------------------------------
# Streamlit Configuration
# ----------------------------------------------------
st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")
st.title("📦 NexGen Logistics — Predictive Delivery Optimizer")
st.markdown("Predict delivery delays and recommend corrective actions using historical logistics data.")

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
@st.cache_data
def load_csv(name):
    """Load CSV if exists in current folder"""
    path = os.path.join(os.getcwd(), name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    return None

def safe_merge(left, right, on, how='left'):
    try:
        return left.merge(right, on=on, how=how)
    except Exception as e:
        st.warning(f"Merge failed on {on}: {e}")
        return left

def preprocess_for_model(df, target_col='is_delayed'):
    """Preprocess numeric and categorical features for ML model."""
    df = df.copy()

    numeric_feats = [c for c in df.select_dtypes(include=['int64','float64']).columns if c != target_col]
    cat_feats = [c for c in df.select_dtypes(include=['object']).columns if df[c].nunique() <= 50]

    if target_col not in df.columns:
        df[target_col] = 0

    X = df[numeric_feats + cat_feats]
    y = df[target_col]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('cat', cat_transformer, cat_feats)
    ])

    return X, y, preprocessor

def recommend_action(row):
    """Simple heuristic recommendation engine."""
    actions = []
    if row.get('delay_days', 0) > 0:
        if row.get('promised_delivery_days', 0) > 5:
            actions.append("Consider express delivery option.")
        if row.get('distance_km', 0) > 150:
            actions.append("Re-evaluate route efficiency or use local hubs.")
        if row.get('delivery_cost_inr', 0) > 500:
            actions.append("Negotiate cost or reassign to cost-effective carrier.")
    return " | ".join(actions) if actions else "On schedule. Continue normal tracking."

# ----------------------------------------------------
# Load Datasets
# ----------------------------------------------------
orders = load_csv("orders.csv")
delivery = load_csv("delivery_performance.csv")
routes = load_csv("routes_distance.csv")
fleet = load_csv("vehicle_fleet.csv")
warehouse = load_csv("warehouse_inventory.csv")
feedback = load_csv("customer_feedback.csv")
costs = load_csv("cost_breakdown.csv")

# ----------------------------------------------------
# Validate Required Data
# ----------------------------------------------------
if orders is None or delivery is None:
    st.error("Missing required datasets: 'orders.csv' and 'delivery_performance.csv'. Please upload them.")
    st.stop()

# ----------------------------------------------------
# Merge Datasets
# ----------------------------------------------------
df = orders.copy()
if 'order_id' in delivery.columns:
    df = safe_merge(df, delivery, on='order_id')
if routes is not None and 'order_id' in routes.columns:
    df = safe_merge(df, routes, on='order_id')
if costs is not None and 'order_id' in costs.columns:
    df = safe_merge(df, costs, on='order_id')
if fleet is not None and 'vehicle_id' in df.columns and 'vehicle_id' in fleet.columns:
    df = safe_merge(df, fleet, on='vehicle_id')

# ----------------------------------------------------
# Derive Delay Columns
# ----------------------------------------------------
if 'promised_delivery_days' in df.columns and 'actual_delivery_days' in df.columns:
    df['promised_delivery_days'] = pd.to_numeric(df['promised_delivery_days'], errors='coerce')
    df['actual_delivery_days'] = pd.to_numeric(df['actual_delivery_days'], errors='coerce')
    df['delay_days'] = df['actual_delivery_days'] - df['promised_delivery_days']
    df['delay_minutes'] = df['delay_days'] * 24 * 60
    df['is_delayed'] = (df['delay_days'] > 0).astype(int)
else:
    st.warning("Columns 'promised_delivery_days' or 'actual_delivery_days' not found in delivery dataset.")

# ----------------------------------------------------
# Summary Metrics
# ----------------------------------------------------
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Orders", len(df))
with col2:
    st.metric("Average Delay (days)", f"{df['delay_days'].mean():.2f}" if 'delay_days' in df.columns else "N/A")
with col3:
    st.metric("Delayed Orders", df['is_delayed'].sum() if 'is_delayed' in df.columns else "N/A")

# ----------------------------------------------------
# Visualizations
# ----------------------------------------------------
st.subheader("📈 Visual Insights")
if 'delay_days' in df.columns:
    fig_delay = px.histogram(df, x='delay_days', nbins=20, title="Distribution of Delivery Delays (days)")
    st.plotly_chart(fig_delay, use_container_width=True)

if 'delivery_cost_inr' in df.columns:
    fig_cost = px.box(df, y='delivery_cost_inr', title="Delivery Cost Distribution (INR)")
    st.plotly_chart(fig_cost, use_container_width=True)

if 'carrier' in df.columns and 'is_delayed' in df.columns:
    fig_carrier = px.bar(df.groupby('carrier')['is_delayed'].mean().reset_index(),
                         x='carrier', y='is_delayed', title="Delay Rate by Carrier")
    st.plotly_chart(fig_carrier, use_container_width=True)

# ----------------------------------------------------
# Model Training
# ----------------------------------------------------
st.subheader("🧠 Train Delay Prediction Model")
delay_threshold = st.slider("Delay threshold (days)", 0, 5, 1)

labeled_df = df.dropna(subset=['delay_days']).copy()
if len(labeled_df) >= 30:
    X, y, preproc = preprocess_for_model(labeled_df, 'is_delayed')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = Pipeline([
        ('preproc', preproc),
        ('model', RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    with st.spinner("Training model..."):
        clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.6).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    st.write(f"**ROC AUC:** {auc:.3f}")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(clf, "delay_classifier.joblib")
    st.success("Model trained and saved as 'delay_classifier.joblib'")
else:
    st.warning("Not enough data to train model. Need at least 30 labeled rows.")

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
st.subheader("🚚 Predict Delay for Orders")
if os.path.exists("delay_classifier.joblib"):
    clf = joblib.load("delay_classifier.joblib")
    predict_df = df.copy()

    sel_order = st.selectbox("Select Order ID to Predict", predict_df['order_id'].unique())
    row = predict_df[predict_df['order_id'] == sel_order].iloc[0]
    X_row = pd.DataFrame([row.drop(labels=['order_id'], errors='ignore')])

    try:
        prob = clf.predict_proba(X_row)[0, 1]
        pred = "Delayed" if prob >= 0.6 else "On-Time"
        st.write(f"**Predicted Probability of Delay:** {prob:.2f}")
        st.write(f"**Prediction:** {pred}")
        st.write(f"**Recommendation:** {recommend_action(row)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Train a model first to enable predictions.")

# ----------------------------------------------------
# Business Impact
# ----------------------------------------------------
st.subheader("💰 Business Impact Simulation")
if 'delivery_cost_inr' in df.columns and 'is_delayed' in df.columns:
    avg_cost_delay = df.loc[df['is_delayed'] == 1, 'delivery_cost_inr'].mean()
    reduction = st.slider("Simulate % reduction in delays", 0, 100, 20)
    saved_orders = df['is_delayed'].sum() * (reduction / 100)
    est_savings = saved_orders * avg_cost_delay
    st.write(f"Estimated Cost Savings: ₹{est_savings:,.0f}")
else:
    st.info("Missing required columns to calculate business impact (delivery_cost_inr, is_delayed).")
