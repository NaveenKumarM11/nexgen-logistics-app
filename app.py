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
from sklearn.inspection import permutation_importance
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
    """
    Minimal preprocessing:
      - drop id columns that leak target if any
      - handle categorical and numeric features with a ColumnTransformer
    Returns X, y, pipeline (transformer), numeric_feats, cat_feats
    """
    df = df.copy()
    # Derived features
    if 'promised_time' in df.columns and 'actual_time' in df.columns:
        df['delay_minutes'] = (pd.to_datetime(df['actual_time']) - pd.to_datetime(df['promised_time'])).dt.total_seconds()/60
    # If target is not present but delay_minutes exist, create binary is_delayed
    if target_col not in df.columns and 'delay_minutes' in df.columns:
        df[target_col] = (df['delay_minutes'] > 60).astype(int)  # delay if > 60 minutes (configurable)
    # Select candidate features
    # Prefer fields available across many rows
    numeric_feats = []
    cat_feats = []
    for c in df.columns:
        if c in [target_col, 'order_id', 'customer_id', 'promised_time', 'actual_time', 'delivery_status']:
            continue
        if df[c].dtype.kind in 'biufc' and df[c].nunique() > 2:
            numeric_feats.append(c)
        elif df[c].dtype == 'object':
            if df[c].nunique() <= 50:
                cat_feats.append(c)
    # If no features found, create simple numeric proxy
    if not numeric_feats and 'distance_km' in df.columns:
        numeric_feats = ['distance_km']
    if not numeric_feats and not cat_feats:
        # fallback create synthetic feature
        df['__const'] = 1
        numeric_feats = ['__const']
    X = df[numeric_feats + cat_feats].copy()
    y = df[target_col] if target_col in df.columns else None
    # Build pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', cat_transformer, cat_feats)
    ], sparse_threshold=0)
    return X, y, preprocessor, numeric_feats, cat_feats

def recommend_action(row, top_features):
    """
    Simple rule-based recommendation engine:
    - If high probability of delay and cause includes traffic/weather/distance -> reroute / expedite
    - If vehicle capacity or vehicle_age related -> reassign vehicle
    - If carrier issues -> change carrier
    This function accepts a pandas Series for a single order (features).
    """
    actions = []
    # heuristics from columns that commonly exist
    if 'traffic_delay_minutes' in row.index and not pd.isna(row['traffic_delay_minutes']) and row['traffic_delay_minutes']>30:
        actions.append("Consider reroute or off-peak dispatch (traffic delay high).")
    if 'weather_impact' in row.index and str(row['weather_impact']).lower() not in ['0','none','nan'] and row['weather_impact']!='0':
        actions.append("Hold / consolidate or switch to slower mode if safe; notify customer of weather delay.")
    if 'distance_km' in row.index and row.get('distance_km',0) > 150:
        actions.append("Use trunk transport + local last-mile; consider express carrier for long-distance express orders.")
    if 'vehicle_age' in row.index and row.get('vehicle_age',0) > 8:
        actions.append("Assign newer vehicle / inspect vehicle to reduce breakdown risk.")
    if 'carrier' in row.index and str(row.get('carrier')).lower() in ['carrier_x','carrier_with_high_delay']:
        actions.append("Consider switching carrier for this route (historically high delays).")
    if not actions:
        actions.append("Prioritize this order for proactive monitoring; notify driver & customer; prepare return/replacement process.")
    return " | ".join(actions)

# ---------------------------
# UI - Sidebar: load files
# ---------------------------
st.sidebar.title("Data input")
st.sidebar.markdown("Upload CSVs or place them in the same folder as app.py with exact names.")
orders = load_csv_if_exists("orders.csv")
delivery_perf = load_csv_if_exists("delivery_performance.csv")
routes = load_csv_if_exists("routes_distance.csv")
fleet = load_csv_if_exists("vehicle_fleet.csv")
warehouse = load_csv_if_exists("warehouse_inventory.csv")
feedback = load_csv_if_exists("customer_feedback.csv")
costs = load_csv_if_exists("cost_breakdown.csv")

uploaded_orders = st.sidebar.file_uploader("orders.csv", type="csv", key="orders")
if uploaded_orders:
    orders = pd.read_csv(uploaded_orders)
uploaded_delivery = st.sidebar.file_uploader("delivery_performance.csv", type="csv", key="delivery")
if uploaded_delivery:
    delivery_perf = pd.read_csv(uploaded_delivery)
# allow other uploads
uploaded_routes = st.sidebar.file_uploader("routes_distance.csv", type="csv", key="routes")
if uploaded_routes:
    routes = pd.read_csv(uploaded_routes)
uploaded_fleet = st.sidebar.file_uploader("vehicle_fleet.csv", type="csv", key="fleet")
if uploaded_fleet:
    fleet = pd.read_csv(uploaded_fleet)
uploaded_wh = st.sidebar.file_uploader("warehouse_inventory.csv", type="csv", key="wh")
if uploaded_wh:
    warehouse = pd.read_csv(uploaded_wh)
uploaded_feedback = st.sidebar.file_uploader("customer_feedback.csv", type="csv", key="feedback")
if uploaded_feedback:
    feedback = pd.read_csv(uploaded_feedback)
uploaded_costs = st.sidebar.file_uploader("cost_breakdown.csv", type="csv", key="costs")
if uploaded_costs:
    costs = pd.read_csv(uploaded_costs)

st.sidebar.markdown("---")
st.sidebar.write("Model & Thresholds")
delay_threshold_minutes = st.sidebar.number_input("Delay threshold (minutes) for labeling 'delayed'", value=60)
prob_threshold = st.sidebar.slider("Probability threshold to classify as 'at-risk'", 0.4, 0.9, 0.6)

# ---------------------------
# Basic checks
# ---------------------------
st.title("NexGen Logistics — Predictive Delivery Optimizer")
st.markdown("Interactive prototype to predict delivery delays and recommend corrective actions.")

if orders is None:
    st.error("orders.csv not loaded. Upload or place the file in the app folder.")
    st.stop()

# show small preview
with st.expander("Preview datasets"):
    st.write("orders.csv preview")
    st.dataframe(orders.head(10))
    st.write("delivery_performance.csv preview")
    st.dataframe(delivery_perf.head(10) if delivery_perf is not None else "Not provided")

# ---------------------------
# Merge datasets into unified table
# ---------------------------
# Start with orders
df = orders.copy()
# normalize column names
df.columns = [c.strip() for c in df.columns]

# Merge delivery performance on order_id if available
if delivery_perf is not None and 'order_id' in delivery_perf.columns:
    delivery_perf.columns = [c.strip() for c in delivery_perf.columns]
    df = safe_merge(df, delivery_perf, on='order_id')
# Merge routes
if routes is not None and 'order_id' in routes.columns:
    routes.columns = [c.strip() for c in routes.columns]
    df = safe_merge(df, routes, on='order_id')
# Merge fleet by vehicle id if present
if fleet is not None and 'vehicle_id' in fleet.columns and 'vehicle_id' in df.columns:
    df = safe_merge(df, fleet, on='vehicle_id')
# Merge costs by order_id
if costs is not None and 'order_id' in costs.columns:
    df = safe_merge(df, costs, on='order_id')

st.markdown("## Key metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_orders = len(orders)
    st.metric("Total orders (dataset)", total_orders)
with col2:
    on_time = 0
    if 'delivery_status' in df.columns:
        on_time = (df['delivery_status'].str.lower() == 'delivered_on_time').sum()
    st.metric("Known on-time deliveries", on_time)
with col3:
    avg_order_value = orders['order_value'].mean() if 'order_value' in orders.columns else None
    st.metric("Avg order value", f"{avg_order_value:.2f}" if avg_order_value is not None else "N/A")
with col4:
    avg_distance = df['distance_km'].mean() if 'distance_km' in df.columns else None
    st.metric("Avg distance (km)", f"{avg_distance:.1f}" if avg_distance is not None else "N/A")

# ---------------------------
# Filters for dashboard
# ---------------------------
st.sidebar.markdown("### Filters")
seg_options = orders['customer_segment'].unique().tolist() if 'customer_segment' in orders.columns else []
sel_segment = st.sidebar.multiselect("Customer segments", options=seg_options, default=seg_options)
priority_options = orders['priority'].unique().tolist() if 'priority' in orders.columns else []
sel_priority = st.sidebar.multiselect("Priority level", options=priority_options, default=priority_options)

filtered = df.copy()
if sel_segment:
    if 'customer_segment' in filtered.columns:
        filtered = filtered[filtered['customer_segment'].isin(sel_segment)]
if sel_priority:
    if 'priority' in filtered.columns:
        filtered = filtered[filtered['priority'].isin(sel_priority)]

# ---------------------------
# Visualizations (4+ types)
# ---------------------------
st.markdown("## Visualizations")
viz_col1, viz_col2 = st.columns([2,1])

with viz_col1:
    # 1) Bar: Orders by product category
    if 'product_category' in filtered.columns:
        fig1 = px.histogram(filtered, x='product_category', title="Orders by Product Category", labels={'count':'orders'})
        st.plotly_chart(fig1, use_container_width=True)
    # 2) Line: Delay trend over time (if timestamps exist)
    if 'order_date' in filtered.columns:
        try:
            filtered['order_date'] = pd.to_datetime(filtered['order_date'])
            trend = filtered.groupby(filtered['order_date'].dt.to_period("W")).size().reset_index(name='orders')
            trend['order_date'] = trend['order_date'].dt.start_time
            fig2 = px.line(trend, x='order_date', y='orders', title="Orders over time (weekly)")
            st.plotly_chart(fig2, use_container_width=True)
        except:
            pass

with viz_col2:
    # 3) Pie / Donut: Delivery status
    if 'delivery_status' in filtered.columns:
        ds = filtered['delivery_status'].value_counts().reset_index()
        ds.columns = ['status','count']
        fig3 = px.pie(ds, names='status', values='count', title="Delivery status distribution")
        st.plotly_chart(fig3, use_container_width=True)
    # 4) Box: Delivery cost distribution
    if 'delivery_cost' in filtered.columns:
        fig4 = px.box(filtered, y='delivery_cost', title="Delivery cost distribution")
        st.plotly_chart(fig4, use_container_width=True)

# 5) Scatter: distance vs delivery time (if both exist)
if 'distance_km' in filtered.columns and 'actual_delivery_minutes' in filtered.columns:
    fig5 = px.scatter(filtered, x='distance_km', y='actual_delivery_minutes', color='priority' if 'priority' in filtered.columns else None,
                      title="Distance vs Actual Delivery Minutes", trendline="ols")
    st.plotly_chart(fig5, use_container_width=True)

# ---------------------------
# Modeling: Delay classification + Regression (if delay minutes exist)
# ---------------------------
st.markdown("## Predictive models")
st.markdown("Train a classifier that predicts whether an order will be delayed (binary). Optionally train a regression to estimate delay minutes.")

# Prepare labeled data where actual_time/promised_time exist
labeled_df = df.copy()
if 'promised_time' in df.columns and 'actual_time' in df.columns:
    labeled_df['promised_time'] = pd.to_datetime(labeled_df['promised_time'], errors='coerce')
    labeled_df['actual_time'] = pd.to_datetime(labeled_df['actual_time'], errors='coerce')
    labeled_df['delay_minutes'] = (labeled_df['actual_time'] - labeled_df['promised_time']).dt.total_seconds()/60
    labeled_df['is_delayed'] = (labeled_df['delay_minutes'] > delay_threshold_minutes).astype(int)
else:
    st.warning("promised_time and/or actual_time not available in dataset — training will use any existing 'is_delayed' label or will be limited.")
    if 'is_delayed' not in labeled_df.columns:
        st.info("No delay labels found. You can label historical rows that are delayed and retry.")
        labeled_df = labeled_df.iloc[0:0]  # empty

if len(labeled_df) >= 30:
    X, y, preproc, num_feats, cat_feats = preprocess_for_model(labeled_df, target_col='is_delayed')
    # train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, labeled_df.index, test_size=0.2, random_state=42, stratify=y)
    # Build pipeline
    clf = Pipeline(steps=[
        ('preproc', preproc),
        ('clf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
    ])
    with st.spinner("Training classifier..."):
        clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= prob_threshold).astype(int)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test))>1 else None
    st.write("### Classification performance (test set)")
    st.write(f"AUC: {auc:.3f}" if auc is not None else "AUC: N/A (single-class in test set)")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    # Feature importance via permutation on transformed data
    try:
        st.write("Top feature importances (permutation importance)")
        X_test_trans = clf.named_steps['preproc'].transform(X_test)
        r = permutation_importance(clf.named_steps['clf'], X_test_trans, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        # map back feature names
        num_cols = num_feats
        cat_cols = []
        if cat_feats:
            ohe = clf.named_steps['preproc'].named_transformers_['cat'].named_steps['onehot']
            ohe_features = list(ohe.get_feature_names_out(cat_feats))
            cat_cols = ohe_features
        feat_names = num_cols + cat_cols
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': r.importances_mean})
        imp_df = imp_df.sort_values('importance', ascending=False).head(15)
        fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h', title="Permutation importances (top 15)")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.info(f"Could not compute permutation importances: {e}")

    # Regression for delay minutes if enough labeled rows
    if labeled_df['delay_minutes'].notna().sum() >= 30:
        Xr = X.copy()
        yr = labeled_df.loc[X.index, 'delay_minutes']
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        reg = Pipeline(steps=[
            ('preproc', preproc),
            ('reg', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        with st.spinner("Training regression model..."):
            reg.fit(Xr_train, yr_train)
        pred_r = reg.predict(Xr_test)
        st.write("### Regression performance (delay minutes)")
        st.write(f"MAE: {mean_absolute_error(yr_test, pred_r):.1f} minutes")
        st.write(f"R2: {r2_score(yr_test, pred_r):.3f}")
    else:
        reg = None
        st.info("Not enough data for regression (delay minutes).")

    # Save models to disk
    joblib.dump(clf, "delay_classifier.joblib")
    if 'reg' in locals() and reg is not None:
        joblib.dump(reg, "delay_regressor.joblib")
    st.success("Trained models saved: delay_classifier.joblib (and delay_regressor.joblib if available)")
else:
    st.warning("Not enough labeled historical rows to train models. Need >= 30 labeled rows with promised_time & actual_time (or is_delayed label).")

# ---------------------------
# Prediction interface for new orders
# ---------------------------
st.markdown("## Predict on selected orders / new data")
st.write("Select orders to score or upload new orders CSV to run batch predictions.")

# Select subset to predict (e.g., transit or recent)
predict_df = df.copy()
if 'delivery_status' in predict_df.columns:
    # e.g., predict on 'In Transit' or missing actual_time
    predict_df['in_transit'] = predict_df['actual_time'].isna() if 'actual_time' in predict_df.columns else False
    subset = predict_df[predict_df['in_transit'] == True]
else:
    subset = predict_df

# show sample
st.write("Sample orders (to predict):")
st.dataframe(subset.head(10))

# If model exists, allow scoring
if os.path.exists("delay_classifier.joblib"):
    clf = joblib.load("delay_classifier.joblib")
    # allow selecting an order_id to score
    if 'order_id' in subset.columns:
        sel = st.selectbox("Pick an order_id to score", options=subset['order_id'].unique())
        order_row = subset[subset['order_id'] == sel].iloc[0]
        # prepare X row like training
        X_row = pd.DataFrame([order_row.drop(labels=[c for c in ['order_id','actual_time','promised_time'] if c in order_row.index])])
        # keep only columns that preprocessor expects
        try:
            # Align column order
            # The pipeline expects the same feature names used during training; to be robust, attempt to select those
            # We'll use preproc from the loaded pipeline
            preproc = clf.named_steps['preproc']
            # Create X_input with numeric & cat used in training: find required input columns from preproc
            required_cols = []
            # This is approximate: if transformer has 'num' and 'cat' names
            # We'll attempt to infer numeric and categorical columns used previously from preproc
            # Fallback: pass X_row as-is; preproc will select known columns and ignore rest if configured properly
            prob = clf.predict_proba(X_row)[0,1]
            pred = (prob >= prob_threshold)
            st.write(f"Predicted probability of delay: **{prob:.2f}** (threshold {prob_threshold})")
            st.write("Predicted class:", "At-risk (Delayed)" if pred else "On-time")
            # recommended actions
            rec = recommend_action(order_row, top_features=None)
            st.write("**Recommended actions:**")
            st.write(rec)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("No order_id column available to pick a single order. You can upload a new CSV of orders for batch scoring.")
else:
    st.info("No trained classifier available. Train model first (requires labeled data).")

# Batch scoring upload
st.markdown("### Batch scoring")
batch_file = st.file_uploader("Upload new orders (CSV) for batch scoring", type="csv", key="batch")
if batch_file is not None and os.path.exists("delay_classifier.joblib"):
    new_df = pd.read_csv(batch_file)
    clf = joblib.load("delay_classifier.joblib")
    try:
        probs = clf.predict_proba(new_df)[:,1]
        new_df['delay_probability'] = probs
        new_df['predicted_delayed'] = (new_df['delay_probability'] >= prob_threshold).astype(int)
        st.dataframe(new_df.head(20))
        st.markdown("Download scored results:")
        st.download_button("Download CSV", data=new_df.to_csv(index=False), file_name="scored_orders.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch scoring failed: {e}")
elif batch_file is not None:
    st.warning("No trained classifier found. Train classifier first.")

# ---------------------------
# Business impact estimation
# ---------------------------
st.markdown("## Business impact estimation (simulation)")
st.write("Estimate cost of delays and potential savings if at-risk orders are proactively handled.")

if 'delivery_cost' in df.columns and 'is_delayed' in df.columns:
    base_delay_cost_per_order = df.loc[df['is_delayed']==1, 'delivery_cost'].mean()
    st.write(f"Estimated average extra cost for delayed order (historical): {base_delay_cost_per_order:.2f}")
    # Simulate reducing delayed count by a percentage
    reduce_pct = st.slider("Simulate % reduction in delayed orders via interventions", 0, 100, 20)
    current_delays = df['is_delayed'].sum()
    reduced = int(current_delays * (1 - reduce_pct/100))
    saved_orders = current_delays - reduced
    est_savings = saved_orders * base_delay_cost_per_order
    st.write(f"Estimated saved orders: {saved_orders}; Estimated cost savings: {est_savings:.2f}")
else:
    st.info("Need delivery_cost and is_delayed columns to estimate monetary impact.")

# ---------------------------
# Export: README and brief
# ---------------------------
st.sidebar.markdown("---")
if st.sidebar.button("Download README"):
    readme_text = """
# Predictive Delivery Optimizer - README

Instructions...
"""
    st.sidebar.download_button("Download README", data=readme_text, file_name="README.md", mime="text/markdown")

st.info("Prototype complete. Use the trained models and recommendations to integrate with NexGen operations: trigger reroute, notify carriers, notify customers, and schedule vehicle swap actions via APIs.")

# End 
