#    ___    _   ___  _    _    
#   | _ \  /_\ |   \(_)__| |_  
#   |   / / _ \| |) | (_-< ' \ 
#   |_|_\/_/ \_\___/|_/__/_||_| V1
#                               
#   R A D I S H | RISK ANALYSIS DASHBOARD
#   MODULE: WHAT-IF SIMULATION
# ----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --- HELPER: BUILD SCORES ---
def make_composite_score(df, feature_cols_raw):
    """Builds a composite Z-score from available columns."""
    feature_cols = [c for c in feature_cols_raw if c in df.columns]
    if not feature_cols: return pd.Series(0, index=df.index)

    tmp = df[feature_cols].copy()
    # Map Y/N to 1/0
    for col in feature_cols:
        if tmp[col].dtype == "object":
            if set(tmp[col].dropna().unique()) <= {"Y", "N"}:
                tmp[col] = tmp[col].map({"Y": 1, "N": 0})
    
    # Standardize
    num_df = tmp.select_dtypes(include=[np.number])
    if num_df.empty: return pd.Series(0, index=df.index)
    
    z = (num_df - num_df.mean()) / num_df.std().replace(0, 1)
    return z.mean(axis=1)

def build_scores(df_raw):
    df = df_raw.copy()
    
    # Define definitions
    definitions = {
        "DEMO_SCORE": ["DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "CODE_GENDER_NUM"],
        "STABILITY_SCORE": ["DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"],
        "INCOME_LOAD_SCORE": ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"],
        "ASSET_SCORE": ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "AMT_GOODS_PRICE", "AMT_CREDIT"],
        "EXT_RISK_SCORE": ["EXT_SOURCE_2", "EXT_SOURCE_3"] 
    }
    
    # Pre-calc ratios if missing
    if "CREDIT_INCOME_RATIO" not in df.columns and "AMT_CREDIT" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
        
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER_NUM"] = df["CODE_GENDER"].map({"F": 0, "M": 1, "Female": 0, "Male": 1}).fillna(0)

    # Build
    score_map = {}
    for name, cols in definitions.items():
        df[name] = make_composite_score(df, cols)
        score_map[name] = cols
        
    # Invert Ext Risk (Higher Ext Score = Lower Risk, so invert for Risk Score)
    df["EXT_RISK_SCORE"] = -df["EXT_RISK_SCORE"]
    
    return df, score_map

# --- MAIN VIEW ---
def show(df):
    st.title("🔮 What-If Simulation Model")
    st.markdown("Train a predictive model in real-time and simulate scenarios using composite risk scores.")

    if 'TARGET' not in df.columns:
        st.error("Data requires a 'TARGET' column for modeling.")
        return

    # 1. PREPARE DATA
    with st.spinner("Building Risk Scores & Training Model..."):
        model_df, score_map = build_scores(df)
        
        score_features = list(score_map.keys())
        train_df = model_df.dropna(subset=score_features + ['TARGET'])
        
        if len(train_df) < 100:
            st.error("Not enough data to train model.")
            return

        X = train_df[score_features]
        y = train_df['TARGET']
        
        # Train
        logit = LogisticRegression(class_weight='balanced', max_iter=1000)
        logit.fit(X, y)
        auc = roc_auc_score(y, logit.predict_proba(X)[:, 1])
        
        coef_series = pd.Series(logit.coef_[0], index=score_features)
        intercept = logit.intercept_[0]

    # 2. UI LAYOUT
    col_info, col_sim = st.columns([1, 2])
    
    with col_info:
        st.success(f"Model Trained (AUC: {auc:.3f})")
        st.info("Adjust the sliders on the right to simulate a theoretical applicant.")
        
        st.markdown("### 🎛️ Risk Drivers")
        # Contribution Chart
        df_coef = coef_series.reset_index().rename(columns={'index': 'Score', 0: 'Impact'})
        fig_coef = px.bar(
            df_coef, x='Impact', y='Score', orientation='h',
            color='Impact', color_continuous_scale="RdBu_r",
            title="Feature Importance (Log-Odds)"
        )
        fig_coef.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_coef, use_container_width=True)

    with col_sim:
        st.subheader("🧪 Scenario Simulator")
        st.markdown('<div class="summary-card">Adjust composite scores to see Prob. of Default</div>', unsafe_allow_html=True)
        
        # Scenario Sliders
        scenario = {}
        cols = st.columns(2)
        
        # Calculate defaults based on mean
        defaults = X.mean()
        
        for idx, feat in enumerate(score_features):
            with cols[idx % 2]:
                min_v, max_v = X[feat].min(), X[feat].max()
                scenario[feat] = st.slider(
                    f"{feat}", 
                    min_value=float(min_v), max_value=float(max_v), 
                    value=float(defaults[feat])
                )

        # Predict
        input_data = pd.DataFrame([scenario])
        logit_score = intercept + np.dot(coef_series.values, input_data.iloc[0].values)
        prob = 1 / (1 + np.exp(-logit_score))
        
        # Result Gauge
        st.markdown("---")
        c_res1, c_res2 = st.columns([2, 1])
        with c_res1:
            st.metric("Simulated Default Probability", f"{prob:.2%}")
            st.progress(prob)
        with c_res2:
            risk_label = "EXTREME" if prob > 0.5 else ("HIGH" if prob > 0.2 else "LOW")
            color = "red" if prob > 0.2 else "green"
            st.markdown(f"### :{color}[{risk_label}]")

    # 3. BUBBLE CHART (FROM TEAMMATE)
    st.markdown("---")
    st.subheader("Deep Dive: 3D Relationship")
    
    b_col1, b_col2, b_col3 = st.columns(3)
    x_axis = b_col1.selectbox("X Axis", score_features, index=0)
    y_axis = b_col2.selectbox("Y Axis", score_features, index=1)
    size_axis = b_col3.selectbox("Size (Z Axis)", score_features, index=2)
    
    # Sample for speed
    plot_df = train_df.sample(min(2000, len(train_df))).copy()
    
    # --- FIX FOR NEGATIVE SIZES ---
    # Plotly crashes if size is negative. We shift values to be positive [1, 20]
    min_val = plot_df[size_axis].min()
    plot_df['Viz_Size'] = plot_df[size_axis] - min_val + 1  # Shift so min is 1
    
    fig_bub = px.scatter(
        plot_df, x=x_axis, y=y_axis, 
        size='Viz_Size',   # Use the safe, positive column for size
        color='TARGET',
        color_continuous_scale=['#2ca02c', '#d62728'],
        size_max=30, opacity=0.6,
        title=f"Interaction: {x_axis} vs {y_axis}",
        hover_data={size_axis: True, 'Viz_Size': False} # Show real value on hover
    )
    st.plotly_chart(fig_bub, use_container_width=True)