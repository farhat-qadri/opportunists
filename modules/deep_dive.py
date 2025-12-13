#    ___    _   ___  _    _    
#   | _ \  /_\ |   \(_)__| |_  
#   |   / / _ \| |) | (_-< ' \ 
#   |_|_\/_/ \_\___/|_/__/_||_| V1
#                               
#   R A D I S H | RISK ANALYSIS DASHBOARD
#   CS4001 DATA VISUALIZATION DESIGN - PROJECT 
#   TEAM OPPORTUNISTS | Shreya Farhat Puneet
#   SEP 2025 
# ----------------------------------------------
#   Support: radish@ds.study.iitm.ac.in
# ----------------------------------------------

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules import data_engine

# --- METADATA FOR FRIENDLY NAMES ---
COLUMN_META = {
    'AMT_INCOME_TOTAL': {'label': 'Total Income', 'desc': 'Total annual income of the applicant.'},
    'AMT_CREDIT': {'label': 'Credit Amount', 'desc': 'Total loan amount requested.'},
    'AMT_ANNUITY': {'label': 'Loan Annuity', 'desc': 'Periodic loan payment amount.'},
    'AMT_GOODS_PRICE': {'label': 'Goods Price', 'desc': 'Price of goods for consumer loans.'},
    'DAYS_BIRTH': {'label': 'Age (Days)', 'desc': 'Age of client in days (negative).'},
    'AGE_YEARS': {'label': 'Age (Years)', 'desc': 'Age of client in years.'},
    'DAYS_EMPLOYED': {'label': 'Employment Length (Days)', 'desc': 'Days employed (negative).'},
    'YEARS_EMPLOYED': {'label': 'Employment Length (Years)', 'desc': 'Years employed.'},
    'YEARS_REGISTRATION': {'label': 'Registration (Years)', 'desc': 'Years since registration changed.'},
    'CNT_CHILDREN': {'label': 'Child Count', 'desc': 'Number of children.'},
    'CNT_FAM_MEMBERS': {'label': 'Family Members', 'desc': 'Count of family members.'},
    'EXT_SOURCE_1': {'label': 'External Score 1', 'desc': 'Normalized score from external data source 1.'},
    'EXT_SOURCE_2': {'label': 'External Score 2', 'desc': 'Normalized score from external data source 2.'},
    'EXT_SOURCE_3': {'label': 'External Score 3', 'desc': 'Normalized score from external data source 3.'},
    'TARGET': {'label': 'Risk Target', 'desc': '0 = Repayer, 1 = Defaulter.'},
    'CODE_GENDER': {'label': 'Gender', 'desc': 'Gender of the client.'},
    'FLAG_OWN_CAR': {'label': 'Car Ownership', 'desc': 'Flag if client owns a car.'},
    'FLAG_OWN_REALTY': {'label': 'Realty Ownership', 'desc': 'Flag if client owns realty.'},
    'NAME_CONTRACT_TYPE': {'label': 'Contract Type', 'desc': 'Cash or Revolving loan.'},
    'NAME_EDUCATION_TYPE': {'label': 'Education', 'desc': 'Level of highest education.'},
    'NAME_INCOME_TYPE': {'label': 'Income Source', 'desc': 'Type of income source.'},
    'NAME_FAMILY_STATUS': {'label': 'Family Status', 'desc': 'Family status of the client.'},
    'NAME_HOUSING_TYPE': {'label': 'Housing Type', 'desc': 'Housing situation.'},
    'OCCUPATION_TYPE': {'label': 'Occupation', 'desc': 'Client occupation.'},
    'ORGANIZATION_TYPE': {'label': 'Organization', 'desc': 'Type of employer organization.'},
}

def get_meta(col):
    return COLUMN_META.get(col, {'label': col.replace('_', ' ').title(), 'desc': 'No description available.'})

def clean_data_for_viz(df):
    df_viz = df.copy()
    magic_num = 365243
    for col in df_viz.select_dtypes(include=[np.number]).columns:
        if 'DAYS' in col:
            mask = (df_viz[col] >= magic_num)
            if mask.any():
                df_viz.loc[mask, col] = np.nan
            df_viz[col] = df_viz[col].abs()

    if 'DAYS_BIRTH' in df_viz.columns:
        df_viz['AGE_YEARS'] = (df_viz['DAYS_BIRTH'] / 365).round(1)
    if 'DAYS_EMPLOYED' in df_viz.columns:
        df_viz['YEARS_EMPLOYED'] = (df_viz['DAYS_EMPLOYED'] / 365).round(1)
    if 'DAYS_REGISTRATION' in df_viz.columns:
        df_viz['YEARS_REGISTRATION'] = (df_viz['DAYS_REGISTRATION'] / 365).round(1)

    return df_viz

def format_discrete_axis(fig, df, col_name, axis_key):
    series = df[col_name].dropna()
    if len(series) == 0: return
    is_integer = np.all(series % 1 == 0)
    distinct_count = series.nunique()
    if is_integer and distinct_count < 15:
        update_dict = {axis_key: dict(type='category', categoryorder='category ascending')}
        fig.update_layout(**update_dict)

def get_semantic_color_map(df, color_var):
    if color_var == "None": return None, None
    if color_var == 'TARGET':
        df['Risk Class'] = df['TARGET'].map({0: 'Repayer (0)', 1: 'Defaulter (1)'})
        return 'Risk Class', {'Repayer (0)': '#2ca02c', 'Defaulter (1)': '#d62728'}
    unique_vals = sorted(df[color_var].dropna().unique())
    if len(unique_vals) == 2:
        if set(unique_vals) == {0, 1}: return color_var, {0: '#2ca02c', 1: '#d62728'} 
        if set(unique_vals) == {'N', 'Y'}: return color_var, {'N': '#2ca02c', 'Y': '#d62728'}
    return color_var, None

def generate_insights(df, vars, color_col):
    insights = []
    currency = data_engine.get_currency_symbol()
    
    if len(vars) == 1:
        col = vars[0]
        label = get_meta(col)['label']
        mean_val = df[col].mean()
        median_val = df[col].median()
        def fmt(v):
            if "AMT" in col or "INCOME" in col: 
                if abs(v) < 100: return f"{currency}{v:,.2f}"
                return f"{currency}{v:,.0f}"
            if "YEARS" in col or "AGE" in col: return f"{v:.1f} yrs"
            return f"{v:,.1f}"
        if mean_val > median_val * 1.1: skew = "right-skewed (high outliers)"
        elif mean_val < median_val * 0.9: skew = "left-skewed"
        else: skew = "relatively balanced"
        insights.append(f"<b>{label}</b> shows a {skew} distribution. The average value is <b>{fmt(mean_val)}</b>, while the median sits at <b>{fmt(median_val)}</b>.")

    if len(vars) == 2:
        l1, l2 = get_meta(vars[0])['label'], get_meta(vars[1])['label']
        corr = df[vars[0]].corr(df[vars[1]])
        direction = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        insights.append(f"There is a <b>{strength} {direction} correlation</b> ({corr:.2f}) between {l1} and {l2}.")
    
    if color_col == 'Risk Class' and len(vars) > 0:
        primary_var = vars[0]
        l1 = get_meta(primary_var)['label']
        means = df.groupby('Risk Class')[primary_var].mean()
        if 'Defaulter (1)' in means and 'Repayer (0)' in means:
            bad_mean = means['Defaulter (1)']
            good_mean = means['Repayer (0)']
            diff_pct = ((bad_mean - good_mean) / good_mean) * 100
            comparison = "higher" if diff_pct > 0 else "lower"
            insights.append(f"On average, Defaulters show <b>{abs(diff_pct):.1f}% {comparison}</b> {l1} compared to Repayers.")

    if not insights: return "Select metrics to see statistical insights."
    return " ".join(insights)

def show(df):
    st.markdown("""
        <style>
            .stMultiSelect > div > div { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255,255,255,0.1); }
            .stSelectbox > div > div { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255,255,255,0.1); }
        </style>
    """, unsafe_allow_html=True)
    
    currency = data_engine.get_currency_symbol()
    plot_df_full = clean_data_for_viz(df)
    
    num_cols = plot_df_full.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = plot_df_full.select_dtypes(exclude=[np.number]).columns.tolist()
    
    default_opts = ['AGE_YEARS', 'AMT_CREDIT', 'YEARS_EMPLOYED', 'AMT_INCOME_TOTAL']
    defaults = [c for c in default_opts if c in num_cols][:2]
    
    st.markdown("### Data & Dimensions Explorer")

    c_sel, c_col = st.columns([3, 1])
    with c_sel:
        selected_vars = st.multiselect("Select Metrics (1-3)", options=num_cols, default=defaults, max_selections=3)
    with c_col:
        color_opts = ["TARGET"] + cat_cols
        color_var = st.selectbox("Select Dimension (Color Grouping)", options=["None"] + color_opts, index=1)

    with st.sidebar:
        st.markdown('<div class="sidebar-section">View Options</div>', unsafe_allow_html=True)
        sample_size = st.slider("Sample Size", 1000, 50000, 5000, step=1000)
        show_insights = st.toggle("Show Insights", value=True)
        disable_dist = len(selected_vars) <= 1
        show_dist = st.toggle("Show Marginal Distributions", value=True, disabled=disable_dist)

    plot_df = plot_df_full.copy()
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)
    
    color_col, color_map = get_semantic_color_map(plot_df, color_var)

    # --- OUTLIER DETECTION ---
    outlier_note = None
    if 'AMT_INCOME_TOTAL' in plot_df.columns and 'AMT_INCOME_TOTAL' in selected_vars:
        limit = plot_df['AMT_INCOME_TOTAL'].quantile(0.995)
        if plot_df['AMT_INCOME_TOTAL'].max() > limit:
            plot_df = plot_df[plot_df['AMT_INCOME_TOTAL'] <= limit]
            outlier_note = f"Note: Top 0.5% income outliers (> {currency}{limit:,.0f}) removed for chart scaling."

    # --- DATA HEALTH ---
    alerts = []
    for var in selected_vars:
        if plot_df[var].sum() == 0:
            alerts.append(f"⚠️ **{get_meta(var)['label']}** contains all ZERO values.")
        elif plot_df[var].isnull().all():
            alerts.append(f"⚠️ **{get_meta(var)['label']}** is completely EMPTY (NaN).")
    if alerts:
        st.warning(" ".join(alerts) + " Please check your source data.")

    if show_insights:
        def get_val_fmt(col_name):
            val = plot_df[col_name].mean()
            if "AMT" in col_name or "INCOME" in col_name: 
                if abs(val) < 100: return f"{currency}{val:,.2f}"
                return f"{currency}{val:,.0f}"
            if "YEARS" in col_name or "AGE" in col_name: return f"{val:.1f} yrs"
            return f"{val:.2f}"
        
        def get_desc_stats(col_name):
            s = plot_df[col_name]
            mn, mx, std = s.min(), s.max(), s.std()
            if "AMT" in col_name: return f"Min: {currency}{mn:,.0f} | Max: {currency}{mx:,.0f} | Std: {currency}{std:,.0f}"
            return f"Min: {mn:,.1f} | Max: {mx:,.1f} | Std: {std:.1f}"

        m1_l, m1_v, m1_d = "Total Records", f"{len(plot_df):,}", ""
        m2_l, m2_v, m2_d = "Metric 1 Avg", "N/A", ""
        m3_l, m3_v, m3_d = "Metric 2 Avg", "N/A", ""

        if len(selected_vars) == 1:
            col_x = selected_vars[0]
            m2_l, m2_v = f"Avg {get_meta(col_x)['label']}", get_val_fmt(col_x)
            m2_d = get_desc_stats(col_x)
            med = plot_df[col_x].median()
            m3_l, m3_v = f"Median {get_meta(col_x)['label']}", f"{med:,.1f}"
            if "AMT" in col_x: m3_v = f"{currency}{med:,.0f}"

        elif len(selected_vars) == 2:
            col_x, col_y = selected_vars[0], selected_vars[1]
            m1_l, m1_v = f"Avg {get_meta(col_x)['label']}", get_val_fmt(col_x)
            m1_d = get_desc_stats(col_x)
            m2_l, m2_v = f"Avg {get_meta(col_y)['label']}", get_val_fmt(col_y)
            m2_d = get_desc_stats(col_y)
            m3_l, m3_v = "Correlation", f"{plot_df[col_x].corr(plot_df[col_y]):.3f}"

        elif len(selected_vars) >= 3:
            col_x, col_y, col_z = selected_vars[0], selected_vars[1], selected_vars[2]
            m1_l, m1_v = f"Avg {get_meta(col_x)['label']}", get_val_fmt(col_x)
            m1_d = get_desc_stats(col_x)
            m2_l, m2_v = f"Avg {get_meta(col_y)['label']}", get_val_fmt(col_y)
            m2_d = get_desc_stats(col_y)
            m3_l, m3_v = f"Avg {get_meta(col_z)['label']}", get_val_fmt(col_z)
            m3_d = get_desc_stats(col_z)

        st.write("")
        with st.expander("Insights", expanded=True):
            s1, s2, s3 = st.columns(3)
            s1.metric(m1_l, m1_v)
            if m1_d: s1.caption(m1_d)
            s2.metric(m2_l, m2_v)
            if m2_d: s2.caption(m2_d)
            s3.metric(m3_l, m3_v)
            if m3_d: s3.caption(m3_d)
            
            if len(selected_vars) > 0:
                st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
                insight_text = generate_insights(plot_df, selected_vars, color_col)
                st.markdown(f"<div style='color: #ccc; font-size: 0.9rem; font-style: italic; opacity: 0.8;'>{insight_text}</div>", unsafe_allow_html=True)

    st.write("") 

    # 4. SPLIT LAYOUT
    has_side_charts = (len(selected_vars) > 1 and show_dist)
    if has_side_charts:
        c_main, c_side = st.columns([2.5, 1])
        main_height = 380 if len(selected_vars) == 2 else 580
    else:
        c_main = st.container()
        c_side = None
        main_height = 500

    with c_main:
        with st.container(border=True):
            if len(selected_vars) == 0:
                st.info("Please select at least one metric above.")
            else:
                title_labels = [get_meta(v)['label'] for v in selected_vars]
                st.markdown(f"<h4 style='margin-bottom:0px; padding-bottom:0px; color:#eee;'>{' vs '.join(title_labels)}</h4>", unsafe_allow_html=True)
                
                if len(selected_vars) == 1:
                    var1 = selected_vars[0]
                    fig = px.histogram(plot_df, x=var1, color=color_col, color_discrete_map=color_map, nbins=50, barmode="overlay", opacity=0.7)
                    format_discrete_axis(fig, plot_df, var1, 'xaxis')
                    fig.update_layout(height=main_height, margin=dict(t=10, l=0, r=0, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title=get_meta(var1)['label'], yaxis_title="Count", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
                    st.plotly_chart(fig, use_container_width=True)

                elif len(selected_vars) == 2:
                    var1, var2 = selected_vars[0], selected_vars[1]
                    fig = px.scatter(plot_df, x=var1, y=var2, color=color_col, color_discrete_map=color_map, opacity=0.6, render_mode='webgl', hover_data=[var1, var2])
                    format_discrete_axis(fig, plot_df, var1, 'xaxis')
                    format_discrete_axis(fig, plot_df, var2, 'yaxis')
                    fig.update_layout(height=main_height, margin=dict(t=10, l=0, r=0, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=get_meta(var1)['label'], showgrid=True, gridcolor='rgba(255,255,255,0.05)'), yaxis=dict(title=get_meta(var2)['label'], showgrid=True, gridcolor='rgba(255,255,255,0.05)'), legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
                    st.plotly_chart(fig, use_container_width=True)

                elif len(selected_vars) == 3:
                    var1, var2, var3 = selected_vars[0], selected_vars[1], selected_vars[2]
                    fig = px.scatter_3d(plot_df, x=var1, y=var2, z=var3, color=color_col, color_discrete_map=color_map, opacity=0.6, hover_data=[var1, var2, var3])
                    scene_dict = dict(xaxis=dict(title=get_meta(var1)['label'], backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)'), yaxis=dict(title=get_meta(var2)['label'], backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)'), zaxis=dict(title=get_meta(var3)['label'], backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)'))
                    for v, ax in zip([var1, var2, var3], ['xaxis', 'yaxis', 'zaxis']):
                        if plot_df[v].nunique() < 15 and np.all(plot_df[v].dropna() % 1 == 0): scene_dict[ax].update(dict(tickmode='linear', dtick=1, tickformat='d'))
                    fig.update_layout(height=main_height, margin=dict(t=0, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)', scene=scene_dict, legend=dict(orientation="h", y=0.9, x=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(selected_vars) == 1:
                    st.caption(f"*{get_meta(selected_vars[0])['desc']}*")
                
                if outlier_note:
                    st.caption(f"⚠️ {outlier_note}")

    if has_side_charts and c_side:
        with c_side:
            for i, var in enumerate(selected_vars):
                with st.container(border=True):
                    var_label = get_meta(var)['label']
                    if color_var != "None":
                        dim_label = get_meta(color_var)['label']
                        header_html = f"{var_label} <span style='opacity:0.6; font-size:0.8em'>x {dim_label}</span>"
                    else:
                        header_html = var_label

                    st.markdown(f"""
                        <div style="line-height: 1.1; margin-bottom: 5px;">
                            <div style="font-weight: 600; font-size: 0.9rem;">{header_html}</div>
                            <div style="font-size: 0.75rem; color: #888; font-style: italic;">{get_meta(var)['desc']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    fig_hist = px.histogram(plot_df, x=var, color=color_col, color_discrete_map=color_map, nbins=30, barmode="overlay")
                    format_discrete_axis(fig_hist, plot_df, var, 'xaxis')
                    
                    # UPDATED: Height 155, Toolbar Disabled, Legend Adjusted
                    fig_hist.update_layout(
                        height=155, 
                        margin=dict(l=0,r=0,t=20,b=0), 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        showlegend=True, 
                        legend=dict(orientation="h", y=1.0, x=1, xanchor="right", font=dict(size=8), title=None, bgcolor='rgba(0,0,0,0)'),
                        xaxis_title=None, 
                        yaxis=dict(showgrid=False, visible=False)
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})