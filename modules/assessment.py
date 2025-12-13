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
import os
from streamlit_option_menu import option_menu
from modules import data_engine

# --- 0. METADATA DICTIONARY ---
COLUMN_META = {
    'NAME_CONTRACT_TYPE': {'label': 'Loan Type', 'desc': 'Identification if loan is cash or revolving.'},
    'CODE_GENDER': {'label': 'Gender', 'desc': 'Gender of the client.'},
    'FLAG_OWN_CAR': {'label': 'Car Ownership', 'desc': 'Flag indicating if the client owns a car.'},
    'FLAG_OWN_REALTY': {'label': 'Realty Ownership', 'desc': 'Flag indicating if the client owns a house or flat.'},
    'NAME_TYPE_SUITE': {'label': 'Accompaniment', 'desc': 'Who accompanied the client when applying for the loan.'},
    'NAME_INCOME_TYPE': {'label': 'Income Source', 'desc': 'Client\'s income type (e.g., Working, State servant, Pensioner).'},
    'NAME_EDUCATION_TYPE': {'label': 'Education Level', 'desc': 'Level of highest education the client achieved.'},
    'NAME_FAMILY_STATUS': {'label': 'Family Status', 'desc': 'Family status of the client.'},
    'NAME_HOUSING_TYPE': {'label': 'Housing Situation', 'desc': 'Housing situation of the client (e.g., Renting, Living with parents).'},
    'OCCUPATION_TYPE': {'label': 'Job Title', 'desc': 'What kind of occupation does the client have.'},
    'WEEKDAY_APPR_PROCESS_START': {'label': 'Application Day', 'desc': 'Day of the week the client applied for the loan.'},
    'HOUR_APPR_PROCESS_START': {'label': 'Application Hour', 'desc': 'Approximate hour the client applied for the loan.'},
    'ORGANIZATION_TYPE': {'label': 'Employer Industry', 'desc': 'Type of organization where the client works.'},
    'REGION_RATING_CLIENT': {'label': 'Region Rating', 'desc': 'Home Credit rating of the region where the client lives.'},
    'FONDKAPREMONT_MODE': {'label': 'Housing Details', 'desc': 'Normalized information about building where the client lives.'},
    'HOUSETYPE_MODE': {'label': 'House Type', 'desc': 'Specific housing type (e.g., block of flats, terraced house).'},
    'EXT_SOURCE_3_BIN': {'label': 'External Credit Score', 'desc': 'Binned External Score (1=Low, 5=High). Strongest predictor.'}
}

def get_meta(col):
    return COLUMN_META.get(col, {'label': col.replace('_', ' ').title(), 'desc': 'No description available.'})

# --- 1. DATA PREP (CACHED) ---
@st.cache_data(show_spinner=False)
def prep_risk_data(df):
    try:
        if 'DAYS_BIRTH' in df.columns:
            df['AGE'] = (df['DAYS_BIRTH'].abs() / 365).astype(int)

        if 'EXT_SOURCE_3' in df.columns:
            fill_val = df['EXT_SOURCE_3'].median()
            df['EXT_SOURCE_3_BIN'] = pd.qcut(
                df['EXT_SOURCE_3'].fillna(fill_val), 
                q=5, 
                labels=["1 - Very Low", "2 - Low", "3 - Medium", "4 - High", "5 - Very High"]
            ).astype(str)

        req_cols = ['PREV_REFUSAL_RATE', 'PREV_DAYS_LAST_DECISION']
        if not all(col in df.columns for col in req_cols):
            prev_path = os.path.join("data", "previous_application (sample).csv")
            if os.path.exists(prev_path):
                prev_df = pd.read_csv(prev_path)
                prev_agg = prev_df.groupby('SK_ID_CURR').agg({
                    'NAME_CONTRACT_STATUS': lambda x: (x == 'Refused').mean(),
                    'DAYS_DECISION': lambda x: x.max(), 
                    'AMT_CREDIT': 'mean',
                    'AMT_ANNUITY': 'mean'
                }).rename(columns={
                    'NAME_CONTRACT_STATUS': 'PREV_REFUSAL_RATE',
                    'DAYS_DECISION': 'PREV_DAYS_LAST_DECISION',
                    'AMT_CREDIT': 'PREV_AVG_CREDIT',
                    'AMT_ANNUITY': 'PREV_AVG_ANNUITY'
                })
                df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
                
                df['PREV_REFUSAL_RATE'] = df['PREV_REFUSAL_RATE'].fillna(0)
                df['PREV_AVG_ANNUITY'] = df['PREV_AVG_ANNUITY'].fillna(0)
                df['EXT_SOURCE_3'] = df['EXT_SOURCE_3'].fillna(0.5)
        return df
    except: return df

@st.cache_data(show_spinner=False)
def calculate_categorical_signals(df):
    base_cats = df.select_dtypes(include=['object']).columns.tolist()
    for col in ['REGION_RATING_CLIENT', 'HOUR_APPR_PROCESS_START', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EXT_SOURCE_3_BIN']:
        if col in df.columns and col not in base_cats: base_cats.append(col)
    
    exclude = ['SK_ID_CURR', 'AGE_BUCKET'] 
    cat_cols = [c for c in base_cats if c not in exclude and len(df[c].unique()) <= 50]
    signals = []
    
    for col in cat_cols:
        group = df.groupby(col)['TARGET'].mean()
        if len(group) > 1:
            signal_strength = group.max() - group.min()
            largest_group = df[col].value_counts().idxmax()
            largest_group_risk = group[largest_group]
            global_risk = df['TARGET'].mean()
            direction = 1 if largest_group_risk < global_risk else -1
            meta = get_meta(col)
            signals.append({'Feature': str(col), 'Label': meta['label'], 'Desc': meta['desc'], 'Score': signal_strength * direction, 'Max_Risk': group.max(), 'Min_Risk': group.min()})
        
    sig_df = pd.DataFrame(signals)
    if not sig_df.empty:
        pos_sum = sig_df[sig_df['Score'] > 0]['Score'].sum()
        neg_sum = abs(sig_df[sig_df['Score'] < 0]['Score'].sum())
        def normalize(row):
            if row['Score'] > 0: return (row['Score'] / pos_sum) * 50 if pos_sum > 0 else 0
            else: return (row['Score'] / neg_sum) * 50 if neg_sum > 0 else 0
        sig_df['Normalized'] = sig_df.apply(normalize, axis=1)
        sig_df['Abs_Norm'] = sig_df['Normalized'].abs()
        sig_df = sig_df.sort_values('Normalized', ascending=True)
    return sig_df

# --- SWIM LANE HELPERS ---
@st.cache_data(show_spinner=False)
def load_full_history():
    try:
        path = os.path.join("data", "previous_application (sample).csv")
        if os.path.exists(path):
            cols = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_APPLICATION', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 'NAME_CLIENT_TYPE', 'NAME_PRODUCT_TYPE', 'DAYS_FIRST_DRAWING', 'DAYS_TERMINATION']
            return pd.read_csv(path, usecols=cols)
    except: pass
    return pd.DataFrame()

def get_applicant_history(sk_id_curr, full_hist):
    if full_hist.empty: return pd.DataFrame()
    history = full_hist[full_hist['SK_ID_CURR'] == sk_id_curr].copy()
    return history.sort_values('DAYS_DECISION', ascending=False)

def render_swim_lanes_assessment(history_df, current_row, currency):
    st.markdown("""
    <style>
        .swim-box {
            background-color: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 6px;
            padding: 15px;
            margin-top: 5px;
        }
        .swim-container { font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; gap: 6px; }
        .swim-lane { display: flex; align-items: center; background: rgba(255, 255, 255, 0.03); border-radius: 3px; padding: 6px 10px; border-left: 3px solid #555; height: 45px; }
        .lane-meta { width: 120px; border-right: 1px solid rgba(255,255,255,0.1); padding-right: 8px; margin-right: 12px; flex-shrink: 0; line-height: 1.1; }
        .lane-title { font-weight: 600; font-size: 0.75rem; color: #eee; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .lane-sub { font-size: 0.65rem; color: #888; }
        .lane-timeline { flex-grow: 1; display: flex; align-items: center; justify-content: flex-start; gap: 15px; position: relative; height: 100%; }
        .lane-line { position: absolute; top: 50%; left: 0; right: 0; height: 1px; background: rgba(255,255,255,0.15); z-index: 0; }
        .lane-event { position: relative; z-index: 1; text-align: center; background: #0e1117; padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.1); min-width: 50px; }
        .event-val { font-size: 0.7rem; color: #ddd; font-weight: 500; }
        
        .status-Approved { border-left-color: #2ca02c !important; }
        .status-Refused { border-left-color: #d62728 !important; }
        .status-Canceled { border-left-color: #888 !important; }
        .status-Pending { border-left-color: #ff9800 !important; background: rgba(255, 152, 0, 0.05) !important; }
    </style>
    """, unsafe_allow_html=True)

    html_blocks = []
    
    # 1. CURRENT APPLICATION ROW
    curr_type = current_row.get('NAME_CONTRACT_TYPE', 'Loan')
    curr_amt = current_row.get('AMT_CREDIT', 0)
    pred_val = "Approve" if current_row.get('EXT_SOURCE_3', 0.5) > 0.5 else "Audit"
    
    html_blocks.append(f"""
    <div class="swim-lane status-Pending">
        <div class="lane-meta"><div class="lane-title">Current</div><div class="lane-sub">Active</div></div>
        <div class="lane-timeline">
            <div class="lane-line"></div>
            <div class="lane-event"><div class="event-val">{curr_type}</div></div>
            <div class="lane-event"><div class="event-val">{currency}{curr_amt:,.0f}</div></div>
            <div class="lane-event"><div class="event-val" style="color:#ff9800">{pred_val}</div></div>
        </div>
    </div>""")

    # 2. DETAILED HISTORY ROWS
    if not history_df.empty:
        for _, row in history_df.head(10).iterrows(): 
            status = row['NAME_CONTRACT_STATUS']
            amt = row['AMT_APPLICATION']
            days = row['DAYS_DECISION']
            ctype = row['NAME_CONTRACT_TYPE']
            
            milestones = []
            milestones.append(f"""<div class="lane-event"><div class="event-val">{currency}{amt:,.0f}</div></div>""")
            
            dec_color = "#2ca02c" if status == "Approved" else "#d62728" if status == "Refused" else "#888"
            milestones.append(f"""<div class="lane-event"><div class="event-val" style="color:{dec_color}">{status}</div></div>""")
            
            if status == "Approved":
                first_draw = row.get('DAYS_FIRST_DRAWING', 365243)
                term = row.get('DAYS_TERMINATION', 365243)
                
                if not pd.isna(first_draw) and first_draw < 365000:
                    milestones.append(f"""<div class="lane-event"><div class="event-val">Disbursed</div></div>""")
                    if not pd.isna(term) and term < 365000:
                        milestones.append(f"""<div class="lane-event"><div class="event-val" style="color:#aaa">Closed</div></div>""")
                    else:
                        milestones.append(f"""<div class="lane-event"><div class="event-val" style="color:#2ca02c">Active</div></div>""")
            
            elif status == "Refused":
                 milestones.append(f"""<div class="lane-event"><div class="event-val" style="color:#888">Closed</div></div>""")

            status_cls = f"status-{status}" if status in ["Approved", "Refused", "Canceled"] else "status-Pending"
            
            html_blocks.append(f"""
            <div class="swim-lane {status_cls}">
                <div class="lane-meta"><div class="lane-title">{ctype}</div><div class="lane-sub">{int(abs(days))} days ago</div></div>
                <div class="lane-timeline">
                    <div class="lane-line"></div>
                    {''.join(milestones)}
                </div>
            </div>""")

    st.markdown(f'<div class="swim-box"><div class="swim-container">{"".join(html_blocks)}</div></div>', unsafe_allow_html=True)

# --- GRAPH HELPERS ---
def render_risk_card(title, value, max_impact, is_bad=True):
    pct = min(100, (value / max_impact) * 100) if max_impact > 0 else 0
    color = "#d62728" if is_bad else "#2ca02c"
    st.markdown(f"""<div style="margin:5px 0;"><div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#aaa; font-family:Consolas;"><span>Impact</span><span style="color:{color};">{value:.1f} pts</span></div><div style="width:100%; background:rgba(255,255,255,0.1); height:6px; border-radius:3px; margin-top:2px;"><div style="width:{pct}%; background:{color}; height:100%; border-radius:3px; transition:width 0.3s ease;"></div></div></div>""", unsafe_allow_html=True)

def render_comparison_box(df, metric, applicant_val, title, color="#17a2b8"):
    fig = go.Figure()
    try: plot_data = pd.to_numeric(df[metric], errors='coerce').dropna()
    except: plot_data = df[metric]
    
    upper = plot_data.quantile(0.98) if pd.api.types.is_numeric_dtype(plot_data) else None
    if upper: plot_data = plot_data[plot_data <= upper]

    fig.add_trace(go.Box(x=plot_data, name="Portfolio", orientation='h', marker_color='#444', line_color='#666', boxpoints=False, hoverinfo='x'))
    fig.add_trace(go.Scatter(x=[applicant_val], y=["Portfolio"], mode='markers', marker=dict(size=15, color=color, symbol='diamond', line=dict(width=2, color='white')), name="Applicant", hoverinfo='skip'))
    
    if pd.api.types.is_numeric_dtype(plot_data):
        mx = max(upper, applicant_val) * 1.1 if upper else applicant_val * 1.1
        fig.update_xaxes(range=[0, mx])

    fig.update_layout(title=dict(text=title, font=dict(size=11, color="#aaa")), height=100, margin=dict(l=0, r=0, t=25, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(showgrid=False, zeroline=False, visible=True, tickfont=dict(size=9, color="#666")), yaxis=dict(showgrid=False, visible=False))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_system_overview(sig_df):
    st.subheader("System Risk Overview")
    st.caption("Engineering view of all categorical signals plotted by Impact vs. Complexity.")
    
    fig = go.Figure()
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#555")
    fig.add_hline(y=sig_df['Max_Risk'].mean(), line_width=1, line_dash="dash", line_color="#555")
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in sig_df['Normalized']]
    
    fig.add_trace(go.Scatter(
        x=sig_df['Normalized'], y=sig_df['Max_Risk'], mode='markers', 
        text=sig_df['Label'], 
        marker=dict(size=sig_df['Abs_Norm'] * 2 + 10, color=colors, opacity=0.8, line=dict(width=1, color='white')),
        hovertemplate="<b>%{text}</b><br>Signal: %{x:.1f}<br>Max Risk: %{y:.1%}<extra></extra>"
    ))
    
    fig.add_annotation(x=-40, y=sig_df['Max_Risk'].max(), text="CRITICAL RISK DRIVERS", showarrow=False, font=dict(color="#d62728", size=10))
    fig.add_annotation(x=40, y=sig_df['Max_Risk'].max(), text="STABILITY ANCHORS", showarrow=False, font=dict(color="#2ca02c", size=10))

    fig.update_layout(
        height=500, 
        xaxis=dict(title="Signal Impact (Negative = Risk)", range=[-55, 55], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="Peak Risk Intensity", showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickformat='.0%'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, l=50, r=30, b=50), showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("""
    <div style="background-color:rgba(255,255,255,0.02); padding:12px; border-left:3px solid #17a2b8; border-radius:4px; font-size:0.9em; color:#ccc;">
        <b>System Insight:</b> Categories in the top-left quadrant represent high-intensity risk factors that strongly correlate with defaults. 
        Larger bubbles indicate signals with greater statistical variance across sub-groups.
    </div>
    """, unsafe_allow_html=True)

# --- 3. MAIN VIEW ---
def show(df):
    st.markdown("""
        <style>
            .stSelectbox > div > div { background-color: rgba(255, 255, 255, 0.05) !important; color: white !important; border: 1px solid rgba(255,255,255,0.1); }
            .stNumberInput > div > div > input { background-color: rgba(255, 255, 255, 0.05); color: white; }
        </style>
    """, unsafe_allow_html=True)
    
    currency = data_engine.get_currency_symbol()
    
    default_id = int(df['SK_ID_CURR'].iloc[0]) if not df.empty else 0
    if 'target_app_id' in st.session_state:
        target = st.session_state['target_app_id']
        if target in df['SK_ID_CURR'].values: default_id = int(target)
        del st.session_state['target_app_id']

    selected_sub = option_menu(
        menu_title=None,
        options=["Categorical Signals", "Applicant Assessment"],
        icons=["graph-up", "person-badge"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px 5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#17a2b8"},
        }
    )
    st.markdown("---")
    
    # =========================================================
    # TAB 1: CATEGORICAL SIGNALS (SPLIT SCREEN WITH CONTAINERS)
    # =========================================================
    if selected_sub == "Categorical Signals":
        df = prep_risk_data(df.copy())
        
        c_left, c_right = st.columns([1, 1], gap="medium")
        
        # --- LEFT PANEL ---
        with c_left:
            with st.container(border=True): 
                st.subheader("Signal Strength")
                st.caption("Normalized impact of categories. Click a bar to drill down.")
                
                sig_df = calculate_categorical_signals(df)
                sig_df = sig_df.sort_values('Normalized', ascending=True)
                
                sig_df['Visual_Val'] = sig_df['Normalized'].apply(lambda x: 2 if 0 <= x < 2 else (-2 if -2 < x < 0 else x))
                max_val = sig_df['Normalized'].abs().max()
                axis_limit = max(10, max_val * 1.15)
                
                fig_sig = go.Figure()
                fig_sig.add_trace(go.Bar(
                    y=sig_df['Label'], 
                    x=sig_df['Visual_Val'], 
                    orientation='h',
                    marker=dict(color=sig_df['Normalized'], colorscale=['#d62728', '#444', '#2ca02c'], cmid=0),
                    text=sig_df['Normalized'].apply(lambda x: f"{'+' if x>0 else ''}{x:.1f}"),
                    textposition='auto',
                    width=0.7, 
                    hoverinfo='none' 
                ))
                
                fig_sig.update_layout(
                    barmode='relative', height=600,
                    xaxis=dict(title="Signal Weight", range=[-axis_limit, axis_limit], zeroline=True, zerolinewidth=2, zerolinecolor='white'),
                    yaxis=dict(title="", showgrid=False, tickfont=dict(size=11), showticklabels=True),
                    margin=dict(l=0, r=0, t=10, b=30),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    clickmode='event+select', showlegend=False
                )
                
                selected = st.plotly_chart(fig_sig, use_container_width=True, on_select="rerun", selection_mode="points", key="sig_chart", config={'displayModeBar': False})
                
                active_cat = None
                if selected and selected["selection"]["points"]:
                    clicked_idx = selected["selection"]["points"][0]["point_index"]
                    active_cat = sig_df.iloc[clicked_idx]['Feature']

        # --- RIGHT PANEL ---
        with c_right:
            is_hour_cat = active_cat and ("HOUR" in active_cat)
            with st.sidebar:
                st.markdown('<div class="sidebar-section">Chart Controls</div>', unsafe_allow_html=True)
                radio_disabled = (active_cat is None) or is_hour_cat
                def_idx = 0 if radio_disabled else 1
                sort_option = st.radio("Sort Breakdown By:", ["Default", "Risk", "Volume"], index=def_idx, disabled=radio_disabled, horizontal=False)

            with st.container(border=True): 
                if active_cat:
                    meta = get_meta(active_cat)
                    st.subheader(f"Deep Dive: {meta['label']}")
                    st.markdown(f"*{meta['desc']}*")
                    
                    cat_stats = df.groupby(active_cat).agg(Count=('TARGET', 'count'), Risk=('TARGET', 'mean'), Value=('AMT_CREDIT', 'sum')).reset_index()
                    
                    if sort_option == "Risk":
                        cat_stats = cat_stats.sort_values('Risk', ascending=True)
                        x_vals = cat_stats[active_cat].astype(str).tolist()
                    elif sort_option == "Volume":
                        cat_stats = cat_stats.sort_values('Count', ascending=False)
                        x_vals = cat_stats[active_cat].astype(str).tolist()
                    else: 
                        if "WEEKDAY" in active_cat:
                            days = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
                            cat_stats[active_cat] = pd.Categorical(cat_stats[active_cat], categories=days, ordered=True)
                            cat_stats = cat_stats.sort_values(active_cat)
                            x_vals = cat_stats[active_cat].astype(str).tolist()
                        elif "HOUR" in active_cat:
                            cat_stats[active_cat] = pd.to_numeric(cat_stats[active_cat], errors='coerce')
                            cat_stats = cat_stats.sort_values(active_cat)
                            x_vals = cat_stats[active_cat].tolist()
                        elif "EXT_SOURCE_3_BIN" in active_cat:
                             order = ["1 - Very Low", "2 - Low", "3 - Medium", "4 - High", "5 - Very High"]
                             cat_stats[active_cat] = pd.Categorical(cat_stats[active_cat], categories=order, ordered=True)
                             cat_stats = cat_stats.sort_values(active_cat)
                             x_vals = cat_stats[active_cat].astype(str).tolist()
                        else:
                            cat_stats = cat_stats.sort_values(active_cat)
                            x_vals = cat_stats[active_cat].astype(str).tolist()
                    
                    total_n = cat_stats['Count'].sum()
                    avg_risk = cat_stats['Risk'].mean()
                    
                    st.markdown(f"""<div style="background-color:rgba(255,255,255,0.05); padding:15px; border-radius:8px; border:1px solid rgba(255,255,255,0.1); margin-bottom:20px; display:flex; justify-content:space-around; text-align:center;"><div><div style="color:#888; font-size:0.8em;">APPLICANTS</div><div style="font-size:1.2em; font-weight:bold;">{total_n:,}</div></div><div><div style="color:#888; font-size:0.8em;">AVG RISK</div><div style="font-size:1.2em; font-weight:bold; color:#ff9800;">{avg_risk:.1%}</div></div><div><div style="color:#888; font-size:0.8em;">PORTFOLIO VALUE</div><div style="font-size:1.2em; font-weight:bold;">{currency}{cat_stats['Value'].sum()/1e6:.1f}M</div></div></div>""", unsafe_allow_html=True)
                    
                    fig_det = go.Figure()
                    fig_det.add_trace(go.Scatter(x=x_vals, y=cat_stats['Risk'], name="Default Rate", line=dict(color='#ff9800', width=3), mode='lines+markers', yaxis='y2', hovertemplate="<b>%{x}</b><br>Risk: %{y:.1%}<extra></extra>"))
                    fig_det.add_trace(go.Bar(x=x_vals, y=cat_stats['Count'], name="Volume", marker_color='rgba(255,255,255,0.2)', hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>"))
                    
                    layout_args = dict(title="Sub-Category Breakdown", yaxis=dict(title="Volume", showgrid=False), yaxis2=dict(title="Default Risk", overlaying='y', side='right', showgrid=False, tickformat='.0%'), xaxis=dict(showgrid=False, type='category', categoryorder='array', categoryarray=x_vals), legend=dict(orientation="h", y=1.1), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hoverlabel=dict(bgcolor="#222", font_size=13, font_family="Consolas"))
                    if "HOUR" in active_cat: layout_args['xaxis'] = dict(showgrid=False, tickmode='linear', dtick=3)
                    fig_det.update_layout(**layout_args)
                    st.plotly_chart(fig_det, use_container_width=True, config={'displayModeBar': False})
                    
                    highest_risk = cat_stats.loc[cat_stats['Risk'].idxmax()]
                    h_name = str(highest_risk[active_cat]).title()
                    h_rate = highest_risk['Risk']
                    global_mean = df['TARGET'].mean()
                    h_lift = highest_risk['Risk'] / global_mean if global_mean > 0 else 0
                    st.markdown(f"""<div style="background-color:rgba(255,255,255,0.02); padding:12px; border-left:3px solid #ff9800; border-radius:4px; font-size:0.9em; color:#ccc;">Applicants in the <b>{h_name}</b> segment show the highest default likelihood ({h_rate:.1%}), which is <b>{h_lift:.1f}x</b> the portfolio average.</div>""", unsafe_allow_html=True)
                else:
                    render_system_overview(sig_df)

    # =========================================================
    # TAB 2: APPLICANT ASSESSMENT
    # =========================================================
    elif selected_sub == "Applicant Assessment":
        with st.sidebar:
            st.markdown('<div class="sidebar-section">View Options</div>', unsafe_allow_html=True)
            show_profile = st.toggle("Show Applicant Profile", value=True)
            show_rel = st.toggle("Show Relationship (History)", value=True) 
            show_portfolio = st.toggle("Show Portfolio Comparison", value=False)
            
        risk_df = prep_risk_data(df.copy())
        
        c_search, c_reset = st.columns([2, 1])
        with c_search:
            search_id = st.number_input("Search Applicant ID", value=default_id, step=1, format="%d", help="Enter SK_ID_CURR")
        
        cohort = risk_df[risk_df['SK_ID_CURR'] == search_id]
        if cohort.empty:
            st.warning(f"Applicant ID {search_id} not found.")
            return
        row = cohort.iloc[0]
        
        if show_profile:
            st.markdown("#### Applicant Profile")
            with st.container(border=True):
                dp1, dp2, dp3, dp4 = st.columns(4)
                dp1.metric("Age / Gender", f"{row.get('AGE', 'N/A')} / {row.get('CODE_GENDER', 'N/A')}")
                edu = row.get('NAME_EDUCATION_TYPE', 'N/A').replace(" / secondary special", "")
                dp2.metric("Education", edu)
                dp3.metric("Income Type", row.get('NAME_INCOME_TYPE', 'N/A'))
                dp4.metric("Total Income", f"{currency}{row.get('AMT_INCOME_TOTAL', 0):,.0f}")

        # --- APPLICANT RELATIONSHIP (SWIM LANES) ---
        if show_rel:
            st.write("")
            st.markdown("#### Applicant Relationship")
            with st.container(border=True):
                full_hist = load_full_history()
                history_df = get_applicant_history(search_id, full_hist)
                
                # Metrics Top
                total_hist = len(history_df)
                approved_hist = len(history_df[history_df['NAME_CONTRACT_STATUS'] == 'Approved']) if not history_df.empty else 0
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Historical Applications", total_hist)
                m2.metric("Past Approvals", approved_hist)
                m3.metric("Relationship Age", f"{abs(int(history_df['DAYS_DECISION'].min())):,} days" if not history_df.empty else "New Client")
                
                # Render Swim Lanes
                render_swim_lanes_assessment(history_df, row, currency)

        if show_portfolio:
            sample_df = df.sample(5000, random_state=42) if len(df) > 5000 else df
            st.write("")
            with st.expander("Portfolio Comparison (Performance vs Peers)", expanded=True):
                p1, p2, p3 = st.columns(3)
                with p1: render_comparison_box(sample_df, 'AGE', row['AGE'], "Age (Older is Better)", "#ff9800")
                with p2: render_comparison_box(sample_df, 'PREV_REFUSAL_RATE', row['PREV_REFUSAL_RATE'], "Hist. Refusal Rate (Lower is Better)", "#d62728")
                with p3: render_comparison_box(sample_df, 'AMT_INCOME_TOTAL', row['AMT_INCOME_TOTAL'], "Income (Higher is Better)", "#17a2b8")
                p4, p5, p6 = st.columns(3)
                with p4: render_comparison_box(sample_df, 'PREV_AVG_ANNUITY', row['PREV_AVG_ANNUITY'], "Hist. Annuity (Higher is Better)", "#2ca02c")
                with p5: render_comparison_box(sample_df, 'EXT_SOURCE_3', row['EXT_SOURCE_3'], "External Score 3 (Higher is Better)", "#2ca02c")
                with p6: render_comparison_box(sample_df, 'AMT_CREDIT', row['AMT_CREDIT'], "Current Credit Ask (Lower is Safer)", "#17a2b8")

        with c_reset:
            st.write("")
            if st.button("Reset to Actuals", type="secondary", use_container_width=True):
                # Reset all session state variables
                st.session_state['s_age'] = int(row.get('AGE', 30))
                st.session_state['s_ref'] = float(row.get('PREV_REFUSAL_RATE', 0.0))
                st.session_state['s_ann'] = int(row.get('PREV_AVG_ANNUITY', 0))
                
                # Reset all 3 EXT scores
                st.session_state['s_ext1'] = float(row.get('EXT_SOURCE_1', 0.5)) if not pd.isna(row.get('EXT_SOURCE_1')) else 0.5
                st.session_state['s_ext2'] = float(row.get('EXT_SOURCE_2', 0.5)) if not pd.isna(row.get('EXT_SOURCE_2')) else 0.5
                st.session_state['s_ext3'] = float(row.get('EXT_SOURCE_3', 0.5)) if not pd.isna(row.get('EXT_SOURCE_3')) else 0.5
                
                st.session_state['s_cred'] = int(row.get('AMT_CREDIT', 100000))
                st.session_state['s_car'] = (row.get('FLAG_OWN_CAR', 'N') == 'Y')
                
                edu_map = ["Higher education", "Secondary / secondary special", "Lower secondary"]
                curr_edu = row.get('NAME_EDUCATION_TYPE', edu_map[1])
                try: idx_edu = edu_map.index(curr_edu)
                except: idx_edu = 1
                st.session_state['s_edu'] = edu_map[idx_edu]
                
                inc_map = ["Working", "Commercial associate", "State servant", "Pensioner", "Unemployed"]
                curr_inc = row.get('NAME_INCOME_TYPE', "Working")
                try: idx_inc = inc_map.index(curr_inc)
                except: idx_inc = 0
                st.session_state['s_inc'] = inc_map[idx_inc]
                st.rerun()

        st.divider()
        header_area = st.empty()
        base_score = 50.0
        current_risk_adds = 0.0
        current_risk_subs = 0.0
        risk_drivers, risk_mitigants = [], []
        
        if row.get('CODE_GENDER') == 'M':
            risk_drivers.append(("Gender Risk (Male)", 5.0)) 
            current_risk_adds += 5.0

        st.markdown("### RED FLAGS (Risk Drivers)")
        r1, r2, r3, r4 = st.columns(4)
        
        with r1:
            with st.container(border=True):
                st.markdown("**Age Factor**")
                value_age = int(row.get('AGE', 30))
                if 's_age' not in st.session_state: st.session_state['s_age'] = value_age
                in_age = st.slider("Applicant Age", 20, 70, key="s_age")
                impact_age = max(0, (70 - in_age) * 0.3)
                render_risk_card("Age Impact", impact_age, 15, is_bad=True)
                current_risk_adds += impact_age
                if impact_age > 5: risk_drivers.append(("Young Age Profile", impact_age))

        with r2:
            with st.container(border=True):
                st.markdown("**Prev. Refusals**")
                value_ref = float(row.get('PREV_REFUSAL_RATE', 0.0))
                if 's_ref' not in st.session_state: st.session_state['s_ref'] = value_ref
                in_ref = st.slider("Refusal Rate", 0.0, 1.0, step=0.1, format="%.1f", key="s_ref")
                impact_ref = in_ref * 20
                render_risk_card("Refusal Impact", impact_ref, 20, is_bad=True)
                current_risk_adds += impact_ref
                if impact_ref > 5: risk_drivers.append(("Past Refusals", impact_ref))

        with r3:
            with st.container(border=True):
                st.markdown("**Education**")
                opts_edu = ["Higher education", "Secondary / secondary special", "Lower secondary"]
                curr_edu = row.get('NAME_EDUCATION_TYPE', opts_edu[1])
                try: idx_edu = opts_edu.index(curr_edu)
                except: idx_edu = 1
                if 's_edu' not in st.session_state: st.session_state['s_edu'] = opts_edu[idx_edu]
                in_edu = st.selectbox("Level", opts_edu, key="s_edu", label_visibility="visible")
                if in_edu == "Lower secondary": impact_edu = 15.0
                elif in_edu == "Secondary / secondary special": impact_edu = 8.0
                else: impact_edu = 0.0 
                render_risk_card("Edu Impact", impact_edu, 15, is_bad=True)
                current_risk_adds += impact_edu
                if impact_edu > 5: risk_drivers.append(("Education Level", impact_edu))

        with r4:
            with st.container(border=True):
                st.markdown("**Employment**")
                opts_inc = ["Working", "Commercial associate", "State servant", "Pensioner", "Unemployed"]
                curr_inc = row.get('NAME_INCOME_TYPE', "Working")
                try: idx_inc = opts_inc.index(curr_inc)
                except: idx_inc = 0
                if 's_inc' not in st.session_state: st.session_state['s_inc'] = opts_inc[idx_inc]
                in_inc = st.selectbox("Type", opts_inc, key="s_inc")
                if in_inc == "Unemployed": impact_inc = 20.0
                elif in_inc == "Working": impact_inc = 5.0
                else: impact_inc = 0.0
                render_risk_card("Job Impact", impact_inc, 20, is_bad=True)
                current_risk_adds += impact_inc
                if impact_inc > 5: risk_drivers.append(("Employment Status", impact_inc))

        st.write("")
        st.markdown("### GREEN LIGHT BOOSTERS (Mitigants)")

        g1, g2, g3 = st.columns(3)

        with g1:
            with st.container(border=True):
                st.markdown("**History Annuity**")
                value_ann = int(row.get('PREV_AVG_ANNUITY', 0))
                safe_max_ann = max(100000, value_ann * 2)
                if 's_ann' not in st.session_state: st.session_state['s_ann'] = value_ann
                in_ann = st.number_input("Avg Prev Annuity ($)", 0, safe_max_ann, step=1000, key="s_ann")
                boost_ann = min(15.0, (in_ann / 30000) * 15)
                render_risk_card("Annuity Boost", boost_ann, 15, is_bad=False)
                current_risk_subs += boost_ann
                if boost_ann > 5: risk_mitigants.append(("Strong Annuity History", boost_ann))

        # --- UPDATED: 3 EXTERNAL SOURCES SLIDERS ---
        with g2:
            with st.container(border=True):
                st.markdown("**External Credit Scores**")
                
                # Fetch actuals or default to 0.5
                v1 = float(row.get('EXT_SOURCE_1', 0.5)) if not pd.isna(row.get('EXT_SOURCE_1')) else 0.5
                v2 = float(row.get('EXT_SOURCE_2', 0.5)) if not pd.isna(row.get('EXT_SOURCE_2')) else 0.5
                v3 = float(row.get('EXT_SOURCE_3', 0.5)) if not pd.isna(row.get('EXT_SOURCE_3')) else 0.5
                
                # Initialize session state if not present
                if 's_ext1' not in st.session_state: st.session_state['s_ext1'] = v1
                if 's_ext2' not in st.session_state: st.session_state['s_ext2'] = v2
                if 's_ext3' not in st.session_state: st.session_state['s_ext3'] = v3
                
                # 3 Mini Sliders
                s1 = st.slider("Source 1", 0.0, 1.0, step=0.01, key="s_ext1")
                s2 = st.slider("Source 2", 0.0, 1.0, step=0.01, key="s_ext2")
                s3 = st.slider("Source 3", 0.0, 1.0, step=0.01, key="s_ext3")
                
                # Combined Impact Logic (Average)
                avg_score = (s1 + s2 + s3) / 3
                boost_ext = avg_score * 20
                
                render_risk_card("Trust Boost", boost_ext, 20, is_bad=False)
                current_risk_subs += boost_ext
                if boost_ext > 5: risk_mitigants.append(("High External Scores", boost_ext))

        with g3:
            with st.container(border=True):
                st.markdown("**Credit Amount**")
                value_cred = int(row.get('AMT_CREDIT', 100000))
                safe_max_c = max(2000000, value_cred * 2)
                if 's_cred' not in st.session_state: st.session_state['s_cred'] = value_cred
                in_cred = st.number_input("Current Credit ($)", 0, safe_max_c, step=10000, key="s_cred")
                value_car = row.get('FLAG_OWN_CAR', 'N') == 'Y'
                if 's_car' not in st.session_state: st.session_state['s_car'] = value_car
                in_car = st.toggle("Owns Car?", key="s_car")
                boost_car = 10.0 if in_car else 0.0
                render_risk_card("Asset Boost", boost_car, 10, is_bad=False)
                current_risk_subs += boost_car
                if boost_car > 0: risk_mitigants.append(("Asset Ownership", boost_car))

        # --- 6. HEADER RENDER ---
        final_score = max(0, min(100, base_score + current_risk_adds - current_risk_subs))
        est_exposure = in_cred * (final_score / 100)
        
        if final_score < 40:
            theme_color, risk_label, approval_prob = "#2ca02c", "LOW RISK", "High (85%)"
        elif final_score < 70:
            theme_color, risk_label, approval_prob = "#ff9800", "MEDIUM RISK", "Medium (50%)"
        else:
            theme_color, risk_label, approval_prob = "#d62728", "HIGH RISK", "Low (15%)"

        risk_drivers.sort(key=lambda x: x[1], reverse=True)
        risk_mitigants.sort(key=lambda x: x[1], reverse=True)
        
        driver_text = f"Primary risk driven by <b>{risk_drivers[0][0]}</b>." if risk_drivers else "No major risk flags detected."
        mitigant_text = f"Partially offset by <b>{risk_mitigants[0][0]}</b>." if risk_mitigants else "Limited mitigating factors present."
        insight_text = f"{driver_text} {mitigant_text}"

        with header_area.container():
            st.markdown(f"""
            <div style="border: 1px solid rgba(255,255,255,0.1); border-left: 6px solid {theme_color}; background: linear-gradient(90deg, rgba(255,255,255,0.03) 0%, rgba(0,0,0,0) 100%); padding: 20px; border-radius: 8px; margin-bottom: 25px; font-family: 'Segoe UI', sans-serif;">
                <div style="display:flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">Risk Analysis Simulation</div>
                        <div style="font-size: 2.2rem; font-weight: 600; color: #fff; line-height: 1.2;">{risk_label} <span style="font-size:1rem; color:{theme_color}">({final_score:.1f}/100)</span></div>
                        <div style="margin-top: 8px; font-size: 0.95rem; color: #ccc; font-style: italic;">"{insight_text}"</div>
                    </div>
                    <div style="text-align: right; display:flex; gap:30px; margin-top: 5px;">
                        <div>
                            <div style="font-size: 0.8rem; color: #888;">Approval Probability</div>
                            <div style="font-size: 1.2rem; color: #fff; font-weight:500;">{approval_prob}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; color: #888;">Est. Exposure</div>
                            <div style="font-size: 1.2rem; color: #fff; font-weight:500;">{currency}{est_exposure:,.0f}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)