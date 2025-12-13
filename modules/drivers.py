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

# --- 1. DATA PREP (CACHED) ---
@st.cache_data(show_spinner=False)
def get_drivers_data(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if 'TARGET' in numeric_df.columns:
        corr = numeric_df.corrwith(df['TARGET']).sort_values(ascending=False)
        corr = corr.drop('TARGET', errors='ignore')
        top_pos = corr.head(10)
        top_neg = corr.tail(10)
        drivers = pd.concat([top_pos, top_neg])
    else:
        drivers = pd.Series()

    if 'DAYS_BIRTH' in df.columns:
        df['AGE'] = (df['DAYS_BIRTH'].abs() / 365).astype(int)
        df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[20, 30, 40, 50, 60, 70], labels=['20-30', '30-40', '40-50', '50-60', '60+'])
        age_risk = df.groupby('AGE_GROUP')['TARGET'].mean().reset_index()
    else:
        age_risk = pd.DataFrame()
    return drivers, age_risk

# --- 2. FAST HISTORY LOADER (CACHED) ---
@st.cache_data(show_spinner=False)
def load_full_history():
    """Loads previous applications into memory once for speed."""
    try:
        path = os.path.join("data", "previous_application (sample).csv")
        if os.path.exists(path):
            # Added timestamps for milestones
            cols = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_APPLICATION', 'NAME_CONTRACT_STATUS', 
                    'DAYS_DECISION', 'NAME_CLIENT_TYPE', 'NAME_PRODUCT_TYPE', 'DAYS_FIRST_DRAWING', 'DAYS_TERMINATION']
            df = pd.read_csv(path, usecols=cols)
            # Create an indexed lookup dictionary or just return DF to filter
            return df
    except: pass
    return pd.DataFrame()

def get_applicant_history(sk_id_curr, full_history_df):
    """Filters the cached dataframe."""
    if full_history_df.empty: return pd.DataFrame()
    history = full_history_df[full_history_df['SK_ID_CURR'] == sk_id_curr].copy()
    return history.sort_values('DAYS_DECISION', ascending=False)

def render_swim_lanes(history_df, current_row, currency):
    """Generates Compact HTML Swim Lanes."""
    
    st.markdown("""
    <style>
        .swim-container {
            font-family: 'Segoe UI', sans-serif;
            margin-top: 15px;
            display: flex;
            flex-direction: column;
            gap: 8px; /* Reduced gap */
        }
        .swim-lane {
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 4px;
            padding: 8px 12px; /* Compact padding */
            border-left: 3px solid #555;
            transition: transform 0.2s;
            height: 55px; /* Fixed compact height */
        }
        .swim-lane:hover {
            background: rgba(255, 255, 255, 0.06);
        }
        .lane-meta {
            width: 140px; /* Reduced width */
            border-right: 1px solid rgba(255,255,255,0.1);
            padding-right: 10px;
            margin-right: 10px;
            flex-shrink: 0;
            line-height: 1.1;
        }
        .lane-title { font-weight: 600; font-size: 0.8rem; color: #eee; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .lane-sub { font-size: 0.7rem; color: #888; margin-top: 2px; }
        
        .lane-timeline {
            flex-grow: 1;
            display: flex;
            align-items: center;
            justify-content: flex-start; /* Left align events */
            gap: 20px;
            position: relative;
            height: 100%;
        }
        /* Connector Line */
        .lane-line {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 1px;
            background: rgba(255,255,255,0.1);
            z-index: 0;
        }
        .lane-event {
            position: relative;
            z-index: 1;
            text-align: center;
            background: #0e1117; 
            padding: 0 8px;
            min-width: 60px;
        }
        .event-icon { font-size: 0.9rem; margin-bottom: 1px; }
        .event-label { font-size: 0.65rem; color: #777; text-transform: uppercase; letter-spacing: 0.5px; }
        .event-val { font-size: 0.7rem; color: #ccc; font-weight: 500; }
        
        /* Status Colors */
        .status-Approved { border-left-color: #2ca02c !important; }
        .status-Refused { border-left-color: #d62728 !important; }
        .status-Canceled { border-left-color: #888 !important; }
        .status-Pending { border-left-color: #ff9800 !important; background: rgba(255, 152, 0, 0.05) !important; }
    </style>
    """, unsafe_allow_html=True)

    html_blocks = []

    # 1. CURRENT APPLICATION (TOP)
    curr_type = current_row.get('NAME_CONTRACT_TYPE', 'Loan')
    curr_amt = current_row.get('AMT_CREDIT', 0)
    
    ext_score = current_row.get('EXT_SOURCE_3', 0.5)
    if pd.isna(ext_score): ext_score = 0.5
    pred_label = "Approve" if ext_score > 0.5 else "Review"
    
    # Flattened HTML
    current_html = f"""<div class="swim-lane status-Pending"><div class="lane-meta"><div class="lane-title">⭐ Current</div><div class="lane-sub">Active Request</div></div><div class="lane-timeline"><div class="lane-line"></div><div class="lane-event"><div class="event-icon">📂</div><div class="event-label">In</div><div class="event-val">{curr_type}</div></div><div class="lane-event"><div class="event-icon">💰</div><div class="event-label">Ask</div><div class="event-val">{currency}{curr_amt:,.0f}</div></div><div class="lane-event"><div class="event-icon">🤖</div><div class="event-label">AI</div><div class="event-val" style="color:#ff9800;">{pred_label}</div></div></div></div>"""
    html_blocks.append(current_html)

    # 2. HISTORICAL APPLICATIONS (BELOW)
    if not history_df.empty:
        for _, row in history_df.iterrows():
            status = row['NAME_CONTRACT_STATUS']
            amt = row['AMT_APPLICATION']
            days = row['DAYS_DECISION']
            ctype = row['NAME_CONTRACT_TYPE']
            
            # Post-Decision Logic
            milestones = []
            
            # M1: Application
            milestones.append(f"""<div class="lane-event"><div class="event-icon">📝</div><div class="event-label">App</div><div class="event-val">{currency}{amt:,.0f}</div></div>""")
            
            # M2: Decision
            if status == "Approved": 
                icon, color = "✅", "#2ca02c"
                milestones.append(f"""<div class="lane-event"><div class="event-icon">{icon}</div><div class="event-label">Dec</div><div class="event-val" style="color:{color}">{status}</div></div>""")
                
                # M3: Disbursed? (If DAYS_FIRST_DRAWING exists and is not 365243)
                first_draw = row.get('DAYS_FIRST_DRAWING', 365243)
                if not pd.isna(first_draw) and first_draw < 365000:
                     milestones.append(f"""<div class="lane-event"><div class="event-icon">💸</div><div class="event-label">Cash</div><div class="event-val">Sent</div></div>""")
                
                # M4: Closed?
                term = row.get('DAYS_TERMINATION', 365243)
                if not pd.isna(term) and term < 365000:
                    milestones.append(f"""<div class="lane-event"><div class="event-icon">🏁</div><div class="event-label">Status</div><div class="event-val">Closed</div></div>""")
                else:
                    milestones.append(f"""<div class="lane-event"><div class="event-icon">⏳</div><div class="event-label">Status</div><div class="event-val">Active</div></div>""")
                    
            elif status == "Refused":
                icon, color = "🛑", "#d62728"
                milestones.append(f"""<div class="lane-event"><div class="event-icon">{icon}</div><div class="event-label">Dec</div><div class="event-val" style="color:{color}">{status}</div></div>""")
                milestones.append(f"""<div class="lane-event"><div class="event-icon">🔒</div><div class="event-label">End</div><div class="event-val">Locked</div></div>""")
                
            else: # Canceled/Unused
                icon, color = "🚫", "#888"
                milestones.append(f"""<div class="lane-event"><div class="event-icon">{icon}</div><div class="event-label">Dec</div><div class="event-val" style="color:{color}">{status}</div></div>""")

            type_icon = "💵" if "Cash" in ctype else "💳" if "Revolving" in ctype else "🛍️"
            status_cls = f"status-{status}" if status in ["Approved", "Refused", "Canceled"] else "status-Pending"

            # Flattened HTML
            hist_html = f"""<div class="swim-lane {status_cls}"><div class="lane-meta"><div class="lane-title">{type_icon} {ctype}</div><div class="lane-sub">{int(abs(days))} days ago</div></div><div class="lane-timeline"><div class="lane-line"></div>{''.join(milestones)}</div></div>"""
            html_blocks.append(hist_html)

    # RENDER ALL
    final_html = f'<div class="swim-container">{"".join(html_blocks)}</div>'
    st.markdown(final_html, unsafe_allow_html=True)

# --- 3. MAIN VIEW ---
def show(df):
    selected_sub = option_menu(
        menu_title=None,
        options=["Key Drivers", "Customer Journey", "Cohort Analysis"],
        icons=["lightning", "signpost-split", "people"],
        default_index=0,
        orientation="horizontal",
        styles={"container": {"padding": "0!important", "background-color": "transparent"}, "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px 5px"}}
    )
    st.markdown("---")

    if selected_sub == "Key Drivers":
        drivers_data, age_data = get_drivers_data(df)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Global Risk Factors")
            fig_corr = go.Figure()
            colors = ['#d62728' if v > 0 else '#2ca02c' for v in drivers_data.values]
            fig_corr.add_trace(go.Bar(y=drivers_data.index, x=drivers_data.values, orientation='h', marker=dict(color=colors)))
            fig_corr.update_layout(height=500, xaxis=dict(title="Correlation"), yaxis=dict(autorange="reversed"), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)
        with c2:
            st.subheader("Demographic Risk")
            if not age_data.empty:
                fig_age = px.bar(age_data, x='AGE_GROUP', y='TARGET', color='TARGET', color_continuous_scale='Reds')
                fig_age.update_layout(height=400, showlegend=False, yaxis=dict(tickformat=".1%", title="Prob"), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_age, use_container_width=True)

    elif selected_sub == "Customer Journey":
        st.subheader("Applicant History & Swim Lanes")
        search_col, _ = st.columns([1, 2])
        default_sk = int(df['SK_ID_CURR'].iloc[0])
        sk_input = search_col.number_input("Enter Applicant ID", value=default_sk, step=1)
        row_set = df[df['SK_ID_CURR'] == sk_input]
        currency = data_engine.get_currency_symbol()
        
        if not row_set.empty:
            current_row = row_set.iloc[0]
            # CACHED LOAD
            full_hist = load_full_history()
            history_df = get_applicant_history(sk_input, full_hist)
            render_swim_lanes(history_df, current_row, currency)
            
            st.write("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Historical Apps", len(history_df))
            m2.metric("Approvals", len(history_df[history_df['NAME_CONTRACT_STATUS'] == 'Approved']) if not history_df.empty else 0)
            m3.metric("Relationship", f"{abs(int(history_df['DAYS_DECISION'].min())):,} days" if not history_df.empty else "New")
        else: st.warning("Applicant not found.")

    elif selected_sub == "Cohort Analysis":
        st.subheader("Risk Cohorts")
        if 'AMT_CREDIT' in df.columns:
            plot_df = df.sample(min(2000, len(df))).copy()
            plot_df['Risk_Level'] = pd.qcut(plot_df['EXT_SOURCE_3'].fillna(0.5), 3, labels=["High", "Med", "Low"])
            fig_3d = px.scatter_3d(plot_df, x='AMT_CREDIT', y='AMT_INCOME_TOTAL', z='AMT_ANNUITY', color='Risk_Level', opacity=0.7)
            fig_3d.update_layout(height=600, margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig_3d, use_container_width=True)