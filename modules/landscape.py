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
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from streamlit_option_menu import option_menu
from modules import data_engine

# --- 1. CACHED DATA LOADER ---
@st.cache_data(show_spinner=False)
def get_pipeline_data(max_points, show_dates):
    """
    Efficiently loads, samples, and pre-calculates pipeline data.
    """
    try:
        prev_app_path = os.path.join("data", "previous_application (sample).csv")
        if not os.path.exists(prev_app_path):
            return None, "File not found"
            
        cols = ['SK_ID_CURR', 'NAME_CONTRACT_STATUS', 'AMT_APPLICATION', 'DAYS_DECISION', 'NAME_CONTRACT_TYPE', 'NAME_PRODUCT_TYPE', 'NAME_CLIENT_TYPE']
        df = pd.read_csv(prev_app_path, usecols=cols)
        
        if len(df) > max_points:
            df = df.sample(n=max_points, random_state=42)
            
        if show_dates:
            df['Plot_X'] = pd.Timestamp.now() + pd.to_timedelta(df['DAYS_DECISION'], unit='D')
        else:
            df['Plot_X'] = df['DAYS_DECISION']
            
        return df, None
    except Exception as e:
        return None, str(e)

# --- 2. POPUP DIALOG ---
@st.dialog("Applicant Profile")
def show_applicant_popup(details, currency):
    st.markdown(f"### {details['NAME_CONTRACT_TYPE']}")
    
    status = details['NAME_CONTRACT_STATUS']
    if status == 'Refused': status_color, status_label = "red", "Declined"
    elif status == 'Approved': status_color, status_label = "green", "Approved"
    else: status_color, status_label = "orange", status
    
    st.markdown(f"**Status:** :{status_color}[{status_label}]")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Loan Amount", f"{currency}{details['AMT_APPLICATION']:,.0f}")
        st.metric("Product", details['NAME_PRODUCT_TYPE'])
    with col2:
        st.metric("Decision Day", f"{details['DAYS_DECISION']} days ago")
        st.metric("Client Type", details['NAME_CLIENT_TYPE'])
        
    st.markdown("---")
    
    if st.button("Open Full Assessment", type="primary", use_container_width=True):
        st.session_state['nav_target'] = "Risk Assessment"
        st.session_state['target_app_id'] = int(details['SK_ID_CURR'])
        st.rerun()

# --- 3. MAIN VIEW ---
def show(df):
    # ---------------------------------------------------------
    # CONFIG
    # ---------------------------------------------------------
    currency = data_engine.get_currency_symbol()
    
    # Initialize variables
    total_apps = 0
    total_apps_full = 0
    pct_shown = 0
    stats_df = pd.DataFrame() 

    # ---------------------------------------------------------
    # STATE & MAIN FILTER
    # ---------------------------------------------------------
    if 'risk_selected_cats' not in st.session_state:
        st.session_state['risk_selected_cats'] = ['CODE_GENDER']
    
    if 'risk_sub_factors' not in st.session_state:
        st.session_state['risk_sub_factors'] = {}

    if 'last_chart_events' not in st.session_state:
        st.session_state['last_chart_events'] = {}

    # --- DATA PREP ---
    if 'CODE_GENDER' in df.columns:
        df['CODE_GENDER'] = df['CODE_GENDER'].replace({'M': 'Male', 'F': 'Female', 'XNA': 'Unknown'})

    # --- TOP SUB-NAVIGATION ---
    selected_sub_tab = option_menu(
        menu_title=None,
        options=["Risk Surface", "Historical Pipeline & Trends"],
        icons=["activity", "funnel"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px 5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#17a2b8"},
        }
    )
    st.markdown("---")

    # ---------------------------------------------------------
    # DYNAMIC SIDEBAR CONTROLS
    # ---------------------------------------------------------
    
    st.sidebar.markdown('<div class="sidebar-section">Filters</div>', unsafe_allow_html=True)
    contract_filter = st.sidebar.multiselect(
        "Contract Type", 
        df['NAME_CONTRACT_TYPE'].unique(), 
        default=df['NAME_CONTRACT_TYPE'].unique(),
        label_visibility="collapsed"
    )
    
    if not contract_filter:
        st.warning("Please select at least one Contract Type.")
        return 
    dff = df[df['NAME_CONTRACT_TYPE'].isin(contract_filter)]

    st.sidebar.markdown('<div class="sidebar-section">Display Options</div>', unsafe_allow_html=True)
    
    if selected_sub_tab == "Risk Surface":
        use_intersection = st.sidebar.toggle("Strict Intersection", value=True, help="Match ALL risk factors vs ANY")
        show_gauges = st.sidebar.toggle("Risk Gauges", value=True)
        show_narrative = st.sidebar.toggle("Narratives", value=False)
        show_bars = st.sidebar.toggle("Detail Bars", value=True)
        
        if st.sidebar.button("Reset Filters", use_container_width=True):
            st.session_state['risk_sub_factors'] = {}
            st.session_state['last_chart_events'] = {}
            st.rerun()
        
    elif selected_sub_tab == "Historical Pipeline & Trends":
        max_points = st.sidebar.slider("Sample Size", 1000, 50000, 5000, step=1000, help="Higher samples may slow down rendering.")
        show_dates = st.sidebar.toggle("Show Calendar Dates", value=True, help="Toggle between Estimated Dates and Relative Days")
        show_insights = st.sidebar.toggle("Show Key Milestones", value=True, help="Overlay insights like Peak Volume")

    # ---------------------------------------------------------
    # VIEW LOGIC
    # ---------------------------------------------------------
    
    # =========================================================
    # VIEW: RISK SURFACE
    # =========================================================
    if selected_sub_tab == "Risk Surface":
        st.subheader("Explore Risk Surface and Exposure by Categories")
        st.caption("Default: **Highest Risk Factor** selected. Click bars to toggle others. Click **↺** to reset category.")
        
        global_mean = dff['TARGET'].mean()
        total_apps = len(df)

        all_cats = ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'CODE_GENDER', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_TYPE_SUITE']
        current_cats = st.session_state['risk_selected_cats']
        available_cats = [c for c in all_cats if c not in current_cats]
        
        for category in current_cats:
            group_stats = dff.groupby(category)['TARGET'].agg(['mean', 'count', 'sum']).reset_index()
            group_stats.columns = [category, 'Default_Rate', 'Count', 'Defaults']
            group_stats = group_stats.sort_values('Default_Rate', ascending=True)
            if category not in st.session_state['risk_sub_factors']:
                highest_risk_factor = group_stats.iloc[-1][category]
                st.session_state['risk_sub_factors'][category] = [highest_risk_factor]

        if use_intersection:
            high_risk_mask = pd.Series([True] * len(dff), index=dff.index)
            logic_label = "Intersection"
        else:
            high_risk_mask = pd.Series([False] * len(dff), index=dff.index)
            logic_label = "Union"

        factors_found = False
        for category in current_cats:
            active = st.session_state['risk_sub_factors'].get(category, [])
            valid = [f for f in active if f in dff[category].unique()]
            if valid:
                if use_intersection: high_risk_mask &= dff[category].isin(valid)
                else: high_risk_mask |= dff[category].isin(valid)
                factors_found = True
            else:
                if use_intersection: high_risk_mask &= False
        
        risk_population = dff[high_risk_mask] if factors_found else pd.DataFrame()
        
        if not risk_population.empty:
            summary = risk_population.groupby(['NAME_CONTRACT_TYPE', 'TARGET']).agg(Count=('TARGET', 'count'), Value=('AMT_CREDIT', 'sum')).reset_index()
            global_credit = df['AMT_CREDIT'].sum()
            global_loss = df[df['TARGET']==1]['AMT_CREDIT'].sum()
            global_loss_rate = global_loss / global_credit if global_credit > 0 else 0
            
            total_val = summary['Value'].sum()
            at_risk_val = summary[summary['TARGET']==1]['Value'].sum()
            segment_loss_rate = at_risk_val / total_val if total_val > 0 else 0
            
            def get_data(ctype, target):
                row = summary[(summary['NAME_CONTRACT_TYPE'] == ctype) & (summary['TARGET'] == target)]
                if row.empty: return 0, 0
                return row.iloc[0]['Count'], row.iloc[0]['Value']

            c0_cnt, c0_val = get_data('Cash loans', 0)
            c1_cnt, c1_val = get_data('Cash loans', 1)
            r0_cnt, r0_val = get_data('Revolving loans', 0)
            r1_cnt, r1_val = get_data('Revolving loans', 1)
            c0_el, c1_el = c0_val * segment_loss_rate, c1_val * segment_loss_rate
            r0_el, r1_el = r0_val * segment_loss_rate, r1_val * segment_loss_rate
            
            needle_deg = max(-90, min(90, (segment_loss_rate * 100 * 1.8) - 90))
            marker_deg = max(-90, min(90, (global_loss_rate * 100 * 1.8) - 90))

            html = []
            html.append('<div class="summary-card">')
            html.append(f'<h4 style="margin-top:0; color:#17a2b8;">High-Risk Segment Analysis ({logic_label})</h4>')
            html.append(f'<p style="font-size: 0.9em; color: #ccc; margin-top: -10px; margin-bottom: 20px;">This segment carries an <b>Expected Loss (EL)</b> of <b>{currency}{at_risk_val:,.0f}</b> based on a historical loss rate of <b>{segment_loss_rate:.1%}</b>.</p>')
            html.append('<div class="flex-row">')
            html.append(f'<div class="flex-col-gauge"><div style="font-size:0.9em; font-weight:bold; color:#ccc; margin-bottom:10px;">Risk Impact (EL)</div><div class="gauge-container"><div class="gauge-body"></div><div class="gauge-mask"></div><div class="gauge-needle" style="transform: rotate({needle_deg}deg);"></div><div class="gauge-marker" style="transform: rotate({marker_deg}deg);"></div><div class="gauge-value">{segment_loss_rate:.1%}</div></div><div style="font-size:0.75em; color:#888; margin-top:5px;"><span style="display:inline-block; width:8px; height:8px; background:#fff; opacity:0.6; margin-right:4px;"></span>Portfolio Avg: {global_loss_rate:.1%}</div></div>')
            html.append('<div class="flex-col-table"><table class="sleek-table"><thead><tr><th style="text-align:left;">Loan Type</th><th colspan="3" style="border-bottom: 2px solid #2ca02c;">Repayers (Target=0)</th><th colspan="3" style="border-bottom: 2px solid #d62728;">Defaulters (Target=1)</th></tr><tr><th></th><th>Count</th><th>Value</th><th title="Expected Loss">Risk Val</th><th>Count</th><th>Value</th><th title="Expected Loss">Risk Val</th></tr></thead><tbody>')
            html.append(f'<tr><td class="risk-label">Cash Loans</td><td class="risk-val val-neutral">{c0_cnt:,.0f}</td><td class="risk-val val-neutral">{currency}{c0_val:,.0f}</td><td class="risk-val val-calc">{currency}{c0_el:,.0f}</td><td class="risk-val val-risk">{c1_cnt:,.0f}</td><td class="risk-val val-risk">{currency}{c1_val:,.0f}</td><td class="risk-val val-calc">{currency}{c1_el:,.0f}</td></tr>')
            html.append(f'<tr><td class="risk-label">Revolving</td><td class="risk-val val-neutral">{r0_cnt:,.0f}</td><td class="risk-val val-neutral">{currency}{r0_val:,.0f}</td><td class="risk-val val-calc">{currency}{r0_el:,.0f}</td><td class="risk-val val-risk">{r1_cnt:,.0f}</td><td class="risk-val val-risk">{currency}{r1_val:,.0f}</td><td class="risk-val val-calc">{currency}{r1_el:,.0f}</td></tr>')
            html.append('</tbody></table></div></div></div>')
            st.markdown("".join(html), unsafe_allow_html=True)
        else:
            if use_intersection: st.info("**No applicants match ALL selected risk criteria.**")
            else: st.info("**No applicants match ANY of the selected risk criteria.**")

        total_slots = len(current_cats)
        if len(current_cats) < 6: total_slots += 1
        cols = st.columns(3)
        
        for i in range(total_slots):
            col_idx = i % 3
            with cols[col_idx]:
                if i < len(current_cats):
                    category = current_cats[i]
                    group_stats = dff.groupby(category)['TARGET'].agg(['mean', 'count', 'sum']).reset_index()
                    group_stats.columns = [category, 'Default_Rate', 'Count', 'Defaults']
                    group_stats = group_stats.sort_values('Default_Rate', ascending=True) 
                    active_factors = st.session_state['risk_sub_factors'].get(category, [])
                    
                    if active_factors:
                        subset_stats = group_stats[group_stats[category].isin(active_factors)]
                        combined_rate = (subset_stats['Default_Rate'] * subset_stats['Count']).sum() / subset_stats['Count'].sum()
                        group_count = subset_stats['Count'].sum()
                        share_of_portfolio = group_count / total_apps if total_apps > 0 else 0
                        risk_lift = combined_rate / global_mean if global_mean > 0 else 0
                        weighted_impact = combined_rate * share_of_portfolio
                        factor_display = f"{len(active_factors)} selected"
                        if len(active_factors) <= 2: factor_display = ", ".join(active_factors)
                    else:
                        combined_rate = 0; risk_lift = 0; weighted_impact = 0; factor_display = "None"

                    if combined_rate < 0.05: story_color = "#2ca02c"
                    elif combined_rate < 0.10: story_color = "#ff9800"
                    else: story_color = "#d62728"

                    if show_narrative:
                        lines = []
                        lines.append(f'<div class="narrative-box" style="border-left-color: {story_color}; font-family: \'Segoe UI\', sans-serif; padding: 10px;">')
                        lines.append(f'<div style="font-size: 1.05em; font-weight: 500; color: #fff; margin-bottom: 4px;">{factor_display}</div>')
                        lines.append(f'<span style="color: {story_color}; font-weight: bold; font-size: 1.2em;">Rate: {combined_rate:.1%}</span>')
                        lines.append(f'<div style="font-size: 0.9em; color: #aaa; margin-top: 2px;">Risk Lift: {risk_lift:.1f}x avg</div>')
                        lines.append('</div>')
                        narrative_html = "".join(lines)

                    if show_gauges:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta", value = combined_rate * 100, 
                            delta = {'reference': global_mean * 100, 'position': "top", 'relative': False, 'valueformat': ".1f"},
                            title = {'text': "Segment Risk", 'font': {'size': 14}}, number = {'suffix': "%", 'font': {'color': story_color}},
                            gauge = {'axis': {'range': [None, 50], 'tickwidth': 1}, 'bar': {'color': story_color}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "gray", 'steps': [{'range': [0, 5], 'color': 'rgba(44, 160, 44, 0.1)'}, {'range': [5, 10], 'color': 'rgba(255, 152, 0, 0.1)'}, {'range': [10, 50], 'color': 'rgba(214, 39, 40, 0.1)'}], 'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': global_mean * 100}}
                        ))
                        fig_gauge.add_annotation(x=0.5, y=0.25, text="vs Portfolio Avg", showarrow=False, font=dict(size=10, color="#888"))
                        fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})

                    if show_bars:
                        colors = ['#d62728' if x in active_factors else '#444' for x in group_stats[category]]
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(y=group_stats[category], x=[1]*len(group_stats), orientation='h', marker_color='rgba(255,255,255,0.05)', hoverinfo='none'))
                        fig_bar.add_trace(go.Bar(y=group_stats[category], x=group_stats['Default_Rate'], orientation='h', marker=dict(color=colors, opacity=1), text=group_stats['Default_Rate'].apply(lambda x: f"{x:.1%}"), textposition='auto', hoverinfo='none', selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
                        fig_bar.update_layout(title=dict(text=f"Breakdown: {category.replace('NAME_', '').title()}", font=dict(size=12)), barmode='overlay', showlegend=False, xaxis=dict(range=[0, max(0.2, group_stats['Default_Rate'].max()*1.2)], visible=False), yaxis=dict(showgrid=False), margin=dict(l=0, r=0, t=30, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', clickmode='event+select', dragmode=False, hovermode=False)

                    with st.container():
                        c_head, c_reset, c_del = st.columns([6, 1, 1])
                        c_head.markdown(f"#### {category.replace('NAME_', '').replace('_', ' ').title()}")
                        if c_reset.button("↺", key=f"reset_{category}", help="Reset to Max Risk"):
                            highest_risk_factor = group_stats.iloc[-1][category]
                            st.session_state['risk_sub_factors'][category] = [highest_risk_factor]
                            st.rerun()
                        if c_del.button("✕", key=f"del_{category}", help="Remove Chart"):
                            st.session_state['risk_selected_cats'].remove(category)
                            st.rerun()

                        if show_narrative: st.markdown(narrative_html, unsafe_allow_html=True)
                        if show_gauges: st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{category}")
                        if show_bars: 
                            selection = st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{category}", on_select="rerun", selection_mode="points")
                            if selection and selection["selection"]["points"]:
                                clicked_idx = selection["selection"]["points"][0]["point_index"]
                                try:
                                    clicked_label = group_stats.iloc[clicked_idx][category]
                                    st.session_state['risk_sub_factors'][category] = [clicked_label]
                                    st.rerun()
                                except: pass
                        st.markdown("---")
                else:
                    with st.container():
                        st.markdown('<h4 style="color: #555;">Add Risk Factor</h4>', unsafe_allow_html=True)
                        st.markdown('<p style="color: #555; font-size: 0.9em;">Compare another category:</p>', unsafe_allow_html=True)
                        for cat in available_cats:
                            btn_label = cat.replace('NAME_', '').replace('_', ' ').title()
                            if st.button(f"{btn_label}", key=f"add_{cat}"):
                                st.session_state['risk_selected_cats'].append(cat)
                                st.rerun()
                        st.markdown("---")

    # =========================================================
    # VIEW: HISTORICAL PIPELINE & TRENDS
    # =========================================================
    elif selected_sub_tab == "Historical Pipeline & Trends":
        st.subheader("Historical Pipeline & Trends")
        st.caption("Visualizing the flow of Approvals, Rejections, and Cancellations over time.")
        
        # 1. LOAD DATA CACHED
        viz_df, err = get_pipeline_data(max_points, show_dates)
        
        if err:
            st.error(f"Error loading pipeline data: {err}")
            return
            
        scatter_mode = go.Scattergl
        total_apps = len(viz_df)
        
        status_counts = viz_df['NAME_CONTRACT_STATUS'].value_counts()
        status_vals = viz_df.groupby('NAME_CONTRACT_STATUS')['AMT_APPLICATION'].sum()
        
        def get_stat(status):
            c = status_counts.get(status, 0)
            v = status_vals.get(status, 0)
            p = (c / total_apps) if total_apps > 0 else 0
            return c, v, p

        ap_c, ap_v, ap_p = get_stat("Approved")
        re_c, re_v, re_p = get_stat("Refused")
        ca_c, ca_v, ca_p = get_stat("Canceled")
        un_c, un_v, un_p = get_stat("Unused offer")

        st.markdown(f"""
        <div class="summary-card" style="padding: 15px; display: flex; justify-content: space-around; text-align: center;">
            <div><div style="font-size:0.8rem; color:#aaa;">SAMPLE SIZE</div><div style="font-size:1.2rem; font-weight:bold; color:#fff;">{total_apps:,}</div></div>
            <div><div style="font-size:0.8rem; color:#2ca02c;">APPROVED</div><div style="font-size:1.2rem; font-weight:bold; color:#2ca02c;">{currency}{ap_v:,.0f}</div><div style="font-size:0.7rem;">{ap_c} apps ({ap_p:.1%})</div></div>
            <div><div style="font-size:0.8rem; color:#d62728;">DECLINED</div><div style="font-size:1.2rem; font-weight:bold; color:#d62728;">{currency}{re_v:,.0f}</div><div style="font-size:0.7rem;">{re_c} apps ({re_p:.1%})</div></div>
            <div><div style="font-size:0.8rem; color:#ffffff;">CANCELED</div><div style="font-size:1.2rem; font-weight:bold; color:#ffffff;">{currency}{ca_v:,.0f}</div><div style="font-size:0.7rem;">{ca_c} apps ({ca_p:.1%})</div></div>
            <div><div style="font-size:0.8rem; color:#ffd700;">UNUSED</div><div style="font-size:1.2rem; font-weight:bold; color:#ffd700;">{currency}{un_v:,.0f}</div><div style="font-size:0.7rem;">{un_c} apps ({un_p:.1%})</div></div>
        </div>
        """, unsafe_allow_html=True)

        fig_pipe = go.Figure()
        fig_pipe.add_hline(y=0, line_width=1, line_color="#555")

        styles = {
            "Approved": {"y_sign": 1, "color": "#2ca02c", "symbol": "circle", "name": "Approved"},
            "Canceled": {"y_sign": 1, "color": "#ffffff", "symbol": "circle", "name": "Canceled"},
            "Refused": {"y_sign": -1, "color": "#d62728", "symbol": "circle", "name": "Declined"},
            "Unused offer": {"y_sign": -1, "color": "#ffd700", "symbol": "circle", "name": "Unused"}
        }

        order = ["Refused", "Approved", "Canceled", "Unused offer"]
        
        for status in order:
            subset = viz_df[viz_df['NAME_CONTRACT_STATUS'] == status]
            if not subset.empty:
                cfg = styles.get(status, {"y_sign": 1, "color": "white", "symbol": "circle", "name": status})
                y_values = subset['AMT_APPLICATION'] * cfg['y_sign']
                name_label = cfg['name']
                fig_pipe.add_trace(scatter_mode(
                    x=subset['Plot_X'], y=y_values, mode='markers', name=name_label,
                    marker=dict(size=np.log1p(subset['AMT_APPLICATION']) * 0.8, color=cfg['color'], opacity=0.7, line=dict(width=0.5, color='black')),
                    text=subset['SK_ID_CURR'],
                    customdata=np.stack((subset['DAYS_DECISION'], subset.index), axis=-1),
                    hovertemplate=f"<b>{name_label}</b><br>ID: %{{text}}<br>Value: {currency}%{{y:,.0f}}<br><extra></extra>"
                ))
        
        if show_insights:
            try:
                daily = viz_df.groupby(['Plot_X'])['AMT_APPLICATION'].sum().reset_index()
                if not daily.empty:
                    max_idx = daily['AMT_APPLICATION'].idxmax()
                    peak_date = daily.iloc[max_idx]['Plot_X']
                    peak_val = daily.iloc[max_idx]['AMT_APPLICATION']
                    
                    # 1. DRAW LINE (Trace)
                    fig_pipe.add_trace(go.Scatter(
                        x=[peak_date, peak_date], y=[0, peak_val * 0.5], 
                        mode='lines+markers', name="Insight",
                        line=dict(color='#00e6ff', width=2, dash='dot'),
                        marker=dict(symbol='diamond', size=12, color='#00e6ff'),
                        showlegend=False, hoverinfo='skip'
                    ))
                    
                    # 2. DRAW LABEL (Annotation)
                    display_date = peak_date.strftime('%b %d, %Y') if show_dates else f"{int(peak_date)} days ago"
                    fig_pipe.add_annotation(
                        x=peak_date, y=peak_val * 0.5,
                        text=f"<b>Peak Volume</b><br>{display_date}<br>{currency}{peak_val:,.0f}",
                        bgcolor="#262730", bordercolor="#00e6ff", opacity=0.9,
                        font=dict(color="#00e6ff", size=11),
                        arrowcolor="#00e6ff", arrowhead=2, ax=0, ay=-40
                    )

                daily_counts = viz_df.groupby(['Plot_X', 'NAME_CONTRACT_STATUS']).size().unstack(fill_value=0)
                if 'Approved' in daily_counts.columns:
                    daily_counts['Total'] = daily_counts.sum(axis=1)
                    significant_days = daily_counts[daily_counts['Total'] > 5] 
                    if not significant_days.empty:
                        significant_days['App_Rate'] = significant_days['Approved'] / significant_days['Total']
                        max_app_date = significant_days['App_Rate'].idxmax()
                        max_app_val = significant_days.loc[max_app_date, 'App_Rate']
                        y_pos = viz_df[(viz_df['Plot_X'] == max_app_date) & (viz_df['NAME_CONTRACT_STATUS'] == 'Approved')]['AMT_APPLICATION'].max()
                        
                        fig_pipe.add_trace(go.Scatter(x=[max_app_date], y=[y_pos], mode='markers', marker=dict(symbol='star', size=15, color='#2ca02c'), showlegend=False, hoverinfo='skip'))
                        
                        display_date_app = max_app_date.strftime('%b %d') if show_dates else f"{int(max_app_date)}d"
                        fig_pipe.add_annotation(
                            x=max_app_date, y=y_pos,
                            text=f"<b>Max Approval Rate</b><br>{display_date_app}<br>{max_app_val:.1%}",
                            bgcolor="#262730", bordercolor="#2ca02c", opacity=0.9,
                            font=dict(color="#2ca02c", size=11),
                            arrowcolor="#2ca02c", arrowhead=2, ax=0, ay=-40
                        )

                if 'Refused' in daily_counts.columns:
                    significant_days['Ref_Rate'] = significant_days['Refused'] / significant_days['Total']
                    max_ref_date = significant_days['Ref_Rate'].idxmax()
                    max_ref_val = significant_days.loc[max_ref_date, 'Ref_Rate']
                    y_pos_ref = -1 * viz_df[(viz_df['Plot_X'] == max_ref_date) & (viz_df['NAME_CONTRACT_STATUS'] == 'Refused')]['AMT_APPLICATION'].max()

                    fig_pipe.add_trace(go.Scatter(x=[max_ref_date], y=[y_pos_ref], mode='markers', marker=dict(symbol='x', size=12, color='#d62728'), showlegend=False, hoverinfo='skip'))
                    
                    display_date_ref = max_ref_date.strftime('%b %d') if show_dates else f"{int(max_ref_date)}d"
                    fig_pipe.add_annotation(
                        x=max_ref_date, y=y_pos_ref,
                        text=f"<b>Max Rejection</b><br>{display_date_ref}<br>{max_ref_val:.1%}",
                        bgcolor="#262730", bordercolor="#d62728", opacity=0.9,
                        font=dict(color="#d62728", size=11),
                        arrowcolor="#d62728", arrowhead=2, ax=0, ay=40
                    )

                daily_avg = viz_df.groupby(['Plot_X'])['AMT_APPLICATION'].mean().reset_index()
                if not daily_avg.empty:
                    max_avg_idx = daily_avg['AMT_APPLICATION'].idxmax()
                    max_avg_date = daily_avg.iloc[max_avg_idx]['Plot_X']
                    max_avg_val = daily_avg.iloc[max_avg_idx]['AMT_APPLICATION']
                    
                    fig_pipe.add_trace(go.Scatter(x=[max_avg_date], y=[max_avg_val], mode='markers', marker=dict(symbol='triangle-up', size=12, color='#ffd700'), showlegend=False, hoverinfo='skip'))
                    
                    display_date_avg = max_avg_date.strftime('%b %d') if show_dates else f"{int(max_avg_date)}d"
                    fig_pipe.add_annotation(
                        x=max_avg_date, y=max_avg_val,
                        text=f"<b>Highest Avg Ticket</b><br>{display_date_avg}<br>{currency}{max_avg_val:,.0f}",
                        bgcolor="#262730", bordercolor="#ffd700", opacity=0.9,
                        font=dict(color="#ffd700", size=11),
                        arrowcolor="#ffd700", arrowhead=2, ax=40, ay=-40
                    )

            except Exception as e: 
                pass

        xaxis_title = "Decision Date" if show_dates else "Days Ago"
        tick_fmt = "%b %Y" if show_dates else None

        fig_pipe.update_layout(
            title="Application Decisions Timeline",
            xaxis=dict(title=xaxis_title, tickformat=tick_fmt),
            yaxis=dict(title=f"Application Amount ({currency})", tickprefix=currency),
            height=650, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=80), dragmode='pan'
        )
        
        selection = st.plotly_chart(fig_pipe, use_container_width=True, on_select="rerun", selection_mode="points", key="pipeline_chart")
    
        if selection and selection["selection"]["points"]:
            point = selection["selection"]["points"][0]
            try:
                if 'customdata' in point:
                    row_idx = point['customdata'][1]
                    applicant_data = viz_df.loc[row_idx]
                    show_applicant_popup(applicant_data, currency)
            except Exception as e: pass

        with st.expander("More information and insights", expanded=False):
            st.markdown("""
            ### Glossary & Insights
            This chart visualizes the flow of historical loan decisions. Each dot represents a single application.
            
            #### Key Milestones
            - **Max Approval Rate:** The day with the highest percentage of successful applications (Approved Count / Total Count). This may indicate policy loosening or high-quality leads.
            - **Max Rejection Rate:** The day with the highest percentage of declined applications. This may signal fraud attacks, bad traffic sources, or strict policy enforcement.
            - **Highest Avg Ticket:** The day with the highest average loan amount requested. Useful for spotting high-net-worth traffic spikes.
            - **Peak Volume:** The single day with the highest total financial value of requests (Sum of Loan Amounts). Essential for liquidity planning.
            """)