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

import pandas as pd
import numpy as np
import os
import json
import streamlit as st
from datetime import datetime

DATA_PATH = "data/"
SETTINGS_FILE = os.path.join(DATA_PATH, "settings.json")
LOG_FILE = os.path.join(DATA_PATH, "cleaning_log.json")

# ---------------------------------------------------------
# SETTINGS & LOGGING
# ---------------------------------------------------------
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {"last_refresh": "Never", "version": "1.0", "include_unknown": True, "currency": "$"}

def save_setting(key, value):
    settings = load_settings()
    settings[key] = value
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)
    load_data.clear() 

def get_currency_symbol():
    settings = load_settings()
    return settings.get("currency", "$")

def get_cleaning_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return ["No logs available."]

def _save_log(log_list):
    with open(LOG_FILE, 'w') as f:
        json.dump(log_list, f, indent=4)

# ---------------------------------------------------------
# DATA OPERATIONS
# ---------------------------------------------------------
def check_data_status():
    files = {
        "Raw - Application": "application_data (sample).csv",
        "Raw - Previous App": "previous_application (sample).csv",
        "Cleaned - Merged": "radish_merged_data.csv"
    }
    status = {}
    for label, filename in files.items():
        exists = os.path.exists(os.path.join(DATA_PATH, filename))
        status[label] = "✅ Available" if exists else "❌ Missing"
    return status

@st.cache_data(show_spinner=False)
def load_data():
    """
    Loads data and performs LIGHTWEIGHT preprocessing on the fly if needed.
    """
    merged_path = os.path.join(DATA_PATH, "radish_merged_data.csv")
    
    if os.path.exists(merged_path):
        df = pd.read_csv(merged_path)
        
        # --- PERFORMANCE: PRE-CALCULATE COLUMNS ---
        if 'DAYS_BIRTH' in df.columns:
            df['AGE'] = (df['DAYS_BIRTH'].abs() / 365).astype(int)
            
        # Safety Fill for Risk Engine (Renamed variable to avoid conflicts)
        fill_values = {
            'PREV_REFUSAL_RATE': 0.0,
            'PREV_AVG_ANNUITY': 0.0,
            'PREV_AVG_CREDIT': 0.0,
            'EXT_SOURCE_3': 0.5,
            'AMT_INCOME_TOTAL': 0,
            'AMT_CREDIT': 0
        }
        for col_name, fill_val in fill_values.items():
            if col_name in df.columns:
                df[col_name] = df[col_name].fillna(fill_val)

        # Apply Global Preferences
        settings = load_settings()
        if not settings.get("include_unknown", True):
            if 'CODE_GENDER' in df.columns:
                df = df[~df['CODE_GENDER'].isin(['Unknown', 'XNA'])]
            if 'NAME_FAMILY_STATUS' in df.columns:
                df = df[df['NAME_FAMILY_STATUS'] != 'Unknown']
        
        return df
    else:
        return None

def get_applicant_history(sk_id):
    try:
        prev_path = os.path.join(DATA_PATH, "previous_application (sample).csv")
        if os.path.exists(prev_path):
            prev = pd.read_csv(prev_path)
            return prev[prev['SK_ID_CURR'] == sk_id]
        else:
            return pd.DataFrame()
    except:
        return pd.DataFrame()

def process_and_merge():
    """
    HEAVY LIFTING: Reads raw files, cleans, merges history, and saves to CSV.
    """
    load_data.clear()
    log = [f"Pipeline started at {datetime.now().strftime('%H:%M:%S')}"]
    
    try:
        app_path = os.path.join(DATA_PATH, "application_data (sample).csv")
        prev_path = os.path.join(DATA_PATH, "previous_application (sample).csv")
        
        if not os.path.exists(app_path) or not os.path.exists(prev_path):
            log.append("Error: Source files not found.")
            _save_log(log)
            return None

        app = pd.read_csv(app_path)
        prev = pd.read_csv(prev_path)
        log.append(f"Loaded: App ({len(app)}), Prev ({len(prev)})")

        # --- RISK FEATURE ENGINEERING ---
        prev_agg = prev.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'AMT_CREDIT': 'mean',
            'AMT_ANNUITY': 'mean',
            'NAME_CONTRACT_STATUS': lambda x: (x == 'Refused').mean(),
            'DAYS_DECISION': lambda x: x.max()
        }).rename(columns={
            'SK_ID_PREV': 'PREV_APP_COUNT',
            'AMT_CREDIT': 'PREV_AVG_CREDIT',
            'AMT_ANNUITY': 'PREV_AVG_ANNUITY',
            'NAME_CONTRACT_STATUS': 'PREV_REFUSAL_RATE',
            'DAYS_DECISION': 'PREV_DAYS_LAST_DECISION'
        }).reset_index()
        
        df = app.merge(prev_agg, on='SK_ID_CURR', how='left')
        
        out_path = os.path.join(DATA_PATH, "radish_merged_data.csv")
        df.to_csv(out_path, index=False)
        
        save_setting("last_refresh", datetime.now().strftime("%a, %d %b %Y, %H:%M:%S"))
        log.append(f"Merge Complete. Saved to {out_path}")
        _save_log(log)
        
        return df
    except Exception as e:
        log.append(f"Critical Error: {str(e)}")
        _save_log(log)
        return None