import streamlit as st
import pandas as pd
import io
import traceback
from data_agent import DataAgent
from code_executor import CodeExecutor
from utils import display_dataframe, create_download_link
import altair as alt
import dataiku
import os
import json
from datetime import datetime
import uuid
from persistence import ChatPersistence
import streamlit_cookies_manager as cookies_manager

# Needed by apply_manual_conversion_fallback
import re
import numpy as np

# Initialize cookies for browser storage with secure password from environment
cookie_password = os.environ.get('COOKIE_SECRET_KEY', str(uuid.uuid4()))
cookies = cookies_manager.EncryptedCookieManager(
    prefix="agentic_ai_",
    password=cookie_password
)
if not cookies.ready():
    st.stop()

# Initialize persistence
persistence = ChatPersistence()

# Browser identification for multi-user isolation
cookies_need_save = False
if 'browser_id' not in st.session_state:
    browser_id = cookies.get('browser_id')
    if not browser_id:
        browser_id = str(uuid.uuid4())
        cookies['browser_id'] = browser_id
        cookies_need_save = True
    st.session_state.browser_id = browser_id

# Session recovery
if 'session_id' not in st.session_state:
    cookie_session_id = cookies.get('session_id')
    session_info = persistence.get_session_info(cookie_session_id) if cookie_session_id else None
    if session_info:
        session_browser_id = session_info.get('browser_id')
        if session_browser_id == st.session_state.browser_id:
            st.session_state.session_id = cookie_session_id
            loaded_state = persistence.load_session_state(cookie_session_id)
            if loaded_state:
                st.session_state.original_df = loaded_state.get('original_df')
                st.session_state.current_df = loaded_state.get('current_df')
                st.session_state.approved_df = loaded_state.get('approved_df')
                st.session_state.current_phase = loaded_state.get('phase', 'upload')
            chat_history = persistence.get_chat_history(cookie_session_id)
            st.session_state.chat_history = chat_history if chat_history else []
        elif session_browser_id is None:
            st.session_state.session_id = cookie_session_id
            import sqlite3
            conn = sqlite3.connect(persistence.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE sessions SET browser_id = ? WHERE session_id = ?",
                           (st.session_state.browser_id, cookie_session_id))
            conn.commit(); conn.close()
            loaded_state = persistence.load_session_state(cookie_session_id)
            if loaded_state:
                st.session_state.original_df = loaded_state.get('original_df')
                st.session_state.current_df = loaded_state.get('current_df')
                st.session_state.approved_df = loaded_state.get('approved_df')
                st.session_state.current_phase = loaded_state.get('phase', 'upload')
            chat_history = persistence.get_chat_history(cookie_session_id)
            st.session_state.chat_history = chat_history if chat_history else []
        else:
            session_id = persistence.create_session(browser_id=st.session_state.browser_id)
            st.session_state.session_id = session_id
            cookies['session_id'] = session_id
            cookies_need_save = True
    else:
        session_id = persistence.create_session(browser_id=st.session_state.browser_id)
        st.session_state.session_id = session_id
        cookies['session_id'] = session_id
        cookies_need_save = True

if cookies_need_save:
    cookies.save()

# Read recipe inputs
user_feedback = dataiku.Folder("VZvmcTtt")
user_feedback_info = user_feedback.get_info()
file_name = "intention_feedback.json"

def deduplicate_across_categories(data_dict):
    merged_latest = {}
    for category, items in data_dict.items():
        for item in items:
            text = item["text"]
            ts = datetime.fromisoformat(item["timestamp"])
            if text not in merged_latest or ts > merged_latest[text]["timestamp"]:
                merged_latest[text] = {"text": text, "timestamp": ts, "category": category}
    result = {"code": [], "insight": []}
    for v in merged_latest.values():
        result[v["category"]].append({"text": v["text"], "timestamp": v["timestamp"].isoformat()})
    return result

with user_feedback.get_download_stream("intention_feedback.json") as stream:
    feedback = json.loads(stream.read().decode("utf-8"))

deduped_data  = deduplicate_across_categories(feedback)

latest_code_entries = {}
for item in deduped_data.get("code", []):
    text = item["text"]; ts = datetime.fromisoformat(item["timestamp"])
    if text not in latest_code_entries or ts > latest_code_entries[text]["timestamp"]:
        latest_code_entries[text] = {"text": text, "timestamp": ts}
latest_code_list = [{"text": v["text"], "timestamp": v["timestamp"].isoformat()} for v in latest_code_entries.values()]
code_feedback = [f'{entry["text"]} -> code' for entry in latest_code_list]
code_feedback_format = "  \n".join(code_feedback)

latest_insight_entries = {}
for item in deduped_data.get("insight", []):
    text = item["text"]; ts = datetime.fromisoformat(item["timestamp"])
    if text not in latest_insight_entries or ts > latest_insight_entries[text]["timestamp"]:
        latest_insight_entries[text] = {"text": text, "timestamp": ts}
latest_insight_list = [{"text": v["text"], "timestamp": v["timestamp"].isoformat()} for v in latest_insight_entries.values()]
insight_feedback = [f'{entry["text"]} -> insight' for entry in latest_insight_list]
insight_feedback_format = "  \n".join(insight_feedback)

def apply_manual_conversion_fallback(df, column_name, target_dtype, sample_data):
    df_copy = df.copy()
    if target_dtype == 'float64':
        def clean_numeric(value):
            if pd.isna(value): return value
            value_str = str(value).strip()
            cleaned = re.sub(r'[$€£¥,\s%]', '', value_str)
            if cleaned.startswith('(') and cleaned.endswith(')'): cleaned = '-' + cleaned[1:-1]
            try: return float(cleaned)
            except: return np.nan
        df_copy[column_name] = df_copy[column_name].apply(clean_numeric)
    elif target_dtype == 'int64':
        def clean_integer(value):
            if pd.isna(value): return value
            value_str = str(value).strip()
            cleaned = re.sub(r'[$€£¥,\s%]', '', value_str)
            if cleaned.startswith('(') and cleaned.endswith(')'): cleaned = '-' + cleaned[1:-1]
            try: return int(float(cleaned))
            except: return np.nan
        df_copy[column_name] = df_copy[column_name].apply(clean_integer)
    elif target_dtype == 'datetime64[ns]':
        df_copy[column_name] = pd.to_datetime(df_copy[column_name], errors='coerce')
    elif target_dtype == 'bool':
        def text_to_bool(value):
            if pd.isna(value): return value
            value_str = str(value).lower().strip()
            if value_str in ['true','t','yes','y','1','on']: return True
            if value_str in ['false','f','no','n','0','off']: return False
            return np.nan
        df_copy[column_name] = df_copy[column_name].apply(text_to_bool)
    else:
        df_copy[column_name] = df_copy[column_name].astype(target_dtype)
    return df_copy

# Always reload chat history
if 'session_id' in st.session_state:
    chat_history = persistence.get_chat_history(st.session_state.session_id)
    st.session_state.chat_history = chat_history if chat_history else []

# Initialize session state
if 'current_phase' not in st.session_state: st.session_state.current_phase = 'upload'  # upload, manipulation, analysis
if 'uploaded_files' not in st.session_state: st.session_state.uploaded_files = {}
if 'selected_file' not in st.session_state: st.session_state.selected_file = None
if 'original_df' not in st.session_state: st.session_state.original_df = None
if 'current_df' not in st.session_state: st.session_state.current_df = None
if 'approved_df' not in st.session_state: st.session_state.approved_df = None
if 'manipulation_history' not in st.session_state: st.session_state.manipulation_history = []
if 'analysis_history' not in st.session_state: st.session_state.analysis_history = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'unique_categories' not in st.session_state: st.session_state.unique_categories = {}
if 'feedback_status' not in st.session_state: st.session_state.feedback_status = None

# Initialize agents
data_agent = DataAgent()
code_executor = CodeExecutor()

st.set_page_config(page_title="Natural Language AI for Reporting & Analysis (NARA)", layout="wide")
st.title("🧠 Natural Language AI for Reporting & Analysis (NARA)")
st.markdown("Upload a CSV and ask questions to analyze your data. Use the Data Types section (optional) to fix column types.")

# Sidebar
with st.sidebar:
    st.header("📋 Workflow Status")
    phases = {
        'upload': '📁 Upload Data',
        'manipulation': '🔧 Data Types',
        'analysis': '📊 Data Analysis'
    }
    for phase_key, phase_name in phases.items():
        if st.session_state.current_phase == phase_key:
            st.write(f"**➤ {phase_name}** ✅")
        elif (phase_key == 'manipulation' and st.session_state.original_df is not None) or \
             (phase_key == 'analysis' and st.session_state.approved_df is not None):
            st.write(f"   {phase_name} ✅")
        else:
            st.write(f"   {phase_name}")
    st.divider()

    st.subheader("💾 Session Management")
    current_session_info = persistence.get_session_info(st.session_state.session_id)
    if current_session_info:
        st.write("**Current Session:**")
        st.write(f"📝 {current_session_info['session_name']}")
        with st.expander("✏️ Rename Session"):
            new_name = st.text_input("New session name:", value=current_session_info['session_name'], key="rename_session_input")
            if st.button("💾 Save Name", key="save_session_name"):
                if new_name:
                    persistence.update_session_name(st.session_state.session_id, new_name)
                    st.success("Session renamed!"); st.rerun()

    all_sessions = persistence.get_all_sessions(browser_id=st.session_state.browser_id)
    if len(all_sessions) > 1:
        with st.expander(f"🔄 Switch Session ({len(all_sessions)} total)"):
            for session in all_sessions[:10]:
                is_current = session['session_id'] == st.session_state.session_id
                button_label = f"{'✅ ' if is_current else ''}{session['session_name']}"
                if not is_current and st.button(button_label, key=f"switch_{session['session_id'][:8]}"):
                    st.session_state.session_id = session['session_id']
                    cookies['session_id'] = session['session_id']; cookies.save()
                    loaded_state = persistence.load_session_state(session['session_id'])
                    if loaded_state:
                        st.session_state.original_df = loaded_state.get('original_df')
                        st.session_state.current_df = loaded_state.get('current_df')
                        st.session_state.approved_df = loaded_state.get('approved_df')
                        st.session_state.current_phase = loaded_state.get('phase', 'upload')
                    chat_history = persistence.get_chat_history(session['session_id'])
                    st.session_state.chat_history = chat_history if chat_history else []
                    st.success(f"Switched to: {session['session_name']}"); st.rerun()

    if st.button("➕ New Session", key="create_new_session"):
        new_session_id = persistence.create_session(
            session_name=f"New Session {pd.Timestamp.now().strftime('%H:%M')}",
            browser_id=st.session_state.browser_id
        )
        st.session_state.session_id = new_session_id
        cookies['session_id'] = new_session_id; cookies.save()
        st.session_state.original_df = None
        st.session_state.current_df = None
        st.session_state.approved_df = None
        st.session_state.current_phase = 'upload'
        st.session_state.chat_history = []
        st.session_state.manipulation_history = []
        st.session_state.analysis_history = []
        st.session_state.uploaded_files = {}
        st.session_state.selected_file = None
        st.session_state.unique_categories = {}
        st.success("New session created!"); st.rerun()

    st.divider()
    st.subheader("🔧 Global Actions")
    if st.button("🔄 Reset Everything", type="secondary"):
        browser_id = st.session_state.get('browser_id')
        new_session_id = persistence.create_session(browser_id=browser_id)
        cookies['session_id'] = new_session_id; cookies.save()
        for key in list(st.session_state.keys()):
            if key != 'browser_id':
                del st.session_state[key]
        st.session_state.session_id = new_session_id
        st.success("Everything has been reset! Starting fresh session."); st.rerun()

    # Dataset switcher
    if len(st.session_state.uploaded_files) > 1:
        st.markdown("**📂 Switch Dataset**")
        current_file = st.session_state.get('selected_file', 'None selected')
        st.write(f"Current: {current_file}")
        available_files = list(st.session_state.uploaded_files.keys())
        current_index = available_files.index(current_file) if current_file in available_files else 0
        new_selection = st.selectbox("Select different dataset:", available_files, index=current_index, key="dataset_switcher")
        if st.button("🔄 Switch Dataset") and new_selection != current_file:
            if st.session_state.get('current_df') is not None and st.session_state.get('selected_file'):
                st.info(f"Switching from {st.session_state.selected_file} to {new_selection}")
            # Switch + auto-approve + go to analysis
            st.session_state.selected_file = new_selection
            st.session_state.original_df = st.session_state.uploaded_files[new_selection].copy()
            st.session_state.current_df = st.session_state.original_df.copy()
            st.session_state.approved_df = st.session_state.current_df.copy()
            st.session_state.current_phase = 'analysis'
            st.session_state.manipulation_history = []
            st.session_state.analysis_history = []
            st.session_state.unique_categories = {}
            for col in st.session_state.current_df.select_dtypes(include=['object', 'category']).columns:
                unique_vals = st.session_state.current_df[col].dropna().unique()
                if len(unique_vals) <= 50:
                    st.session_state.unique_categories[col] = unique_vals.tolist()
            st.success(f"Switched to dataset: {new_selection} (ready for analysis)")
            persistence.save_session_state(
                st.session_state.session_id,
                original_df=st.session_state.original_df,
                current_df=st.session_state.current_df,
                approved_df=st.session_state.approved_df,
                phase='analysis'
            )
            st.rerun()

    st.divider()

    # Phase-specific sidebar actions
    if st.session_state.current_phase == 'manipulation
