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
from PIL import Image, ImageDraw, ImageFont

# Needed by apply_manual_conversion_fallback
import re
import numpy as np

# =========================================
# Import Images/Files (Dataiku)
# =========================================

# Read recipe inputs
Images = dataiku.Folder("pyKiKvku")
Images_info = Images.get_info()

logo_file_name = "medtech_icon.png"
logo_local_path = os.path.join("/tmp", logo_file_name)

with Images.get_download_stream(logo_file_name) as stream:
    with open(logo_local_path, "wb") as out_file:
        out_file.write(stream.read())

jnj_logo = Image.open(logo_local_path)

# =========================================
# Save cookies/session 
# =========================================

# Initialize cookies for browser storage with secure password from environment
cookie_password = os.environ.get('COOKIE_SECRET_KEY', str(uuid.uuid4()))
cookies = cookies_manager.EncryptedCookieManager(prefix="agentic_ai_", password=cookie_password)
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
            # Claim old session
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
            # New secure session
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

# =========================================
# Feedback (Dataiku)
# =========================================

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

deduped_data = deduplicate_across_categories(feedback)

latest_code_entries = {}
for item in deduped_data.get("code", []):
    text = item["text"]; ts = datetime.fromisoformat(item["timestamp"])
    if text not in latest_code_entries or ts > latest_code_entries[text]["timestamp"]:
        latest_code_entries[text] = {"text": text, "timestamp": ts}
latest_code_list = [{"text": v["text"], "timestamp": v["timestamp"].isoformat()} for v in latest_code_entries.values()]
code_feedback_format = "  \n".join([f'{e["text"]} -> code' for e in latest_code_list])

latest_insight_entries = {}
for item in deduped_data.get("insight", []):
    text = item["text"]; ts = datetime.fromisoformat(item["timestamp"])
    if text not in latest_insight_entries or ts > latest_insight_entries[text]["timestamp"]:
        latest_insight_entries[text] = {"text": text, "timestamp": ts}
latest_insight_list = [{"text": v["text"], "timestamp": v["timestamp"].isoformat()} for v in latest_insight_entries.values()]
insight_feedback_format = "  \n".join([f'{e["text"]} -> insight' for e in latest_insight_list])

# =========================================
# Utils/ Helper Functions
# =========================================

def apply_manual_conversion_fallback(df, column_name, target_dtype, sample_data):
    """
    Manual fallback conversion for common data type issues when AI fails.
    """
    df_copy = df.copy()
    if target_dtype == 'float64':
        def clean_numeric(value):
            if pd.isna(value): return value
            value_str = str(value).strip()
            cleaned = re.sub(r'[$€£¥,\s%]', '', value_str)
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            try: return float(cleaned)
            except: return np.nan
        df_copy[column_name] = df_copy[column_name].apply(clean_numeric)
    elif target_dtype == 'int64':
        def clean_integer(value):
            if pd.isna(value): return value
            value_str = str(value).strip()
            cleaned = re.sub(r'[$€£¥,\s%]', '', value_str)
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
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

def _normalize_to_dataframe(obj):
    """Turn common table-shaped outputs into a DataFrame."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # list of dicts
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        # dict of lists/scalars
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    return None

def _should_show_table(user_query: str, df: pd.DataFrame) -> bool:
    """
    Show the full table if the user explicitly asks for it OR
    if it's reasonably small and there isn't an obvious chart to show.
    """
    if df is None:
        return False

    q = (user_query or "").lower()
    explicit_keywords = [
        "table", "tabular", "as a table", "show table", "list out", "summary table",
        "pivot", "cross-tab", "crosstab"
    ]
    if any(k in q for k in explicit_keywords):
        return True

    # Heuristic: small enough to be readable
    return (df.shape[0] <= 200 and df.shape[1] <= 20)

def _render_table(df: pd.DataFrame, key: str):
    """Nice-looking Streamlit table with number formatting and hidden index.
    Ensures index (e.g., Year) is visible as a column."""
    pretty = df.copy()

    # 1) If the index is named or not a simple RangeIndex, bring it back as a column
    if not isinstance(pretty.index, pd.RangeIndex) or getattr(pretty.index, "name", None):
        pretty = pretty.reset_index()

    # 2) Flatten any MultiIndex columns
    if isinstance(pretty.columns, pd.MultiIndex):
        pretty.columns = ["_".join([str(x) for x in tup if x != ""]) for tup in pretty.columns.values]

    # 3) Case-insensitive preferred ordering (Year/Month/Date first if present)
    cols = list(pretty.columns)
    lower_map = {c.lower(): c for c in cols}
    preferred = [lower_map[k] for k in ["year", "month", "date"] if k in lower_map]
    rest = [c for c in cols if c not in preferred]
    pretty = pretty[preferred + rest] if preferred else pretty

    # 4) Ensure numeric formatting
    numeric_cols = pretty.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        pretty[c] = pd.to_numeric(pretty[c], errors="coerce")

    col_cfg = {c: st.column_config.NumberColumn(c, format="%,.2f") for c in numeric_cols}

    st.dataframe(pretty, hide_index=True, use_container_width=True, column_config=col_cfg, key=key)

def _parse_output_intent(user_query: str):
    q = (user_query or "").lower()
    wants_table = any(k in q for k in [
        "table", "tabular", "as a table", "show table", "list out", "summary table",
        "pivot", "cross-tab", "crosstab", "dataframe", "table format", "grid"
    ])
    wants_both  = any(k in q for k in [
        "both", "table and chart", "chart and table", "show both"
    ])
    return wants_table, wants_both

def _render_chart_from_result(result: dict, key_prefix: str = "") -> bool:
    """Render a chart if present; return True if one was rendered."""
    if result.get('plotly_fig') is not None:
        st.plotly_chart(result['plotly_fig'], use_container_width=True,
                        key=f"{key_prefix}plotly_{int(pd.Timestamp.now().value)}")
        return True
    if result.get('plot') is not None:
        st.pyplot(result['plot'])
        return True
    return False

# --- helper: persist chat history uniformly ---
def _save_chat_message(kind: str, text: str):
    """Append a chat message to session state and DB."""
    if 'chat_history' not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({
        'type': kind,
        'query': text,
        'timestamp': pd.Timestamp.now()
    })
    try:
        persistence.save_chat_message(st.session_state.session_id, text, kind)
    except Exception:
        pass

# --- end helper ---

# --- Retry LLMs ---
def run_with_retry(base_df, user_query, data_agent, code_executor, unique_categories, chat_history, max_attempts=3):
    """
    Returns (result_dict, attempts_used)
    result_dict respects CodeExecutor's schema and includes 'ok' boolean.
    """
    attempts = 0

    # 1) First code generation
    code = data_agent.generate_analysis_code(
        base_df,
        user_query,
        chat_history=chat_history,
        unique_categories=unique_categories
    )
    if not code:
        return ({'ok': False, 'error': 'Code generation failed', 'text_output': ''}, attempts)

    while attempts < max_attempts:
        attempts += 1
        res = code_executor.execute_analysis(base_df, code, user_query)
        if res and res.get('ok'):
            # Mark self-repair usage if attempts > 1
            if attempts > 1:
                res['self_repair_used'] = True
            return (res, attempts)

        # Prepare repair prompt using error + final_code from executor
        err = (res or {}).get('error', '')
        tb = (res or {}).get('traceback', '')
        last_code = (res or {}).get('final_code', code)

        repair_prompt = (
            f"{user_query}\n\n"
            f"The previous Python code failed with this error:\n{err}\n\n"
            f"Traceback:\n{tb}\n\n"
            f"Here is the failing code:\n\n{last_code}\n\n"
            "Return a fully corrected Python script ONLY (no markdown). "
            "It must be robust to missing columns and data types, avoid inplace pitfalls, "
            "and execute without errors."
        )
        code = data_agent.generate_analysis_code(
            base_df,
            repair_prompt,
            chat_history=chat_history,
            unique_categories=unique_categories
        )
        if not code:
            break

    return (res or {'ok': False, 'error': 'Exhausted retries'}, attempts)

# --- End Retry LLMs ---

# =========================================
# Streamit App Setup
# =========================================

st.set_page_config(page_title="Natural Language AI for Reporting & Analysis (NARA)", layout="wide")
st.title("🧠 Natural Language AI for Reporting & Analysis (NARA)")
st.markdown("Upload a CSV file and ask questions to analyze your data. Use the inline **Data Types** tool (optional) to fix column types.")

# =========================================
# Session State/Memory
# =========================================

# Always reload chat history for consistency
if 'session_id' in st.session_state:
    chat_history = persistence.get_chat_history(st.session_state.session_id)
    st.session_state.chat_history = chat_history if chat_history else []

# Initialize session state
if 'current_phase' not in st.session_state: st.session_state.current_phase = 'upload'  # 'upload' or 'analysis'
if 'uploaded_files' not in st.session_state: st.session_state.uploaded_files = {}
if 'selected_file' not in st.session_state: st.session_state.selected_file = None
if 'original_df' not in st.session_state: st.session_state.original_df = None
if 'current_df' not in st.session_state: st.session_state.current_df = None
if 'approved_df' not in st.session_state: st.session_state.approved_df = None
if 'analysis_history' not in st.session_state: st.session_state.analysis_history = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'unique_categories' not in st.session_state: st.session_state.unique_categories = {}
if 'feedback_status' not in st.session_state: st.session_state.feedback_status = None

# Initialize agents
data_agent = DataAgent()
code_executor = CodeExecutor()

# =========================================
# Sidebar
# =========================================

with st.sidebar:
    # Display JNJ Logo
    st.image(jnj_logo, width=300)

    st.header("📋 Workflow Status")
    phases = {
        'upload': '📁 Upload Data',
        'analysis': '📊 Data Analysis'
    }
    for phase_key, phase_name in phases.items():
        if st.session_state.current_phase == phase_key:
            st.write(f"**➤ {phase_name}** ✅")
        elif (phase_key == 'analysis' and st.session_state.approved_df is not None):
            st.write(f"   {phase_name} ✅")
        else:
            st.write(f"   {phase_name}")
    st.divider()

# =========================================
# Main content
# =========================================

if st.session_state.current_phase == 'upload':
    st.header("📁 Upload Your Datasets")
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload one or more CSV files to start"
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_files[uploaded_file.name] = df.copy()
                    st.success(f"✅ Loaded {uploaded_file.name} ({len(df)} rows, {len(df.columns)} cols)")
                except Exception as e:
                    st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")

    if st.session_state.uploaded_files:
        st.subheader("📂 Select Dataset to Work With")
        file_options = list(st.session_state.uploaded_files.keys())
        selected_file = st.selectbox(
            "Choose a dataset:",
            options=file_options,
            index=0 if not st.session_state.selected_file else file_options.index(st.session_state.selected_file) if st.session_state.selected_file in file_options else 0
        )

        if st.button("🚀 Start Analysis", type="primary"):
            # Auto-approve and jump straight to analysis
            st.session_state.selected_file = selected_file
            st.session_state.original_df = st.session_state.uploaded_files[selected_file].copy()
            st.session_state.current_df = st.session_state.original_df.copy()
            st.session_state.approved_df = st.session_state.current_df.copy()
            # Build unique categories for AI context
            st.session_state.unique_categories = {}
            for col in st.session_state.current_df.columns:
                if st.session_state.current_df[col].dtype == 'object':
                    unique_vals = st.session_state.current_df[col].unique()
                    if len(unique_vals) <= 50:
                        st.session_state.unique_categories[col] = unique_vals.tolist()
            st.session_state.current_phase = 'analysis'
            persistence.update_session_name(st.session_state.session_id, f"Session - {selected_file}")
            persistence.save_session_state(
                st.session_state.session_id,
                original_df=st.session_state.original_df,
                current_df=st.session_state.current_df,
                approved_df=st.session_state.approved_df,
                phase='analysis'
            )
            st.success(f"✅ {selected_file} ready for analysis")
            st.rerun()

        st.subheader("📊 Dataset Previews")
        for filename, df in st.session_state.uploaded_files.items():
            with st.expander(f"Preview: {filename} ({len(df)} rows, {len(df.columns)} columns)"):
                st.dataframe(df.head(5), use_container_width=True)

elif st.session_state.current_phase == 'analysis':

    with st.sidebar:

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
                        st.success("Session renamed!")
                        st.rerun()

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
                        st.success(f"Switched to: {session['session_name']}")
                        st.rerun()

        if st.button("➕ New Session", key="create_new_session"):
            # Create a fresh session id but keep current dataset & phase
            new_session_id = persistence.create_session(
                session_name=f"New Session {pd.Timestamp.now().strftime('%H:%M')}",
                browser_id=st.session_state.browser_id
            )
            st.session_state.session_id = new_session_id
            cookies['session_id'] = new_session_id
            cookies.save()

            # Keep the dataset and phase as-is (stay in analysis mode)
            # DO NOT reset original_df/current_df/approved_df/selected_file/uploaded_files/unique_categories
            # Only clear chat history
            st.session_state.chat_history = []

            # (Optional) If you also want to clear the analysis history, uncomment the next line:
            st.session_state.analysis_history = []

            # Make sure we persist the current dataset & phase into the new session row
            current_phase = 'analysis' if st.session_state.approved_df is not None else st.session_state.current_phase
            persistence.save_session_state(
                st.session_state.session_id,
                original_df=st.session_state.original_df,
                current_df=st.session_state.current_df,
                approved_df=st.session_state.approved_df,
                phase=current_phase
            )

            st.success("New session created — kept current dataset & analysis mode, cleared chat history.")
            st.rerun()

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
            st.success("Everything has been reset! Starting fresh session.")
            st.rerun()

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
                # Build unique categories for AI context
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

    st.header("📊 Dataset Summary")

    # ---------- Dataset info & table ----------
    if st.session_state.approved_df is not None:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Rows", len(st.session_state.approved_df))
        with col2: st.metric("Columns", len(st.session_state.approved_df.columns))
        with col3: st.metric("Analyses", len(st.session_state.analysis_history))

    #st.subheader("📊 Working Dataset")
    #st.info("This is the dataset used for all analyses. Fix types above if needed, then ask your question below.")
    display_dataframe(st.session_state.approved_df, unique_key="approved")

    # ---------- Inline Data Types Modifications (on Analysis page) ----------
    st.subheader("🔧 Fix Column Data Types (Optional)")
    if st.session_state.approved_df is not None:
        with st.expander("Modify Column Data Types", expanded=False):
            working_df = st.session_state.approved_df.copy()

            col1, col2 = st.columns(2)
            with col1:
                column_to_modify = st.selectbox(
                    "Select column:",
                    options=working_df.columns.tolist(),
                    key="analysis_dtype_col_select"
                )
                current_dtype = str(working_df[column_to_modify].dtype)
                st.text(f"Current type: {current_dtype}")

            with col2:
                dtype_options = {
                    'object': 'Text (object)',
                    'int64': 'Integer (int64)',
                    'float64': 'Decimal (float64)',
                    'datetime64[ns]': 'Date/Time',
                    'bool': 'True/False (bool)',
                    'category': 'Category'
                }
                new_dtype = st.selectbox(
                    "New data type:",
                    options=list(dtype_options.keys()),
                    format_func=lambda x: dtype_options[x],
                    key="analysis_dtype_new_select"
                )

            if st.button("Apply Data Type Change", key="analysis_apply_dtype"):
                df_temp = working_df.copy()
                success = False
                error_message = ""

                try:
                    if new_dtype == 'datetime64[ns]':
                        df_temp[column_to_modify] = pd.to_datetime(df_temp[column_to_modify])
                    elif new_dtype == 'category':
                        df_temp[column_to_modify] = df_temp[column_to_modify].astype('category')
                    elif new_dtype == 'bool':
                        df_temp[column_to_modify] = df_temp[column_to_modify].astype('bool')
                    else:
                        df_temp[column_to_modify] = df_temp[column_to_modify].astype(new_dtype)
                    success = True
                except Exception as default_error:
                    error_message = str(default_error)
                    st.warning(f"⚠️ Default conversion failed: {error_message}")
                    st.info("🤖 Trying AI-powered data type conversion...")

                    try:
                        with st.spinner("Generating intelligent conversion code..."):
                            sample_data = df_temp[column_to_modify].dropna().head(10).tolist()
                            conversion_prompt = f"""Generate Python code to convert a pandas DataFrame column to {new_dtype}.
                            Column name: {column_to_modify}
                            Current data type: {current_dtype}
                            Target data type: {new_dtype}
                            Sample values: {sample_data}
                            Error from standard conversion: {error_message}
                            Requirements:
                            - Use 'df' as the dataframe variable and '{column_to_modify}' as the column name
                            - Handle edge cases; clean data if needed
                            - Return ONLY Python code (no markdown)
                            """
                            generated_code = data_agent.generate_manipulation_code(
                                df_temp, conversion_prompt, chat_history=[], unique_categories={}
                            )
                            if generated_code:
                                with st.expander("🔍 View Generated Conversion Code"):
                                    st.code(generated_code, language="python")
                                result_df = code_executor.execute_manipulation(df_temp, generated_code)
                                if result_df is not None:
                                    df_temp = result_df
                                    success = True
                                    st.success("✅ AI-powered conversion successful!")
                                else:
                                    st.error("❌ AI-generated code failed; attempting manual cleanup...")
                                    try:
                                        df_temp = apply_manual_conversion_fallback(
                                            df_temp, column_to_modify, new_dtype, sample_data
                                        )
                                        success = True
                                        st.success("✅ Manual cleanup conversion successful!")
                                    except Exception as manual_error:
                                        st.error(f"❌ Manual cleanup also failed: {str(manual_error)}")
                            else:
                                st.error("❌ Could not generate conversion code")
                    except Exception as ai_error:
                        st.error(f"❌ AI conversion failed: {str(ai_error)}")

                if success:
                    # Keep both frames in sync so analysis immediately reflects the change
                    st.session_state.approved_df = df_temp.copy()
                    st.session_state.current_df = df_temp.copy()

                    # Rebuild unique_categories (helps downstream prompts)
                    st.session_state.unique_categories = {}
                    for col in st.session_state.approved_df.select_dtypes(include=['object', 'category']).columns:
                        vals = st.session_state.approved_df[col].dropna().unique()
                        if len(vals) <= 50:
                            st.session_state.unique_categories[col] = vals.tolist()

                    # Persist and refresh
                    persistence.save_session_state(
                        st.session_state.session_id,
                        original_df=st.session_state.original_df,
                        current_df=st.session_state.current_df,
                        approved_df=st.session_state.approved_df,
                        phase='analysis'
                    )
                    st.success(f"✅ Changed {column_to_modify} to {dtype_options[new_dtype]}")
                    st.rerun()
                else:
                    st.error("❌ Both default and AI-powered conversions failed")

    

    # ---------- Analysis chat ----------
    st.subheader("💬 AI Assistant for Interactive Data Insights")

    # ---- Show chat history (previous questions) ----
    if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
        with st.expander(f"💬 Chat History ({len(st.session_state.chat_history)} questions)", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                query_text = chat.get('query', '')
                ts = chat.get('timestamp', '')
                try:
                    time_str = ts if isinstance(ts, str) else ts.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    time_str = str(ts)
                st.markdown(f"📊 **Q{len(st.session_state.chat_history)-i}:** {query_text}")
                if time_str:
                    st.caption(f"🕐 {time_str}")
                if i < len(st.session_state.chat_history) - 1:
                    st.divider()

    # Quick actions
    st.write("**Quick Actions:**")
    col1, col2, col3, col4 = st.columns(4)
    quick_action_clicked = False; quick_action_query = ""
    with col1:
        if st.button("📊 Data Summary", key="quick_summary"):
            quick_action_query = "Provide a comprehensive summary of this dataset"
            quick_action_clicked = True
    with col2:
        if st.button("🔗 Find Correlations", key="quick_correlations"):
            quick_action_query = "Show correlation matrix for numeric columns only and identify strongest relationships"
            quick_action_clicked = True
    with col3:
        if st.button("📈 Key Trends", key="quick_trends"):
            quick_action_query = "Identify key trends and patterns in the data with appropriate visualizations"
            quick_action_clicked = True
    with col4:
        if st.button("⚠️ Data Quality", key="quick_quality"):
            quick_action_query = "Analyze data quality issues and missing values"
            quick_action_clicked = True

    st.write("---")
    analysis_query = st.chat_input(
        placeholder="Ask a question (e.g., correlation between age and salary, histogram of sales...)",
        key="analysis_input"
    )
    if quick_action_clicked:
        analysis_query = quick_action_query
        st.info(f"🚀 Processing: {analysis_query}")

    if (analysis_query and analysis_query.strip()) or quick_action_clicked:
        with st.spinner("🤖 Analyzing with AI..."):
            try:
                intent = data_agent.route_query_intelligently(analysis_query, code_feedback_format, insight_feedback_format)
                # record the user's question in chat history (works for both branches)
                _save_chat_message('analysis', analysis_query)
                st.session_state.last_query = analysis_query
                st.session_state.last_intent = intent
                st.session_state.feedback_status = True

                base_df = st.session_state.approved_df if st.session_state.approved_df is not None else st.session_state.current_df

                if intent == "insight":
                    st.info("🧠 Generating strategic business insights...")
                    insights = data_agent.generate_enhanced_insights(base_df, analysis_query)
                    if insights:
                        analysis_result = {'text_output': insights, 'dataframe': None, 'plotly_fig': None, 'plot': None, 'type': 'insight'}
                        st.session_state.analysis_history.append({
                            'query': analysis_query, 'code': None, 'result': analysis_result, 'intent': 'insight', 'timestamp': pd.Timestamp.now()
                        })
                        st.success("✅ Strategic analysis completed!")
                        st.subheader("📊 Business Insights"); st.write(insights)
                        try:
                            from advanced_analysis import generate_insight_narrative_summary
                            summary = generate_insight_narrative_summary(insights, analysis_query)
                            if summary: st.subheader("🎯 Key Takeaways"); st.write(summary)
                        except Exception:
                            pass
                    else:
                        st.error("❌ Failed to generate insights. Please try rephrasing your query.")
                else:
                    st.info("⚙️ Generating and executing analysis code (with auto-repair)...")
                    base_df = st.session_state.approved_df if st.session_state.approved_df is not None else st.session_state.current_df

                    result, attempts_used = run_with_retry(
                        base_df=base_df,
                        user_query=analysis_query,
                        data_agent=data_agent,
                        code_executor=code_executor,
                        unique_categories=st.session_state.unique_categories,
                        chat_history=st.session_state.chat_history,
                        max_attempts=3
                    )

                    #if result and result.get('ok'):
                    if result:
                        st.session_state.analysis_history.append({
                            'query': analysis_query,
                            'code': None,  # keep code hidden per your preference
                            'result': result,
                            'intent': 'code',
                            'timestamp': pd.Timestamp.now()
                        })
                        msg = "✅ Analysis completed successfully!"
                        if result.get('self_repair_used'):
                            msg += f" (auto-repaired in {attempts_used} attempt(s))"
                        st.success(msg)

                        # --- Show the generated / repaired Python code ---
                        final_code = result.get('final_code')
                        if final_code:
                            with st.expander("🔍 View Generated Python Code", expanded=False):
                                st.code(final_code, language="python")

                        # --- AI text (unchanged) ---
                        st.subheader("📊 AI Response")
                        text = result.get('text_output')
                        if text:
                            st.write(text)
                        else:
                            st.info("No written response was produced.")

                        # --- Unified output chooser: show ONE primary output unless user asks for both ---
                        wants_table, wants_both = _parse_output_intent(analysis_query)

                        # Gather possible outputs
                        df_out = result.get('dataframe')
                        if df_out is None:
                            df_out = _normalize_to_dataframe(result.get('result'))

                        has_table = df_out is not None
                        has_chart = (result.get('plotly_fig') is not None) or (result.get('plot') is not None)

                        # Decide what to show
                        if wants_both and (has_table or has_chart):
                            # Show both (table first), only when explicitly requested
                            if has_table:
                                st.subheader("📋 Table")
                                # full table only if it's small or explicitly asked; otherwise preview + download
                                if _should_show_table(analysis_query, df_out):
                                    _render_table(df_out, key=f"analysis_result_{len(st.session_state.analysis_history)}")
                                else:
                                    with st.expander("📋 Table available (preview first rows)"):
                                        _render_table(df_out.head(20), key=f"analysis_result_preview_{len(st.session_state.analysis_history)}")
                                    try:
                                        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                                        st.download_button(
                                            "Download full table as CSV",
                                            data=csv_bytes,
                                            file_name="analysis_result.csv",
                                            mime="text/csv",
                                            key=f"dl_{len(st.session_state.analysis_history)}"
                                        )
                                    except Exception:
                                        pass

                            if has_chart:
                                st.subheader("📈 Chart")
                                _render_chart_from_result(result, key_prefix=f"analysis_chart_")

                        else:
                            # Show a single primary output
                            if wants_table and has_table:
                                st.subheader("📋 Table")
                                if _should_show_table(analysis_query, df_out):
                                    _render_table(df_out, key=f"analysis_result_{len(st.session_state.analysis_history)}")
                                else:
                                    with st.expander("📋 Table available (preview first rows)"):
                                        _render_table(df_out.head(20), key=f"analysis_result_preview_{len(st.session_state.analysis_history)}")
                                    try:
                                        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                                        st.download_button(
                                            "Download full table as CSV",
                                            data=csv_bytes,
                                            file_name="analysis_result.csv",
                                            mime="text/csv",
                                            key=f"dl_{len(st.session_state.analysis_history)}"
                                        )
                                    except Exception:
                                        pass

                            elif has_chart:
                                st.subheader("📈 Chart")
                                _render_chart_from_result(result, key_prefix=f"analysis_chart_")

                            elif has_table:
                                # No chart but we do have a table → show it
                                st.subheader("📋 Table")
                                if _should_show_table(analysis_query, df_out):
                                    _render_table(df_out, key=f"analysis_result_{len(st.session_state.analysis_history)}")
                                else:
                                    with st.expander("📋 Table available (preview first rows)"):
                                        _render_table(df_out.head(20), key=f"analysis_result_preview_{len(st.session_state.analysis_history)}")
                                    try:
                                        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                                        st.download_button(
                                            "Download full table as CSV",
                                            data=csv_bytes,
                                            file_name="analysis_result.csv",
                                            mime="text/csv",
                                            key=f"dl_{len(st.session_state.analysis_history)}"
                                        )
                                    except Exception:
                                        pass

                    else:
                        st.error("❌ Analysis failed after automatic repair attempts for 3 times. Please try rephrasing your request")
                        # Optional, for debugging:
                        # st.caption(result.get('error', ''))
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Please try rephrasing your analysis request.")

    # Prior analysis conversations
    if st.session_state.analysis_history and len(st.session_state.analysis_history) > 1:
        st.divider(); st.subheader("💬 Previous Analysis Conversations")
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[:-1])):
            analysis_num = len(st.session_state.analysis_history) - i - 1
            with st.expander(f"Analysis #{analysis_num}: {analysis['query'][:60]}..." if len(analysis['query']) > 60 else f"Analysis #{analysis_num}: {analysis['query']}"):
                col1, col2 = st.columns([3,1])
                with col1: st.write(f"**Query:** {analysis['query']}")
                with col2: st.write(f"**Time:** {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                if analysis.get('code'):
                    with st.expander("🔍 View Generated Code"): st.code(analysis['code'], language="python")
                if analysis['result']:
                    if analysis['result'].get('text_output'): st.write("**Analysis Summary:**"); st.write(analysis['result']['text_output'])
                    if analysis['result'].get('dataframe') is not None:
                        st.write("**Result Data:**"); display_dataframe(analysis['result']['dataframe'], unique_key=f"history_{i}")
                    if analysis['result'].get('plotly_fig'): st.plotly_chart(analysis['result']['plotly_fig'], use_container_width=True, key=f"history_chart_{i}")
                    elif analysis['result'].get('plot'): st.pyplot(analysis['result']['plot'])

    # ========== ALWAYS-ON FEEDBACK ==========
    last_query = st.session_state.get("last_query")
    last_intent = st.session_state.get("last_intent")
    if st.session_state.feedback_status:
        st.markdown("---"); st.markdown("**Was this AI Response helpful?**")
        with user_feedback.get_download_stream("intention_feedback.json") as stream:
            feedback = json.loads(stream.read().decode("utf-8"))
        col1, col2, col3 = st.columns([1,1,8])
        with col1:
            if st.button("👍", help="This AI response was helpful"):
                try:
                    timestamp_str = datetime.now().isoformat()
                    if last_intent == 'code':
                        feedback['code'].append({'text': last_query, 'timestamp': timestamp_str})
                    else:
                        feedback['insight'].append({'text': last_query, 'timestamp': timestamp_str})
                    with user_feedback.get_writer("intention_feedback.json") as w:
                        w.write(json.dumps(feedback, indent=2).encode("utf-8"))
                    st.toast("Saved to feedback system ✅")
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")
        with col2:
            if st.button("👎", help="This AI response was not helpful"):
                try:
                    timestamp_str = datetime.now().isoformat()
                    if last_intent == 'code':
                        feedback['insight'].append({'text': last_query, 'timestamp': timestamp_str})
                    else:
                        feedback['code'].append({'text': last_query, 'timestamp': timestamp_str})
                    with user_feedback.get_writer("intention_feedback.json") as w:
                        w.write(json.dumps(feedback, indent=2).encode("utf-8"))
                    st.toast("Saved to feedback system ✅")
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")
