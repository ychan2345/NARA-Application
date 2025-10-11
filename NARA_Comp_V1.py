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

# Initialize cookies for browser storage with secure password from environment
import os
cookie_password = os.environ.get('COOKIE_SECRET_KEY', str(uuid.uuid4()))  # Use env var or generate unique per deployment
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
    # Try to get browser_id from cookies
    browser_id = cookies.get('browser_id')
    if not browser_id:
        # Create new browser_id for this browser
        browser_id = str(uuid.uuid4())
        cookies['browser_id'] = browser_id
        cookies_need_save = True
    st.session_state.browser_id = browser_id

# Session recovery: Check for existing session in cookies or create new
if 'session_id' not in st.session_state:
    # Try to get session from cookies
    cookie_session_id = cookies.get('session_id')
    session_info = persistence.get_session_info(cookie_session_id) if cookie_session_id else None
    
    if session_info:
        # Session exists - verify browser_id matches for security
        session_browser_id = session_info.get('browser_id')
        
        if session_browser_id == st.session_state.browser_id:
            # Valid session for this browser - load it
            st.session_state.session_id = cookie_session_id
            
            # Load session state from database
            loaded_state = persistence.load_session_state(cookie_session_id)
            if loaded_state:
                st.session_state.original_df = loaded_state.get('original_df')
                st.session_state.current_df = loaded_state.get('current_df')
                st.session_state.approved_df = loaded_state.get('approved_df')
                st.session_state.current_phase = loaded_state.get('phase', 'upload')
            
            # Load chat history
            chat_history = persistence.get_chat_history(cookie_session_id)
            st.session_state.chat_history = chat_history if chat_history else []
            
        elif session_browser_id is None:
            # Old session from before migration - claim it for this browser
            st.session_state.session_id = cookie_session_id
            
            # Update session with current browser_id
            import sqlite3
            conn = sqlite3.connect(persistence.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE sessions SET browser_id = ? WHERE session_id = ?", 
                          (st.session_state.browser_id, cookie_session_id))
            conn.commit()
            conn.close()
            
            # Load session state
            loaded_state = persistence.load_session_state(cookie_session_id)
            if loaded_state:
                st.session_state.original_df = loaded_state.get('original_df')
                st.session_state.current_df = loaded_state.get('current_df')
                st.session_state.approved_df = loaded_state.get('approved_df')
                st.session_state.current_phase = loaded_state.get('phase', 'upload')
            
            # Load chat history
            chat_history = persistence.get_chat_history(cookie_session_id)
            st.session_state.chat_history = chat_history if chat_history else []
        else:
            # Session belongs to different browser - security violation!
            # Invalidate cookie and create new session
            session_id = persistence.create_session(browser_id=st.session_state.browser_id)
            st.session_state.session_id = session_id
            cookies['session_id'] = session_id
            cookies_need_save = True
    else:
        # No valid cookie session - create a new session for this browser
        session_id = persistence.create_session(browser_id=st.session_state.browser_id)
        st.session_state.session_id = session_id
        cookies['session_id'] = session_id
        cookies_need_save = True

# Save cookies once if needed (avoid duplicate key errors)
if cookies_need_save:
    cookies.save()

# Read recipe inputs
user_feedback = dataiku.Folder("VZvmcTtt")
user_feedback_info = user_feedback.get_info()

file_name = "intention_feedback.json"

## Deduplicate user feedback
def deduplicate_across_categories(data_dict):
    merged_latest = {}

    # Merge all items from both categories
    for category, items in data_dict.items():
        for item in items:
            text = item["text"]
            ts = datetime.fromisoformat(item["timestamp"])
            # Keep latest timestamp and category
            if text not in merged_latest or ts > merged_latest[text]["timestamp"]:
                merged_latest[text] = {
                    "text": text,
                    "timestamp": ts,
                    "category": category
                }

    # Rebuild final structure with no duplicates
    result = {"code": [], "insight": []}
    for v in merged_latest.values():
        result[v["category"]].append({
            "text": v["text"],
            "timestamp": v["timestamp"].isoformat()
        })

    return result

# Load learned examples
with user_feedback.get_download_stream("intention_feedback.json") as stream:
    feedback = json.loads(stream.read().decode("utf-8"))

deduped_data  = deduplicate_across_categories(feedback)

# Deduplicate by keeping latest timestamp for each text
latest_code_entries = {}
for item in deduped_data.get("code", []):
    text = item["text"]
    ts = datetime.fromisoformat(item["timestamp"])
    if text not in latest_code_entries or ts > latest_code_entries[text]["timestamp"]:
        latest_code_entries[text] = {
            "text": text,
            "timestamp": ts
        }

# Convert back to list with ISO timestamps
latest_code_list = [
    {"text": v["text"], "timestamp": v["timestamp"].isoformat()}
    for v in latest_code_entries.values()]

code_feedback = [f'{entry["text"]} -> code' for entry in latest_code_list]

code_feedback_format = "  \n".join(code_feedback)

# Deduplicate by keeping latest timestamp for each text
latest_insight_entries = {}
for item in deduped_data.get("insight", []):
    text = item["text"]
    ts = datetime.fromisoformat(item["timestamp"])
    if text not in latest_insight_entries or ts > latest_insight_entries[text]["timestamp"]:
        latest_insight_entries[text] = {
            "text": text,
            "timestamp": ts
        }

# Convert back to list with ISO timestamps
latest_insight_list = [
    {"text": v["text"], "timestamp": v["timestamp"].isoformat()}
    for v in latest_insight_entries.values()
]

insight_feedback = [f'{entry["text"]} -> insight' for entry in latest_insight_list]

insight_feedback_format = "  \n".join(insight_feedback)

def apply_manual_conversion_fallback(df, column_name, target_dtype, sample_data):
    """
    Manual fallback conversion for common data type issues when AI fails.
    """
    df_copy = df.copy()
    
    if target_dtype == 'float64':
        # Handle currency and numeric data with special characters
        def clean_numeric(value):
            if pd.isna(value):
                return value
            # Convert to string and remove common formatting
            value_str = str(value).strip()
            # Remove currency symbols, commas, spaces
            cleaned = re.sub(r'[$€£¥,\s%]', '', value_str)
            # Handle parentheses for negative numbers
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            try:
                return float(cleaned)
            except:
                return np.nan
        
        df_copy[column_name] = df_copy[column_name].apply(clean_numeric)
        
    elif target_dtype == 'int64':
        # Similar to float but convert to int
        def clean_integer(value):
            if pd.isna(value):
                return value
            value_str = str(value).strip()
            cleaned = re.sub(r'[$€£¥,\s%]', '', value_str)
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            try:
                return int(float(cleaned))  # Convert through float first
            except:
                return np.nan
        
        df_copy[column_name] = df_copy[column_name].apply(clean_integer)
        
    elif target_dtype == 'datetime64[ns]':
        # Handle various date formats
        df_copy[column_name] = pd.to_datetime(df_copy[column_name], errors='coerce')
        
    elif target_dtype == 'bool':
        # Handle text to boolean conversion
        def text_to_bool(value):
            if pd.isna(value):
                return value
            value_str = str(value).lower().strip()
            if value_str in ['true', 't', 'yes', 'y', '1', 'on']:
                return True
            elif value_str in ['false', 'f', 'no', 'n', '0', 'off']:
                return False
            else:
                return np.nan
        
        df_copy[column_name] = df_copy[column_name].apply(text_to_bool)
        
    else:
        # For other types, try direct conversion
        df_copy[column_name] = df_copy[column_name].astype(target_dtype)
    
    return df_copy

# Always reload chat history from database for consistency across reruns
if 'session_id' in st.session_state:
    chat_history = persistence.get_chat_history(st.session_state.session_id)
    st.session_state.chat_history = chat_history if chat_history else []

# Initialize session state
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = 'upload'  # upload, manipulation, analysis
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'approved_df' not in st.session_state:
    st.session_state.approved_df = None
if 'manipulation_history' not in st.session_state:
    st.session_state.manipulation_history = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'unique_categories' not in st.session_state:
    st.session_state.unique_categories = {}
if 'feedback_status' not in st.session_state:
    st.session_state.feedback_status = None

# Initialize agents
data_agent = DataAgent()
code_executor = CodeExecutor()

st.set_page_config(page_title="Natural Language AI for Reporting & Analysis (NARA)", layout="wide")

st.title("🧠 Natural Language AI for Reporting & Analysis (NARA)")
st.markdown("Upload a CSV file and use natural language to manipulate and analyze your data!")

# Sidebar for phase navigation and controls
with st.sidebar:
    st.header("📋 Workflow Status")
    
    # Phase indicators
    phases = {
        'upload': '📁 Upload Data',
        'manipulation': '🔧 Data Manipulation', 
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

    # Session Management
    st.subheader("💾 Session Management")
    
    # Show current session info
    current_session_info = persistence.get_session_info(st.session_state.session_id)
    if current_session_info:
        st.write(f"**Current Session:**")
        st.write(f"📝 {current_session_info['session_name']}")
        
        # Session renaming
        with st.expander("✏️ Rename Session"):
            new_name = st.text_input(
                "New session name:",
                value=current_session_info['session_name'],
                key="rename_session_input"
            )
            if st.button("💾 Save Name", key="save_session_name"):
                if new_name:
                    persistence.update_session_name(st.session_state.session_id, new_name)
                    st.success("Session renamed!")
                    st.rerun()
    
    # List all sessions for this browser only (multi-user isolation)
    all_sessions = persistence.get_all_sessions(browser_id=st.session_state.browser_id)
    if len(all_sessions) > 1:
        with st.expander(f"🔄 Switch Session ({len(all_sessions)} total)"):
            for session in all_sessions[:10]:  # Show up to 10 most recent
                is_current = session['session_id'] == st.session_state.session_id
                button_label = f"{'✅ ' if is_current else ''}{session['session_name']}"
                
                if not is_current and st.button(button_label, key=f"switch_{session['session_id'][:8]}"):
                    # Switch to this session
                    st.session_state.session_id = session['session_id']
                    cookies['session_id'] = session['session_id']
                    cookies.save()
                    
                    # Load session state
                    loaded_state = persistence.load_session_state(session['session_id'])
                    if loaded_state:
                        st.session_state.original_df = loaded_state.get('original_df')
                        st.session_state.current_df = loaded_state.get('current_df')
                        st.session_state.approved_df = loaded_state.get('approved_df')
                        st.session_state.current_phase = loaded_state.get('phase', 'upload')
                    
                    # Load chat history
                    chat_history = persistence.get_chat_history(session['session_id'])
                    st.session_state.chat_history = chat_history if chat_history else []
                    
                    st.success(f"Switched to: {session['session_name']}")
                    st.rerun()
    
    # Create new session
    if st.button("➕ New Session", key="create_new_session"):
        new_session_id = persistence.create_session(
            session_name=f"New Session {pd.Timestamp.now().strftime('%H:%M')}",
            browser_id=st.session_state.browser_id
        )
        st.session_state.session_id = new_session_id
        cookies['session_id'] = new_session_id
        cookies.save()
        
        # Reset session state
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
        
        st.success("New session created!")
        st.rerun()
    
    st.divider()
    
    # Global controls - always available
    st.subheader("🔧 Global Actions")
    
    if st.button("🔄 Reset Everything", type="secondary"):
        # Save browser_id before reset (needed for security)
        browser_id = st.session_state.get('browser_id')
        
        # Create new session for fresh start
        new_session_id = persistence.create_session(browser_id=browser_id)
        
        # Update cookie with new session
        cookies['session_id'] = new_session_id
        cookies.save()
        
        # Clear all session state except browser_id
        for key in list(st.session_state.keys()):
            if key != 'browser_id':
                del st.session_state[key]
        
        # Set new session_id
        st.session_state.session_id = new_session_id
        
        st.success("Everything has been reset! Starting fresh session.")
        st.rerun()
    
    # Dataset switcher - show when multiple files are uploaded
    if len(st.session_state.uploaded_files) > 1:
        st.markdown("**📂 Switch Dataset**")
        current_file = st.session_state.get('selected_file', 'None selected')
        st.write(f"Current: {current_file}")
        
        # Create selectbox for switching datasets
        available_files = list(st.session_state.uploaded_files.keys())
        if current_file in available_files:
            current_index = available_files.index(current_file)
        else:
            current_index = 0
            
        new_selection = st.selectbox(
            "Select different dataset:",
            available_files,
            index=current_index,
            key="dataset_switcher"
        )
        
        if st.button("🔄 Switch Dataset") and new_selection != current_file:
            # Save current work if any
            if st.session_state.get('current_df') is not None and st.session_state.get('selected_file'):
                st.info(f"Switching from {st.session_state.selected_file} to {new_selection}")
            
            # Switch to new dataset
            st.session_state.selected_file = new_selection
            st.session_state.original_df = st.session_state.uploaded_files[new_selection].copy()
            st.session_state.current_df = st.session_state.original_df.copy()
            st.session_state.approved_df = None
            
            # Reset work-specific state but keep chat history
            st.session_state.manipulation_history = []
            st.session_state.analysis_history = []
            st.session_state.current_phase = 'manipulation'
            
            # Update unique categories for new dataset
            st.session_state.unique_categories = {}
            for col in st.session_state.current_df.select_dtypes(include=['object', 'category']).columns:
                unique_vals = st.session_state.current_df[col].dropna().unique()
                if len(unique_vals) <= 50:  # Only store if reasonable number of unique values
                    st.session_state.unique_categories[col] = unique_vals.tolist()
            
            st.success(f"Switched to dataset: {new_selection}")
            st.rerun()
    
    st.divider()
    
    # Controls based on current phase
    if st.session_state.current_phase == 'manipulation' and st.session_state.current_df is not None:
        st.subheader("🎯 Dataset Actions")
        if st.button("✅ Approve Dataset for Analysis", type="primary"):
            st.session_state.approved_df = st.session_state.current_df.copy()
            st.session_state.approved_source = "current"  # <-- add this line
            st.session_state.current_phase = 'analysis'
            # Save session state to database
            persistence.save_session_state(
                st.session_state.session_id,
                original_df=st.session_state.original_df,
                current_df=st.session_state.current_df,
                approved_df=st.session_state.approved_df,
                phase='analysis'
            )
            st.success("Dataset approved! Moving to analysis phase.")
            st.rerun()
        
        if st.button("🔄 Reset to Original"):
            if st.session_state.original_df is not None:
                st.session_state.current_df = st.session_state.original_df.copy()
                st.session_state.manipulation_history = []
                # Clear approved_df since we've reset to original - must re-approve
                st.session_state.approved_df = None
                st.session_state.approved_source = None  # <-- add this line
                st.success("Reset to original dataset!")
                st.rerun()
    
    elif st.session_state.current_phase == 'analysis':
        st.subheader("🎯 Analysis Actions")
        if st.button("🔧 Back to Manipulation"):
            st.session_state.current_phase = 'manipulation'
            # Clear approved_df so it must be re-approved after any changes
            st.session_state.approved_df = None
            st.session_state.approved_source = None  # <-- add this line
            st.rerun()
        
        if st.button("📁 Upload New Dataset"):
            # Reset all session state
            st.session_state.uploaded_files = {}
            st.session_state.selected_file = None
            for key in ['original_df', 'current_df', 'approved_df', 'manipulation_history', 'analysis_history', 'chat_history', 'unique_categories']:
                if 'history' in key or key == 'unique_categories':
                    st.session_state[key] = [] if 'history' in key else {}
                else:
                    st.session_state[key] = None
            st.session_state.current_phase = 'upload'
            st.rerun()

# Main content area
if st.session_state.current_phase == 'upload':
    st.header("📁 Upload Your Datasets")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Choose CSV files", 
        type="csv",
        accept_multiple_files=True,
        help="Upload one or more CSV files to start data manipulation and analysis"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                try:
                    # Read the CSV file
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_files[uploaded_file.name] = df.copy()
                    st.success(f"✅ Successfully loaded {uploaded_file.name} with {len(df)} rows and {len(df.columns)} columns!")
                    
                except Exception as e:
                    st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
    
    # File selection if multiple files are uploaded
    if st.session_state.uploaded_files:
        st.subheader("📂 Select Dataset to Work With")
        
        file_options = list(st.session_state.uploaded_files.keys())
        selected_file = st.selectbox(
            "Choose a dataset:",
            options=file_options,
            index=0 if not st.session_state.selected_file else file_options.index(st.session_state.selected_file) if st.session_state.selected_file in file_options else 0
        )
        
        if st.button("🚀 Start Working with Selected Dataset", type="primary"):
            st.session_state.selected_file = selected_file
            st.session_state.original_df = st.session_state.uploaded_files[selected_file].copy()
            st.session_state.current_df = st.session_state.uploaded_files[selected_file].copy()
            
            # Extract unique categories for AI context
            st.session_state.unique_categories = {}
            for col in st.session_state.current_df.columns:
                if st.session_state.current_df[col].dtype == 'object':
                    unique_vals = st.session_state.current_df[col].unique()
                    if len(unique_vals) <= 50:  # Only store if reasonable number of unique values
                        st.session_state.unique_categories[col] = unique_vals.tolist()
            
            st.session_state.current_phase = 'manipulation'
            # Update session name with dataset
            persistence.update_session_name(
                st.session_state.session_id,
                f"Session - {selected_file}"
            )
            
            # Save session state to database
            persistence.save_session_state(
                st.session_state.session_id,
                original_df=st.session_state.original_df,
                current_df=st.session_state.current_df,
                approved_df=None,
                phase='manipulation'
            )
            st.success(f"✅ Started working with {selected_file}")
            st.rerun()
        
        # Preview datasets
        st.subheader("📊 Dataset Previews")
        for filename, df in st.session_state.uploaded_files.items():
            with st.expander(f"Preview: {filename} ({len(df)} rows, {len(df.columns)} columns)"):
                st.dataframe(df.head(5), use_container_width=True)

elif st.session_state.current_phase == 'manipulation':
    st.header("🔧 Data Manipulation Phase")

    # Show current dataset
    #st.subheader(f"📊 Current Dataset: {st.session_state.selected_file}")
    #display_dataframe(st.session_state.current_df, unique_key="current")

    # Display current dataset info
    #if st.session_state.current_df is not None:
    #    col1, col2, col3 = st.columns(3)
    #    with col1:
    #        st.metric("Rows", len(st.session_state.current_df))
    #    with col2:
    #        st.metric("Columns", len(st.session_state.current_df.columns))
    #    with col3:
    #        st.metric("Operations", len(st.session_state.manipulation_history))
    
    # Data type modification section
    st.subheader("🔧 Data Types Modifications")
    if st.session_state.current_df is not None:
        with st.expander("Modify Column Data Types"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Select column to modify
                column_to_modify = st.selectbox(
                    "Select column to modify:",
                    options=st.session_state.current_df.columns.tolist()
                )
                
                # Show current data type
                current_dtype = str(st.session_state.current_df[column_to_modify].dtype)
                st.text(f"Current type: {current_dtype}")
                
            with col2:
                # Select new data type
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
                    format_func=lambda x: dtype_options[x]
                )
                
                if st.button("Apply Data Type Change"):
                    df_temp = st.session_state.current_df.copy()
                    success = False
                    error_message = ""
                    
                    # First try: Default pandas conversion
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
                        
                        # Second try: Use GPT to generate custom conversion code
                        try:
                            with st.spinner("Generating intelligent conversion code..."):
                                # Sample some data for context
                                sample_data = df_temp[column_to_modify].dropna().head(10).tolist()
                                
                                conversion_prompt = f"""Generate Python code to convert a pandas DataFrame column from its current data type to {new_dtype} ({dtype_options[new_dtype]}).

                                Column name: {column_to_modify}
                                Current data type: {current_dtype}
                                Target data type: {new_dtype}
                                Sample values: {sample_data}
                                Error from standard conversion: {error_message}

                                Requirements:
                                - Use 'df' as the dataframe variable and '{column_to_modify}' as the column name
                                - Handle potential errors and edge cases intelligently
                                - Clean/preprocess the data if needed before conversion
                                - Use appropriate pandas methods for the conversion
                                - Return ONLY the Python code without any markdown formatting or explanations
                                - Make sure the code handles the specific error mentioned above

                                Example approaches based on the error:
                                - For currency strings like '$127613': Remove dollar signs, commas, and other non-numeric characters before converting to float
                                - For datetime: handle different date formats, clean strings first  
                                - For numeric: remove currency symbols ($, €, £), commas, percent signs, and other formatting
                                - For boolean: map text values to True/False intelligently
                                - For category: handle missing values appropriately

                                The code should be robust and handle the specific formatting issues shown in the sample data."""

                                generated_code = data_agent.generate_manipulation_code(
                                    df_temp, 
                                    conversion_prompt,
                                    chat_history=[],
                                    unique_categories={}
                                )
                                
                                if generated_code:
                                    # Show the generated code for debugging
                                    with st.expander("🔍 View Generated Conversion Code"):
                                        st.code(generated_code, language="python")
                                    
                                    # Execute the AI-generated conversion code
                                    result_df = code_executor.execute_manipulation(df_temp, generated_code)
                                    
                                    if result_df is not None:
                                        df_temp = result_df
                                        success = True
                                        st.success("✅ AI-powered conversion successful!")
                                    else:
                                        st.error("❌ AI-generated code failed to execute properly")
                                        st.info("💡 The generated code above had execution issues. Trying manual cleanup...")
                                        
                                        # Final fallback: Apply common data cleaning patterns based on the data type
                                        try:
                                            df_temp = apply_manual_conversion_fallback(df_temp, column_to_modify, new_dtype, sample_data)
                                            success = True
                                            st.success("✅ Manual cleanup conversion successful!")
                                        except Exception as manual_error:
                                            st.error(f"❌ Manual cleanup also failed: {str(manual_error)}")
                                else:
                                    st.error("❌ Could not generate conversion code")
                                    
                        except Exception as ai_error:
                            st.error(f"❌ AI conversion failed: {str(ai_error)}")
                    
                    # Apply changes if successful
                    if success:
                        st.session_state.current_df = df_temp
                        st.session_state.manipulation_history.append(f"Changed {column_to_modify} data type to {new_dtype}")
                        st.success(f"✅ Successfully changed {column_to_modify} to {dtype_options[new_dtype]}")
                        st.rerun()
                    else:
                        st.error("❌ Both default and AI-powered conversions failed")
    
    # Show current dataset
    st.subheader("📊 Current Dataset")
    display_dataframe(st.session_state.current_df, unique_key="current")
    
    # Download current dataset
    if st.session_state.current_df is not None:
        csv_data = st.session_state.current_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Current Dataset",
            data=csv_data,
            file_name="manipulated_data.csv",
            mime="text/csv"
        )
    
    # Natural language query input
    st.subheader("💬 Natural Language Data Manipulation")
    
    st.info("💡 **Tip**: This phase is for data manipulation (filtering, grouping, creating columns, etc.). For visualizations and analysis, approve your dataset first and move to the Analysis phase.")
    
    # Chat input for natural Enter key support
    query = st.chat_input(
        placeholder="Describe what you want to do with your data (e.g., Remove rows where age is less than 18, create new column for age groups...)",
        key="manipulation_input"
    )
    
    if query and query.strip():
        # Check if the user is asking for visualization
        visualization_keywords = ['plot', 'chart', 'graph', 'histogram', 'distribution', 'visualize', 'show', 'display']
        if any(keyword in query.lower() for keyword in visualization_keywords):
            st.warning("🎯 **This looks like a visualization request!** Please approve your dataset first and use the Analysis phase for charts and graphs.")
            st.info("Click '✅ Approve Dataset for Analysis' in the sidebar to move to the Analysis phase.")
        else:
            with st.spinner("🤖 Generating and executing code..."):
                try:
                    
                    # Generate code using AI agent with chat history and unique categories
                    generated_code = data_agent.generate_manipulation_code(
                        st.session_state.current_df, 
                        query,
                        chat_history=st.session_state.chat_history,
                        unique_categories=st.session_state.unique_categories
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'type': 'manipulation',
                        'query': query,
                        'timestamp': pd.Timestamp.now()
                    })

                    # Save to database
                    persistence.save_chat_message(st.session_state.session_id, query, 'manipulation')
                    
                    if generated_code:
                        st.subheader("🔍 Generated Code")
                        st.code(generated_code, language="python")
                        
                        # Execute the code
                        result_df = code_executor.execute_manipulation(
                            st.session_state.current_df, 
                            generated_code
                        )
                        
                        if result_df is not None:
                            # Update current dataframe
                            st.session_state.current_df = result_df
                            
                            # Add to history
                            st.session_state.manipulation_history.append({
                                'query': query,
                                'code': generated_code,
                                'timestamp': pd.Timestamp.now()
                            })
                            
                            st.success("✅ Operation completed successfully!")
                            st.subheader("📊 Updated Dataset")
                            display_dataframe(result_df, unique_key="updated")
                            
                            # Show changes
                            if st.session_state.original_df is not None:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Rows", len(result_df), 
                                            delta=len(result_df) - len(st.session_state.original_df))
                                with col2:
                                    st.metric("Columns", len(result_df.columns),
                                            delta=len(result_df.columns) - len(st.session_state.original_df.columns))
                            
                            st.rerun()
                        else:
                            st.error("❌ Code execution failed. Please try a different query.")
                    else:
                        st.error("❌ Failed to generate code. Please try rephrasing your query.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("Please try rephrasing your query or check your data.")
    
    # Chat History Display
    if st.session_state.chat_history:
        with st.expander("💬 Chat History", expanded=False):
            st.subheader("Recent Conversations")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
                with st.container():
                    # Handle both string and datetime timestamp formats
                    timestamp = chat.get('timestamp', '')
                    if isinstance(timestamp, str):
                        time_str = timestamp
                    else:
                        try:
                            time_str = timestamp.strftime('%H:%M:%S')
                        except:
                            time_str = str(timestamp)
                    
                    st.write(f"**{chat['type'].title()}** - {time_str}")
                    st.write(f"Query: {chat['query']}")
                    st.divider()
    
    # Show manipulation history
    if st.session_state.manipulation_history:
        st.subheader("📋 Manipulation History")
        for i, operation in enumerate(reversed(st.session_state.manipulation_history)):
            # Handle both old string format and new dict format for backward compatibility
            if isinstance(operation, str):
                with st.expander(f"Operation {len(st.session_state.manipulation_history) - i}: {operation[:50]}..."):
                    st.write(f"**Query:** {operation}")
                    st.write("**Time:** Legacy operation")
                    st.write("**Code:** Not available for legacy operations")
            else:
                with st.expander(f"Operation {len(st.session_state.manipulation_history) - i}: {operation['query'][:50]}..."):
                    st.write(f"**Query:** {operation['query']}")
                    st.write(f"**Time:** {operation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.code(operation['code'], language="python")

elif st.session_state.current_phase == 'analysis':
    st.header("📊 Data Analysis Phase")
    
    # Display approved dataset info
    if st.session_state.approved_df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(st.session_state.approved_df))
        with col2:
            st.metric("Columns", len(st.session_state.approved_df.columns))
        with col3:
            st.metric("Analyses", len(st.session_state.analysis_history))
    
    # Show approved dataset
    st.subheader("📊 Approved Dataset")
    st.info("🎯 **Ready for Analysis!** This is your approved dataset from the manipulation phase. All analyses will automatically use this data.")
    display_dataframe(st.session_state.approved_df, unique_key="approved")
    
    # Natural language analysis query - KEEP AT TOP
    st.subheader("🔍 Natural Language Data Analysis")

    # Show chat history if it exists
    if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
        with st.expander(f"💬 Chat History ({len(st.session_state.chat_history)} questions)", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                query_type = chat.get('type', 'unknown')
                query_text = chat.get('query', 'No query')
                timestamp = chat.get('timestamp', '')
                
                # Format timestamp if available
                time_str = ""
                if timestamp:
                    try:
                        if isinstance(timestamp, str):
                            time_str = timestamp
                        else:
                            time_str = timestamp.strftime("%H:%M:%S")
                    except:
                        time_str = str(timestamp)
                
                # Display with icon based on type
                icon = "🔧" if query_type == "manipulation" else "📊"
                st.markdown(f"{icon} **Q{len(st.session_state.chat_history)-i}:** {query_text}")
                if time_str:
                    st.caption(f"🕐 {time_str}")
                if i < len(st.session_state.chat_history) - 1:
                    st.divider()
    
    # Quick action buttons for common analyses
    st.write("**Quick Actions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    # Handle quick action clicks
    quick_action_clicked = False
    quick_action_query = ""
    
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
    
    # Chat input for natural Enter key support
    analysis_query = st.chat_input(
        placeholder="What analysis would you like to perform? (e.g., Show correlation between age and salary, create histogram of sales...)",
        key="analysis_input"
    )
    
    # For quick actions, use the stored query and show what's being processed
    if quick_action_clicked:
        analysis_query = quick_action_query
        st.info(f"🚀 Processing: {analysis_query}")
    
    # Process analysis if we have a query (either from text input or quick actions)
    if (analysis_query and analysis_query.strip()) or quick_action_clicked:
        with st.spinner("🤖 Analyzing with advanced AI capabilities..."):
            try:
                # Use intelligent routing to determine approach
                intent = data_agent.route_query_intelligently(analysis_query, code_feedback_format, insight_feedback_format)
                
                # Store last query and intent for feedback system
                st.session_state.last_query = analysis_query
                st.session_state.last_intent = intent

                st.session_state.feedback_status = True
                
                if intent == "insight":
                    # Generate enhanced business insights
                    st.info("🧠 Generating strategic business insights...")
                    if st.session_state.approved_df is not None:
                        insights = data_agent.generate_enhanced_insights(
                            st.session_state.approved_df,
                            analysis_query
                        )
                    else:
                        insights = "No approved dataset available for analysis."
                    
                    if insights:
                        # Add to history
                        analysis_result = {
                            'text_output': insights,
                            'dataframe': None,
                            'plotly_fig': None,
                            'plot': None,
                            'type': 'insight'
                        }
                        
                        st.session_state.analysis_history.append({
                            'query': analysis_query,
                            'code': None,
                            'result': analysis_result,
                            'intent': 'insight',
                            'timestamp': pd.Timestamp.now()
                        })
                        
                        st.success("✅ Strategic analysis completed!")
                        st.subheader("📊 Business Insights")
                        st.write(insights)
                        
                        # Generate key takeaways
                        try:
                            from advanced_analysis import generate_insight_narrative_summary
                            summary = generate_insight_narrative_summary(insights, analysis_query)
                            if summary:
                                st.subheader("🎯 Key Takeaways")
                                st.write(summary)
                        except Exception:
                            pass

                    else:
                        st.error("❌ Failed to generate insights. Please try rephrasing your query.")
                        
                else:
                    # Generate and execute analysis code with self-repair
                    st.info("⚙️ Generating and executing analysis code...")
                    generated_code = data_agent.generate_analysis_code(
                        st.session_state.approved_df, 
                        analysis_query,
                        chat_history=st.session_state.chat_history,
                        unique_categories=st.session_state.unique_categories
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'type': 'analysis', 
                        'query': analysis_query,
                        'timestamp': pd.Timestamp.now()
                    })

                    # Save to database
                    persistence.save_chat_message(st.session_state.session_id, analysis_query, 'analysis')
                    
                    if generated_code:
                        # Execute with enhanced capabilities and self-repair
                        analysis_result = code_executor.execute_analysis(
                            st.session_state.approved_df, 
                            generated_code,
                            analysis_query  # Pass query for self-repair
                        )
                        
                        if analysis_result:
                            # Add to history
                            st.session_state.analysis_history.append({
                                'query': analysis_query,
                                'code': generated_code,
                                'result': analysis_result,
                                'intent': 'code',
                                'timestamp': pd.Timestamp.now()
                            })
                            
                            success_msg = "✅ Analysis completed successfully!"
                            if analysis_result.get('self_repair_used'):
                                success_msg += " (Enhanced with automatic error correction)"
                            st.success(success_msg)
                            
                            # Show generated code BEFORE results
                            with st.expander("🔍 View Generated Python Code"):
                                st.code(generated_code, language="python")
                            
                            st.subheader("📊 Analysis Results")
                            
                            # Display results
                            results_displayed = False
                            
                            # Show text output (analysis summary) if available
                            if analysis_result.get('text_output'):
                                st.write("**Analysis Summary:**")
                                st.write(analysis_result['text_output'])
                                results_displayed = True
                            
                            if analysis_result.get('dataframe') is not None:
                                st.write("**Result DataFrame:**")
                                display_dataframe(analysis_result['dataframe'], unique_key="analysis_result")
                                results_displayed = True
                            
                            if analysis_result.get('plotly_fig'):
                                st.write("**Visualization:**")
                                st.plotly_chart(analysis_result['plotly_fig'], use_container_width=True, key="current_analysis_chart")
                                results_displayed = True
                                    
                            elif analysis_result.get('plot'):
                                st.write("**Visualization:**")
                                st.pyplot(analysis_result['plot'])
                                results_displayed = True
                            
                            if not results_displayed:
                                st.info("Code executed successfully but no output was generated. This might be normal for some analyses.")
                            
                        else:
                            st.error("❌ Analysis execution failed. Please try a different query.")
                            # Still show the code even if execution failed
                            with st.expander("🔍 View Generated Python Code"):
                                st.code(generated_code, language="python")
                    else:
                        st.error("❌ Failed to generate analysis code. Please try rephrasing your query.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Please try rephrasing your analysis request.")
                    
    
    # Show conversation history AFTER the chat interface (only previous analyses, not current one)
    if st.session_state.analysis_history and len(st.session_state.analysis_history) > 1:
        st.divider()
        st.subheader("💬 Previous Analysis Conversations")
        # Show all except the latest one (which is displayed above during execution)
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[:-1])):
                # Calculate the actual position in the original list
                analysis_num = len(st.session_state.analysis_history) - i - 1
                with st.expander(f"Analysis #{analysis_num}: {analysis['query'][:60]}..." if len(analysis['query']) > 60 else f"Analysis #{analysis_num}: {analysis['query']}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Query:** {analysis['query']}")
                    with col2:
                        st.write(f"**Time:** {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if analysis.get('code'):
                        with st.expander("🔍 View Generated Code"):
                            st.code(analysis['code'], language="python")
                    
                    if analysis['result']:
                        if 'text_output' in analysis['result'] and analysis['result']['text_output']:
                            st.write("**Analysis Summary:**")
                            st.write(analysis['result']['text_output'])
                        
                        if 'dataframe' in analysis['result'] and analysis['result']['dataframe'] is not None:
                            st.write("**Result Data:**")
                            display_dataframe(analysis['result']['dataframe'], unique_key=f"history_{i}")
                        
                        if 'plotly_fig' in analysis['result'] and analysis['result']['plotly_fig']:
                            st.plotly_chart(analysis['result']['plotly_fig'], use_container_width=True, key=f"history_chart_{i}")
                        elif 'plot' in analysis['result'] and analysis['result']['plot']:
                            st.pyplot(analysis['result']['plot'])
    
    # ========== ALWAYS-ON FEEDBACK (ADD THIS BLOCK) ==========
    last_query = st.session_state.get("last_query")
    last_intent = st.session_state.get("last_intent")

    if st.session_state.feedback_status:

        # Add feedback buttons for insight responses
        st.markdown("---")
        st.markdown("**Was this AI Response helpful?**")
        # Load learned examples
        with user_feedback.get_download_stream("intention_feedback.json") as stream:
            feedback = json.loads(stream.read().decode("utf-8"))
        col1, col2, col3 = st.columns([1, 1, 8])
        
        with col1:
            if st.button("👍", help="This AI response was helpful"):
                try:
                    timestamp_str = datetime.now().isoformat()
                    if last_intent == 'code':
                        feedback['code'].append({
                            'text': last_query,
                            'timestamp': timestamp_str
                        })
                    else:
                        feedback['insight'].append({
                            'text': last_query,
                            'timestamp': timestamp_str
                        })
                    
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
                        feedback['insight'].append({
                            'text': last_query,
                            'timestamp': timestamp_str
                        })
                    else:
                        feedback['code'].append({
                            'text': last_query,
                            'timestamp': timestamp_str
                        })
                    with user_feedback.get_writer("intention_feedback.json") as w:
                        w.write(json.dumps(feedback, indent=2).encode("utf-8"))

                    st.toast("Saved to feedback system ✅")

                except Exception as e:
                    st.error(f"Error saving feedback: {e}")
            
