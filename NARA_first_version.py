import altair as alt
import dataiku
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import re
import sys
from typing import Union, Tuple, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

# Read recipe inputs
user_feedback = dataiku.Folder("VZvmcTtt")
user_feedback_info = user_feedback.get_info()

file_name = "intention_feedback.json"

st.set_page_config(page_title="Natural Language AI for Reporting & Analysis (NARA)", layout="wide")

def run_code_with_self_repair(
    code: str,
    exec_env: Dict[str, Any],
    query: str,
    max_tries: int = 3
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Exec code; if it errors, ask the LLM to fix with up to max_tries attempts.
    Returns (observation_dict, final_code or None on failure).
    observation_dict may have keys: fig, result, errors, repair_attempts, max_retries_exceeded
    """
    import re, sys
    current_code = code
    repair_history = []
    obs: Dict[str, Any] = {}  # ensure defined even if something odd happens

    for attempt in range(max_tries):
        # Clear previous results
        exec_env.pop("fig", None)
        exec_env.pop("result", None)
        exec_env.pop("results", None)
        obs = {}
        cleaned = re.sub(r'\w*\.show\(\)', '', current_code)

        # Execute current code
        try:
            original = sys.getrecursionlimit()
            sys.setrecursionlimit(100)
            exec(cleaned, exec_env)
        except Exception as e:
            obs["errors"] = f"{type(e).__name__}: {e}"
        finally:
            sys.setrecursionlimit(original)

        # Capture results
        if "fig" in exec_env:
            obs["fig"] = exec_env["fig"]
        if "result" in exec_env:
            obs["result"] = exec_env["result"]
        if "results" in exec_env and "result" not in obs:
            obs["result"] = exec_env["results"]

        # If successful, return
        if not obs.get("errors"):
            if attempt > 0:
                obs["repair_attempts"] = attempt
                obs["repair_history"] = repair_history
            return obs, current_code

        # If this was the last attempt, return with failure
        if attempt == max_tries - 1:
            obs["max_retries_exceeded"] = True
            obs["repair_history"] = repair_history
            return obs, None

        # Record this attempt for history
        repair_history.append({
            "attempt": attempt + 1,
            "error": obs["errors"],
            "code": current_code[:200] + "..." if len(current_code) > 200 else current_code
        })

        # Generate fix using GPT-4o
        fix_prompt = f"""You wrote Python for this task and it errored on attempt {attempt + 1}/{max_tries}. Fix it and return ONLY code.

Task: {query}
Current Error: {obs['errors']}
Current code:
{current_code}

Previous repair attempts: {len(repair_history)}
{chr(10).join([f"Attempt {h['attempt']}: {h['error']}" for h in repair_history[-2:]]) if len(repair_history) > 0 else ""}

Rules:
- Use the existing variables/libraries already available in the environment (df, pd, px, np, stats, go, make_subplots).
- Assign any figure to 'fig' and data/table to 'result'.
- Do not call .show().
- Learn from previous repair attempts to avoid repeating the same mistakes.
"""
        try:
            llm_client = AzureChatOpenAI(
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version="2024-10-21",
                #max_tokens=1500,  # Increased from 800 to handle larger datasets
                temperature=0)

            fixer_response = llm_client.invoke(fix_prompt)
            fixer = fixer_response.content if hasattr(fixer_response, 'content') else str(fixer_response)
            if "```" in fixer:
                parts = fixer.split("```")
                if len(parts) >= 3:
                    fixer = parts[-2]
                if fixer.lstrip().startswith("python"):
                    fixer = fixer.split("python", 1)[1].strip()
            current_code = fixer.strip() if fixer else current_code
        except Exception as repair_error:
            obs["repair_generation_error"] = str(repair_error)
            return obs, None

    return obs, None

def self_check_and_rewrite_insight(insights_text: str, metrics_text: str) -> str:
    """
    Ask the LLM to check alignment with computed metrics and rewrite if needed.
    Returns the final (possibly rewritten) insight text.
    """
    check_prompt = f"""Review your analysis against the provided metrics.
- Flag unsupported claims, contradictions, or numbers not grounded in metrics.
- If issues exist, REWRITE a corrected version below your critique.
- Keep it concise, executive-level, and grounded in the metrics only.

Analysis:
{insights_text}

Metrics:
{metrics_text}
"""
    llm_client = AzureChatOpenAI(
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version="2024-10-21",
                #max_tokens=1500,  # Increased from 800 to handle larger datasets
                temperature=0)
    
    resp_obj = llm_client.invoke(check_prompt)
    resp = resp_obj.content if hasattr(resp_obj, 'content') else str(resp_obj)
    # Heuristic: if it contains the word "rewrite" or looks like a revised section, use the last block after a separator.
    # Otherwise, keep original.
    # You can make this smarter; this is intentionally simple.
    if resp and ("rewrite" in resp.lower() or "revised" in resp.lower()):
        return resp  # show critique + rewrite
    return insights_text
# --- End Agentic helpers ---

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

def safe_key_format(key):
    """Safely format a key that might be a string, tuple, or other type"""
    if isinstance(key, str):
        return key.replace('_', ' ').title()
    else:
        return str(key).replace('_', ' ').title()

def generate_narrative_from_results(exec_env, query, api_key, endpoint):
    """Generate narrative summary from execution results"""
    try:
        # Extract key results from execution environment
        summary_points = []
        
        # Check for statistical results
        if 'result' in exec_env and isinstance(exec_env['result'], dict):
            result = exec_env['result']
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if 'mean' in str(key).lower():
                        summary_points.append(f"Mean {key}: {value:.2f}")
                    elif 'correlation' in str(key).lower() or 'corr' in str(key).lower():
                        summary_points.append(f"Correlation: {value:.3f}")
                    elif 'p' in str(key).lower() and 'value' in str(key).lower():
                        summary_points.append(f"Statistical significance: p={value:.4f}")
                    elif 't_statistic' in str(key) or 'statistic' in str(key):
                        summary_points.append(f"Test statistic: {value:.3f}")
                    else:
                        summary_points.append(f"{key}: {value:.3f}")
        
        # Check for DataFrame results  
        if 'result' in exec_env and hasattr(exec_env['result'], 'shape'):
            df_result = exec_env['result']
            summary_points.append(f"Results shape: {df_result.shape[0]} rows, {df_result.shape[1]} columns")
            
        # Check for figure/chart information
        if 'fig' in exec_env:
            summary_points.append("Visualization created")
            
        # If we have meaningful results, generate narrative
        if summary_points:
            results_text = "; ".join(summary_points)
            
            narrative_prompt = f'''Based on these analysis results for the query "{query}":

Results: {results_text}

As a Principal Data Scientist, provide 2-4 concise bullet-point insights or a 2-3 sentence executive summary that:
- Highlights the key business implications
- Uses simple, professional language  
- Focuses on actionable insights
- Avoids technical jargon

Format as bullet points starting with • or as a brief executive summary.'''

            narrative_client = AzureChatOpenAI(
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version="2024-10-21",
                #max_tokens=1500,  # Increased from 800 to handle larger datasets
                temperature=0
                                    )
            
            response = narrative_client.invoke(narrative_prompt)
            return response.content
            
    except Exception as e:
        return None
    
    return None

def generate_insight_narrative_summary(insights_text, query, api_key, endpoint):
    """Generate concise summary from detailed insights"""
    try:
        summary_prompt = f'''From this detailed analysis for the query "{query}":

{insights_text}

Extract 2-4 key actionable takeaways as bullet points. Focus on:
- The most important business implications
- Clear, simple language
- Actionable recommendations
- Avoid repeating the full analysis

Format as bullet points starting with •'''

        summary_client = AzureChatOpenAI(
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version="2024-10-21",
                #max_tokens=1500,  # Increased from 800 to handle larger datasets
                temperature=0
                        )
        
        response = summary_client.invoke(summary_prompt)
        return response.content
        
    except Exception as e:
        return None

# Sidebar for navigation and guided tour
with st.sidebar:

    #st.image(jnj_logo)
    #st.title("🎥 Video Introduction")

    #col1, col2, col3 = st.columns([3,1,1])

    #with col1:
    #    st.video(video_bytes)

    st.title("🧭 Navigation")

    # Initialize tour state
    if "show_tour" not in st.session_state:
        st.session_state.show_tour = False
    if "tour_step" not in st.session_state:
        st.session_state.tour_step = 0

    st.markdown("### Quick Start")
    if st.button("🎯 Start Guided Tour", key="start_tour"):
        st.session_state.show_tour = True
        st.session_state.tour_step = 1
        st.rerun()

    if st.button("📋 Reset Everything", key="reset_all"):
        # Store current uploader key before clearing session state
        current_uploader_key = st.session_state.get("uploader_key", 0)
        # Clear all session state
        #for key in list(st.session_state.keys()):
        #    del st.session_state[key]
        # Increment uploader key to force file uploader reset
        st.session_state.uploader_key = current_uploader_key + 1
        st.cache_data.clear()
        st.rerun()

    st.markdown("### Help & Tips")
    with st.expander("💡 How to Use"):
        st.markdown("""
        **1. Upload Data**: Use the file uploader to add CSV files

        **2. Choose Mode**: 
        - Single Dataset: Analyze one file
        - Auto-join: Merge files on common columns
        - Compare: Analyze multiple datasets separately

        **3. Ask Questions**: Use natural language or quick actions

        **4. View Results**: Charts, tables, and insights appear below
        """)

    with st.expander("🔍 Example Questions"):
        st.markdown("""
        **Visualizations:**
        - "Show distribution of age"
        - "Create a scatter plot of income vs expenses"
        - "Plot correlation heatmap"

        **Analysis:**
        - "What are the key insights?"
        - "Find data quality issues"
        - "Compare groups by category"

        **General Knowledge:**
        - "What is machine learning?"
        - "Explain correlation vs causation"
        """)

    with st.expander("⚙️ Features"):
        st.markdown("""
        - **Multi-file Upload**: Handle multiple CSV files
        - **Smart Auto-join**: Merge datasets on common columns
        - **AI Intent Detection**: Automatic routing to best response type
        - **Persistent Chat**: All results saved in history
        - **Quick Actions**: One-click common analyses
        - **General Knowledge**: Ask any question beyond data
        """)

st.title("🤖 Natural Language AI for Reporting & Analysis (NARA)")
st.markdown("Upload CSV files and ask questions in natural language!")

# Display guided tour if active
if st.session_state.show_tour:
    tour_steps = [
        {
            "title": "Welcome to AI Data Analysis! 🎉",
            "content": "This tour will guide you through all the features. Let's start by uploading some data.",
            "action": "Look for the file uploader below to add your CSV files."
        },
        {
            "title": "Step 1: Upload Your Data 📁",
            "content": "Use the file uploader to add one or multiple CSV files. The system will automatically detect common columns for joining.",
            "action": "Upload CSV files using the interface below."
        },
        {
            "title": "Step 2: Choose Analysis Mode 🔧",
            "content": "For multiple files, you can analyze them separately, join them automatically, or compare them side by side.",
            "action": "Select your preferred analysis mode from the radio buttons."
        },
        {
            "title": "Step 3: Quick Actions ⚡",
            "content": "Use these buttons for common analyses like data summaries, correlations, trends, and quality checks.",
            "action": "Try clicking any quick action button."
        },
        {
            "title": "Step 4: Ask Questions 💬",
            "content": "Type any question about your data in natural language. Ask for charts, calculations, or general insights.",
            "action": "Type a question and click Analyze."
        },
        {
            "title": "Step 5: View Results 📊",
            "content": "Results appear as charts, tables, or text insights. Everything is saved in chat history for easy reference.",
            "action": "Scroll down to see your analysis results."
        },
        {
            "title": "Tour Complete! 🎓",
            "content": "You're ready to analyze data! Use the sidebar help sections anytime for tips and examples.",
            "action": "Start exploring your data with confidence!"
        }
    ]

    if st.session_state.tour_step <= len(tour_steps):
        current_step = tour_steps[st.session_state.tour_step - 1]

        st.info(f"**{current_step['title']}**\n\n{current_step['content']}\n\n*{current_step['action']}*")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.session_state.tour_step > 1:
                if st.button("← Previous", key="tour_prev"):
                    st.session_state.tour_step -= 1
                    st.rerun()
        with col2:
            if st.session_state.tour_step < len(tour_steps):
                if st.button("Next →", key="tour_next"):
                    st.session_state.tour_step += 1
                    st.rerun()
            else:
                if st.button("Finish Tour", key="tour_finish"):
                    st.session_state.show_tour = False
                    st.session_state.tour_step = 0
                    st.rerun()
        with col3:
            if st.button("Skip Tour", key="tour_skip"):
                st.session_state.show_tour = False
                st.session_state.tour_step = 0
                st.rerun()

        st.markdown("---")

# Initialize uploader key for reset functionality
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# File upload - allow multiple files
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True, key=f"file_uploader_{st.session_state.uploader_key}")

if uploaded_files:
    try:
        # Initialize session state for datasets
        if "datasets" not in st.session_state:
            st.session_state.datasets = {}

        # Clear existing datasets when new files are uploaded
        current_file_names = [f.name.replace('.csv', '') for f in uploaded_files]
        st.session_state.datasets = {}
        # Mark that we have data uploaded for routing logic
        st.session_state.has_data = True

        # Load all uploaded files
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name.replace('.csv', '')
            df_temp = pd.read_csv(uploaded_file)
            st.session_state.datasets[file_name] = df_temp

        st.success(f"Loaded {len(st.session_state.datasets)} dataset(s): {list(st.session_state.datasets.keys())}")

        # Auto-detect common columns for potential joins
        common_cols = set()
        if len(st.session_state.datasets) > 1:
            all_columns = [set(dataset.columns) for dataset in st.session_state.datasets.values()]
            common_cols = set.intersection(*all_columns)
            if common_cols:
                st.info(f"Common columns detected for potential joins: {', '.join(common_cols)}")

        # Dataset selection and configuration
        current_df = None
        dataset_context = ""

        if len(st.session_state.datasets) > 1:
            st.subheader("Dataset Selection")

            analysis_mode = st.radio(
                "Choose analysis mode:",
                ["Single Dataset", "Auto-join Datasets", "Compare Datasets"],
                key="analysis_mode"
            )

            if analysis_mode == "Single Dataset":
                selected_dataset = st.selectbox(
                    "Select dataset for analysis:",
                    list(st.session_state.datasets.keys()),
                    key="dataset_selector"
                )

                # Update unique categories for selected dataset
                st.session_state.unique_categories = {}
                for col in current_df.select_dtypes(include=['object', 'category']).columns:
                    unique_vals = current_df[col].dropna().unique()
                    if len(unique_vals) <= 50:  # Only store if reasonable number of unique values
                        st.session_state.unique_categories[col] = unique_vals.tolist()

                current_df = st.session_state.datasets[selected_dataset]
                dataset_context = f"Using dataset '{selected_dataset}' with {current_df.shape[0]} rows and {current_df.shape[1]} columns. Columns: {list(current_df.columns)}"

            elif analysis_mode == "Auto-join Datasets":
                if common_cols:
                    st.info(f"Common columns available: {', '.join(sorted(common_cols))}")
                    join_column = st.selectbox(
                        "Select column to join on:",
                        sorted(list(common_cols)),
                        key="join_column"
                    )
                else:
                    st.warning("No common columns found across all datasets.")
                    # Show all unique columns for manual selection
                    all_unique_cols = set()
                    for dataset in st.session_state.datasets.values():
                        all_unique_cols.update(dataset.columns)

                    join_column = st.selectbox(
                        "Select column to attempt join (may have missing values):",
                        sorted(list(all_unique_cols)),
                        key="join_column_manual"
                    )

                # Join type selection
                join_type = st.selectbox(
                    "Select join type:",
                    ["outer", "inner", "left", "right"],
                    index=0,
                    help="outer: keep all records, inner: only matching records, left: keep first dataset records, right: keep second dataset records",
                    key="join_type"
                )

                # Perform join
                datasets_list = list(st.session_state.datasets.items())
                current_df = datasets_list[0][1].copy()

                for name, dataset in datasets_list[1:]:
                    if join_column in dataset.columns:
                        current_df = current_df.merge(dataset, on=join_column, how=join_type, suffixes=('', f'_{name}'))
                    else:
                        st.warning(f"Column '{join_column}' not found in dataset '{name}', skipping join.")

                dataset_context = f"Joined datasets on '{join_column}' using {join_type} join: {', '.join(st.session_state.datasets.keys())}. Result: {current_df.shape[0]} rows, {current_df.shape[1]} columns"

                # Update unique categories for joined dataset
                st.session_state.unique_categories = {}
                for col in current_df.select_dtypes(include=['object', 'category']).columns:
                    unique_vals = current_df[col].dropna().unique()
                    if len(unique_vals) <= 50:  # Only store if reasonable number of unique values
                        st.session_state.unique_categories[col] = unique_vals.tolist()

                # Store reference to current DataFrame
                st.session_state.current_df = current_df

            else:  # Compare Datasets
                current_df = None
                dataset_context = f"Multiple datasets available for comparison: {', '.join(st.session_state.datasets.keys())}"
        else:
            # Single dataset
            current_df = list(st.session_state.datasets.values())[0]
            dataset_name = list(st.session_state.datasets.keys())[0]
            dataset_context = f"Using dataset '{dataset_name}' with {current_df.shape[0]} rows and {current_df.shape[1]} columns. Columns: {list(current_df.columns)}"

            # Update unique categories for current dataset (smart category matching)
            if current_df is not None:
                st.session_state.unique_categories = {}
                for col in current_df.select_dtypes(include=['object', 'category']).columns:
                    unique_vals = current_df[col].dropna().unique()
                    if len(unique_vals) <= 50:  # Only store if reasonable number of unique values
                        st.session_state.unique_categories[col] = unique_vals.tolist()

            # Store reference to current DataFrame for analysis
            st.session_state.current_df = current_df

        # Display dataset overview
        with st.expander("Dataset Overview"):
            st.write(dataset_context)
            if current_df is not None:
                st.dataframe(current_df.head())
                st.write(f"Data types: {dict(current_df.dtypes)}")
            else:
                for name, dataset in st.session_state.datasets.items():
                    st.write(f"**{name}**: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
                    st.dataframe(dataset.head(3))
                    st.write("---")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.subheader("Ask AI Assistant")

        # Quick action buttons for common analyses
        st.write("**Quick Actions:**")
        col1, col2, col3, col4 = st.columns(4)

        # Initialize current query in session state
        if "current_query" not in st.session_state:
            st.session_state.current_query = ""

        # Handle quick action clicks
        quick_action_clicked = False
        with col1:
            if st.button("📊 Data Summary", key="quick_summary"):
                st.session_state.current_query = "provide a comprehensive summary of this dataset"
                quick_action_clicked = True
        with col2:
            if st.button("🔗 Find Correlations", key="quick_correlations"):
                st.session_state.current_query = "show correlation matrix for numeric columns only and identify strongest relationships"
                quick_action_clicked = True
        with col3:
            if st.button("📈 Key Trends", key="quick_trends"):
                st.session_state.current_query = "create visualizations showing key trends in the data"
                quick_action_clicked = True
        with col4:
            if st.button("⚠️ Data Quality", key="quick_quality"):
                st.session_state.current_query = "analyze data quality issues and missing values"
                quick_action_clicked = True

        # Chat History Section (collapsed by default)
        if st.session_state.messages:
            with st.expander(f"📜 Chat History ({len(st.session_state.messages)} messages)", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Clear History", type="secondary", key="clear_history_btn"):
                        st.session_state.messages = []
                        st.rerun()
                
                # Display chat messages
                for i, message in enumerate(st.session_state.messages):
                    with st.chat_message(message["role"]):
                        if message.get("type") == "text":
                            st.write(message["content"])
                        elif message.get("type") == "chart":
                            history_chart_key = f"history_chart_{i}_{hash(str(message.get('content', '')))}"
                            st.plotly_chart(message["chart"], use_container_width=True, key=history_chart_key)
                        elif message.get("type") == "table":
                            st.dataframe(message["data"])
                        elif message.get("type") == "metric":
                            st.metric(message["label"], message["value"])
                        else:
                            st.write(message["content"])

        # User query input with Enter key support
        query = st.chat_input(
            "Type your question and press Enter...",
            key="query_input"
        )

        # Handle query submission - st.chat_input returns value when user submits
        if query:
            st.session_state.current_query = query
        
        # For quick actions, use the stored query and show what's being processed
        if quick_action_clicked:
            query = st.session_state.current_query
            st.info(f"🔍 Processing: {query}")

        # Process analysis if we have a query (either from chat input or quick actions)
        if query or quick_action_clicked:
            # Check for advanced analytics requests that should be blocked
            advanced_keywords = [
                'random forest', 'randomforest', 'rf model', 'machine learning', 'ml model',
                'predictive model', 'prediction model', 'predict', 'neural network', 'deep learning',
                'classification model', 'regression model', 'svm', 'support vector',
                'decision tree', 'gradient boosting', 'xgboost', 'lightgbm',
                'ensemble model', 'cross validation', 'feature importance', 'model training',
                'hyperparameter', 'overfitting', 'underfitting', 'accuracy score',
                'precision', 'recall', 'f1 score', 'roc curve', 'auc score',
                'feature engineering', 'feature selection', 'clustering', 'k-means', 'pca',
                'logistic regression', 'linear regression', 'train model', 'build model'
            ]

            is_blocked = False
            if query and any(keyword in query.lower() for keyword in advanced_keywords):
                blocked_message = "Sorry, I cannot answer this question."

                # Add to messages history
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "type": "text"
                })

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": blocked_message,
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "type": "text"
                })

                st.warning(blocked_message)
                #st.stop()
                is_blocked = True

            if not is_blocked:

                with st.spinner("Analyzing..."):
                    try:
                        # Initialize Azure OpenAI client using LangChain's dedicated Azure client
                        client = AzureChatOpenAI(
                                openai_api_key=api_key,
                                azure_endpoint=endpoint,
                                openai_api_version="2024-10-21",
                               # max_tokens=1500,  # Increased from 800 to handle larger datasets
                                temperature=0
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to initialize Azure OpenAI client: {str(e)}")
                        st.info("Please verify your Azure OpenAI configuration:")
                        st.stop()

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
                
                # Determine intent via GPT-4o (simplified for data analysis)
                intent_prompt = f'''You are a Principal Data Scientist with 10+ years of experience. The user asked: "{query}"

                As an expert analyst, determine the most appropriate approach and respond with exactly one word:
                - "insight" - if asking for strategic business insights, executive summary, or high-level analysis that requires professional interpretation of the uploaded data
                - "code" - if asking for specific visualizations, calculations, statistical analysis, statistical tests, correlations, or data manipulations that require code execution

                Examples:
                {code_feedback_format}
                {insight_feedback_format}

                Only respond with one word: insight or code'''

                intent_response = client.invoke(intent_prompt)
                intent_content = intent_response.content if hasattr(intent_response, 'content') else str(intent_response)
                intent = str(intent_content).strip().lower() if intent_content else "code"

                st.session_state.last_query = query
                st.session_state.last_intent = intent

                # Branch on intent
                if intent == "insight":
                    # Generate enhanced business insights
                    st.info("🧠 Generating strategic business insights...")
                    if current_df is not None:
                        # Create detailed data context for better insights
                        numeric_cols = current_df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = current_df.select_dtypes(include=['object', 'category']).columns.tolist()

                        # ENHANCED: Run code-driven calculations under the hood for insights
                        computed_metrics = {}
                        try:
                            # Calculate key metrics automatically
                            if numeric_cols:
                                # Basic statistics
                                stats_summary = current_df[numeric_cols].describe().round(2)
                                computed_metrics['basic_stats'] = stats_summary.to_dict()

                                # Correlations if multiple numeric columns
                                if len(numeric_cols) > 1:
                                    corr_matrix = current_df[numeric_cols].corr().round(3)
                                    # Find strongest correlations
                                    corr_pairs = []
                                    for i in range(len(corr_matrix.columns)):
                                        for j in range(i+1, len(corr_matrix.columns)):
                                            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                                            corr_val = corr_matrix.iloc[i, j]
                                            if abs(corr_val) > 0.3:  # Only meaningful correlations
                                                corr_pairs.append((col1, col2, corr_val))
                                    computed_metrics['correlations'] = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]

                                # Distribution insights
                                for col in numeric_cols:
                                    col_data = current_df[col].dropna()
                                    if len(col_data) > 0:
                                        computed_metrics[f'{col}_insights'] = {
                                            'mean': col_data.mean().round(2),
                                            'median': col_data.median().round(2),
                                            'std': col_data.std().round(2),
                                            'skewness': col_data.skew().round(2),
                                            'outliers_count': len(col_data[col_data > col_data.mean() + 3*col_data.std()])
                                        }

                            # Categorical insights
                            if categorical_cols:
                                for col in categorical_cols: 
                                    value_counts = current_df[col].value_counts()
                                    computed_metrics[f'{col}_distribution'] = {
                                        'unique_count': len(value_counts),
                                        'top_categories': value_counts.head(5).to_dict(),
                                        'missing_count': current_df[col].isnull().sum()
                                    }

                        except Exception as e:
                            st.warning(f"Some calculations could not be completed: {e}")

                        # Build enhanced data summary with computed metrics
                        data_summary = f"""Dataset Overview:
                        - Shape: {current_df.shape[0]} rows, {current_df.shape[1]} columns
                        - Numeric columns: {numeric_cols[:5]}  # Show first 5 to avoid overwhelming
                        - Categorical columns: {categorical_cols[:5]}
                        - Missing values: {current_df.isnull().sum().sum()} total across all columns"""

                        # Add computed insights to data summary
                        if computed_metrics:
                            data_summary += "\n\nComputed Insights:"

                            # Add correlation insights
                            if 'correlations' in computed_metrics and computed_metrics['correlations']:
                                data_summary += "\n- Key Correlations:"
                                for col1, col2, corr_val in computed_metrics['correlations'][:5]:
                                    strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.5 else "Weak"
                                    direction = "positive" if corr_val > 0 else "negative"
                                    data_summary += f"\n  • {col1} vs {col2}: {strength} {direction} correlation ({corr_val:.3f})"

                            # Add distribution insights
                            for key, value in computed_metrics.items():
                                if key.endswith('_insights'):
                                    col_name = key.replace('_insights', '')
                                    data_summary += f"\n- {col_name}: Mean={value['mean']}, Median={value['median']}, Std={value['std']}, Outliers={value['outliers_count']}"
                                elif key.endswith('_distribution'):
                                    col_name = key.replace('_distribution', '')
                                    top_cat = list(value['top_categories'].keys())[0] if value['top_categories'] else "N/A"
                                    data_summary += f"\n- {col_name}: {value['unique_count']} unique values, most common: '{top_cat}'"

                        # Add statistical summary for numeric columns (keep existing behavior)
                        if numeric_cols:
                            stats_summary = current_df[numeric_cols].describe().round(2)  # Limit to first 3 numeric columns
                            data_summary += f"\n\nKey Statistics (All Numeric Columns):\n{stats_summary.to_string()}"

                        insights_prompt = f'''You are a Principal Data Scientist with 10+ years of experience and a seasoned Business Analyst with extensive real-world business expertise. Your role is to provide professional, strategic insights that bridge technical analysis with actionable business recommendations.

The user asked: "{query}"

Drawing from your extensive experience, provide a comprehensive analysis that demonstrates deep understanding of both technical analysis and business strategy.

{data_summary}

Sample data:
{current_df.head(3).to_string()}

IMPORTANT: I have already performed key calculations under the hood. Use the computed insights above to provide data-driven evidence with specific numbers, correlations, and statistical findings. Reference these exact calculated values in your analysis.

Professional Analysis Framework:
1. Executive Summary: Start with key findings that directly answer: "{query}" - use the computed metrics above
2. Data-Driven Evidence: Use the specific calculated values, correlations, and statistical trends from the computed insights section
3. Business Context: Interpret the calculated findings through the lens of real-world business implications

Provide your analysis as a structured, professional report that executives and business stakeholders would find valuable. Ground every insight in the specific calculated values provided above.'''
                    else:
                        # Generate enhanced business insights
                        st.info("🧠 Generating strategic business insights...")
                        insights_prompt = f'''You are a Principal Data Scientist with 10+ years of experience and a seasoned Business Analyst with extensive real-world business expertise. Your role is to provide professional, strategic insights that bridge technical analysis with actionable business recommendations.

The user asked: "{query}"

Drawing from your extensive experience in multi-dataset analysis, provide a comprehensive cross-dataset analysis that demonstrates deep understanding of both technical analysis and business strategy.

{dataset_context}

Professional Multi-Dataset Analysis Framework:
1. Executive Summary: Start with key findings from cross-dataset analysis that directly answer: "{query}"
2. Data-Driven Evidence: Use actual data values, statistics, and trends from all datasets
3. Business Context: Interpret findings through the lens of real-world business implications

Provide your analysis as a structured, professional report that executives and business stakeholders would find valuable for strategic decision-making.'''

                    # Create new client with higher temperature for insights
                    insights_client = AzureChatOpenAI(
                                    openai_api_key=api_key,
                                    azure_endpoint=endpoint,
                                    openai_api_version="2024-10-21",
                                    max_tokens=1500,  # Increased from 800 to handle larger datasets
                                    temperature=0)

                    response = insights_client.invoke(insights_prompt)
                    insights_text = response.content

                    # --- Self-check & possible rewrite for insights (ADD THIS) ---
                    try:
                        metrics_text_for_check = data_summary  # this already includes computed numbers/correlations
                        checked_insights = self_check_and_rewrite_insight(insights_text, metrics_text_for_check)
                        insights_text = checked_insights  # replace with corrected version if provided
                        if checked_insights != response.content:
                            st.info("🔍 Analysis was automatically reviewed and enhanced for accuracy")
                    except Exception as e:
                        st.warning(f"Insight self-check skipped: {e}")
                    # --- End self-check ---

                    st.subheader("Analysis Results:")
                    st.write(insights_text)

                    # Generate additional narrative summary for insights
                    try:
                        insight_summary = generate_insight_narrative_summary(insights_text, query, api_key, endpoint)
                        if insight_summary:
                            st.subheader("📋 Key Takeaways:")
                            st.write(insight_summary)
                            st.session_state.messages.append({"role": "assistant", "content": insight_summary, "type": "text"})
                    except Exception as e:
                        st.warning(f"Could not generate key takeaways: {e}")

                    # Save to chat history
                    st.session_state.messages.append({"role": "user", "content": query, "type": "text"})
                    st.session_state.messages.append({"role": "assistant", "content": insights_text, "type": "text"})

                else:
                    # Generate and execute analysis code with self-repair
                    st.info("⚙️ Generating and executing analysis code...")

                    # Code generation branch - handles ALL data analysis tasks
                    if current_df is not None:
                        # Get column information for better code generation
                        columns_list = list(current_df.columns)
                        sample_data = current_df.head(2).to_string()

                        # Prepare unique categories context for smart matching
                        unique_categories = st.session_state.get('unique_categories', {})
                        categories_info = ""
                        if unique_categories:
                            categories_info = "\nUnique categorical values in dataset:\n"
                            for col, values in unique_categories.items():
                                categories_info += f"- {col}: {', '.join(str(v) for v in values[:10])}{'...' if len(values) > 10 else ''}\n"
                            categories_info += "\nNote: When users mention abbreviated names (like 'FL' for Florida, 'JNJ' for Johnson and Johnson), match them to the full names in the data above."

                        prompt = f'''You are a Principal Data Scientist with 10+ years of experience and a seasoned Business Analyst with extensive real-world business expertise. Your role is to provide professional, insightful data analysis that bridges technical findings with business value.

                        Generate Python code to analyze: "{query}"

                        As an expert analyst, handle these analysis types with business context:
                        - Market share calculations (percentage of total sales/revenue by category, manufacturer, segment, etc.)
                        - Business metrics calculations (percentages, ratios, growth rates, market penetration)
                        - Statistical tests (t-tests, correlations, ANOVA, chi-square, etc.) with practical business interpretations - ALWAYS use scipy.stats and return result dictionary
                        - Data visualizations (histograms, scatter plots, heatmaps, etc.) that tell compelling business stories
                        - Data summaries and calculations that highlight key business metrics
                        - Basic descriptive statistics with actionable insights for decision-makers

                        IMPORTANT: For statistical test requests (containing words like "test_statistics", "test", "statistical", "significance", "compare groups"):
                        - DO NOT create visualizations unless specifically requested
                        - Focus on performing the actual statistical test using scipy.stats
                        - Return a result dictionary with test statistics, p-values, and group statistics

                        Dataset Information:
                        - Available columns: {columns_list}
                        - Sample data:
                        {sample_data}{categories_info}

                        Use the existing DataFrame variable 'df' - DO NOT create sample data.
                        Available libraries: pandas as pd, plotly.express as px, numpy as np, scipy.stats as stats

                        IMPORTANT: Use the EXACT column names from the available columns list above.

                        Specific guidelines:
                        - For "market share" or "share" calculations: Calculate percentage = (subset_total / overall_total) * 100, create summary table with percentages
                        - For "distribution" requests: Use px.histogram(df, x='actual_column_name')
                        - For "map" requests: Use px.scatter_mapbox() or px.choropleth() if geographic data exists
                        - For correlations: Filter to numeric columns first using df.select_dtypes(include=['number']). If no numeric columns exist, inform the user that correlations require numeric data. Otherwise use .corr() then px.imshow() for heatmaps
                        - For statistical tests: Use scipy.stats (t-test, ANOVA, chi-square, etc.)
                        - For comparisons: Use px.bar() or px.box()
                        - **For Sankey diagrams**:
                                1. Encode node names as integer codes:
                                    ```python
                                    src = df['source_column'].astype('category').cat.codes
                                    tgt = df['target_column'].astype('category').cat.codes
                                    ```
                                2. Build your labels list:
                                    ```python
                                    labels = list(df['source_column'].unique()) + list(df['target_column'].unique())
                                    ```
                                3. Pick your weights (must match `src`/`tgt` length):
                                    ```python
                                    values = df['value_column']
                                    ```
                                4. Wrap the Sankey trace in a Figure **and assign it to `fig`**:
                                    ```python
                                    fig = go.Figure(data=[go.Sankey(
                                        node=dict(label=labels),
                                        link=dict(source=src, target=tgt, value=values)
                                    )])
                                    fig.update_layout(title_text="Sankey Diagram", font_size=10)
                                    ```
                                5. **Do not** use any other variable name—Streamlit looks for `fig`.

                        Response format:
                        - For single visualizations: create a 'fig' variable using plotly express
                        - For multiple visualizations: create a 'fig' variable with subplots using plotly.subplots.make_subplots
                        - For tables/summaries: create a result DataFrame and assign to 'result'  
                        - For calculations: assign the final value to 'result'
                        - For statistical tests: create a results dictionary with test statistics and p-values

                        IMPORTANT: DO NOT include any .show() calls (fig.show(), plt.show(), etc.)
                        The visualization will be displayed automatically.
                        IMPORTANT: Always use a single 'fig' variable for all visualizations - combine multiple charts using subplots if needed.

                        ALWAYS check the column names and use exact matches from: {columns_list}

                        Return only Python code that uses the existing 'df' variable. Do not include display commands.'''
                    else:
                        # Generate and execute analysis code with self-repair
                        st.info("⚙️ Generating and executing analysis code...")
                        # Multiple datasets scenario
                        datasets_info = ""
                        all_categories = {}
                        
                        for name, dataset in st.session_state.datasets.items():
                            datasets_info += f"\nDataset '{name}': {dataset.shape[0]} rows, {dataset.shape[1]} columns\n"
                            datasets_info += f"Columns: {list(dataset.columns)}\n"
                            datasets_info += f"Sample:\n{dataset.head(2).to_string()}\n"
                            
                            # Collect categories for this dataset
                            for col in dataset.select_dtypes(include=['object', 'category']).columns:
                                unique_vals = dataset[col].dropna().unique()
                                if len(unique_vals) <= 50:
                                    if col not in all_categories:
                                        all_categories[col] = {}
                                    all_categories[col][name] = unique_vals.tolist()

                        # Prepare categories context
                        categories_info = ""
                        if all_categories:
                            categories_info = "\nCategorical values across datasets:\n"
                            for col, datasets in all_categories.items():
                                categories_info += f"- Column '{col}':\n"
                                for dataset_name, values in datasets.items():
                                    categories_info += f"  {dataset_name}: {', '.join(str(v) for v in values[:8])}{'...' if len(values) > 8 else ''}\n"
                            categories_info += "\nNote: When users mention abbreviated names (like 'FL' for Florida, 'JNJ' for Johnson and Johnson), match them to the full names in the data above.\n"

                        prompt = f'''You are a Principal Data Scientist with 10+ years of experience. Generate Python code to analyze: "{query}"

                        Available datasets in st.session_state.datasets:
                        {datasets_info}{categories_info}

                        Access datasets using: datasets['dataset_name']

                        For cross-dataset analysis:
                        - Compare statistics, distributions, or trends across datasets
                        - Look for patterns, outliers, or insights that emerge from multiple datasets
                        - Create visualizations that highlight differences or similarities

                        Available libraries: pandas as pd, plotly.express as px, numpy as np, scipy.stats as stats

                        Return only Python code. Assign results to 'result' or figures to 'fig'.'''

                    response = client.invoke(prompt)
                    code = response.content if hasattr(response, 'content') else str(response)
                    code = str(code) if code else ""

                    # Strip markdown fences
                    if "```python" in code:
                        code = code.split("```python")[1].split("```")[0].strip()
                    elif "```" in code:
                        code = code.split("```")[1].split("```")[0].strip()

                    #with st.expander("Show Generated Python Code"):
                    #    st.code(code, language="python")

                    st.subheader("Result:")

                    # Setup execution environment with basic libraries                    
                    if current_df is not None:
                        exec_env = {
                            'df': current_df, 
                            'pd': pd, 
                            'px': px, 
                            'np': np,
                            'stats': stats,
                            'go': go,
                            'make_subplots': make_subplots
                        }
                    else:
                        exec_env = {
                            'datasets': st.session_state.datasets, 
                            'pd': pd, 
                            'px': px,
                            'np': np,
                            'stats': stats,
                            'go': go,
                            'make_subplots': make_subplots
                        }

                    try:
                        # Use self-repair system for code execution
                        obs, final_code = run_code_with_self_repair(str(code) if code else "", exec_env, query)

                        # Show the final code actually used (helpful transparency)
                        with st.expander("🔍 Show Generated Python Code"):
                            st.code(final_code or code, language="python")
                            if final_code and final_code != code:
                                st.info("🔧 Code was automatically repaired to fix execution errors")

                        # Save and display results
                        st.session_state.messages.append({"role": "user", "content": query, "type": "text"})
                        found = False

                        if "fig" in obs:
                            chart_key = f"chart_{len(st.session_state.messages)}_{hash(query)}"
                            st.plotly_chart(obs["fig"], use_container_width=True, key=chart_key)
                            st.session_state.messages.append({"role": "assistant", "type": "chart", "chart": obs["fig"], "content": f"Generated visualization for: {query}", "intent": "code"})
                            found = True

                        if "result" in obs:
                            result = obs["result"]
                            # Handle the result from the self-repair system
                            if hasattr(result, 'to_frame') or hasattr(result, 'index'):
                                st.dataframe(result)
                                st.session_state.messages.append({"role": "assistant", "type": "table", "data": result, "content": f"Data table for: {query}"})
                            elif isinstance(result, (int, float)):
                                st.metric("Result", f"{result:.2f}")
                                st.session_state.messages.append({"role": "assistant", "type": "metric", "label": "Result", "value": f"{result:.2f}", "content": f"Calculation result for: {query}"})
                            elif isinstance(result, dict):
                                # Handle statistical test results with interpretations
                                st.subheader("Statistical Test Results:")

                                # Separate descriptive statistics from test statistics
                                test_stats = {}
                                descriptive_stats = {}

                                for key, value in result.items():
                                    key_str = str(key).lower() if not isinstance(key, str) else str(key).lower()
                                    if any(stat in key_str for stat in ['mean', 'std', 'sample_size', 'count', 'size']):
                                        descriptive_stats[key] = value
                                    else:
                                        test_stats[key] = value

                                # Display descriptive statistics first if available
                                if descriptive_stats:
                                    st.subheader("Group Statistics:")

                                    # Create columns for group comparisons
                                    groups = {}
                                    for key, value in descriptive_stats.items():
                                        group_name = key.split('_')[0] if '_' in key else 'Group'
                                        stat_type = '_'.join(key.split('_')[1:]) if '_' in key else key

                                        if group_name not in groups:
                                            groups[group_name] = {}
                                        groups[group_name][stat_type] = value

                                    if len(groups) > 1:
                                        cols = st.columns(len(groups))
                                        for i, (group_name, stats) in enumerate(groups.items()):
                                            with cols[i]:
                                                st.markdown(f"**{group_name}**")
                                                for stat_name, stat_value in stats.items():
                                                    if isinstance(stat_value, (int, float)):
                                                        if 'mean' in stat_name.lower():
                                                            st.metric(f"Mean", f"{stat_value:.2f}")
                                                        elif 'std' in stat_name.lower():
                                                            st.metric(f"Std Dev", f"{stat_value:.2f}")
                                                        elif 'size' in stat_name.lower() or 'count' in stat_name.lower():
                                                            st.metric(f"Sample Size", f"{int(stat_value)}")
                                                        else:
                                                            st.metric(safe_key_format(stat_name), f"{stat_value:.3f}")
                                    else:
                                        # Single group or general descriptive stats
                                        for key, value in descriptive_stats.items():
                                            if isinstance(value, (int, float)):
                                                st.metric(safe_key_format(key), f"{value:.3f}")

                                # Display test statistics
                                if test_stats:
                                    st.subheader("Test Statistics:")
                                    for key, value in test_stats.items():
                                        st.write(value)
                                        if isinstance(value, (int, float)):
                                            st.metric(safe_key_format(key), f"{value:.4f}")
                                            #st.metric(key, f"{value:.4f}")
                                        else:
                                            st.write(f"**{safe_key_format(key)}:** {value}")

                                # Add interpretations for common statistical tests
                                st.subheader("Interpretation:")
                                if 'p_value' in result or 'P-value' in result:
                                    p_val = result.get('p_value', result.get('P-value', 0))
                                    if p_val < 0.001:
                                        st.success("**Highly Significant** (p < 0.001): Very strong evidence of a difference between groups")
                                    elif p_val < 0.01:
                                        st.success("**Significant** (p < 0.01): Strong evidence of a difference between groups") 
                                    elif p_val < 0.05:
                                        st.warning("**Marginally Significant** (p < 0.05): Moderate evidence of a difference between groups")
                                    else:
                                        st.info("**Not Significant** (p ≥ 0.05): No significant difference found between groups")

                                if 't_statistic' in result or 'T-statistic' in result:
                                    t_val = result.get('t_statistic', result.get('T-statistic', 0))
                                    if abs(t_val) > 2:
                                        st.info(f"**Large effect size** (|t| = {abs(t_val):.2f}): Substantial practical difference")
                                    else:
                                        st.info(f"**Small to moderate effect size** (|t| = {abs(t_val):.2f}): Limited practical difference")

                                # Add practical significance interpretation if means are available
                                if any('mean' in str(key).lower() for key in descriptive_stats.keys()):
                                    mean_keys = [k for k in descriptive_stats.keys() if 'mean' in str(k).lower()]
                                    if len(mean_keys) == 2:
                                        mean1, mean2 = descriptive_stats[mean_keys[0]], descriptive_stats[mean_keys[1]]
                                        diff = abs(mean1 - mean2)
                                        group1 = str(mean_keys[0]).replace('_mean', '').replace('_', ' ').title()
                                        group2 = str(mean_keys[1]).replace('_mean', '').replace('_', ' ').title()
                                        st.info(f"**Practical Difference**: {group1} average is {diff:.2f} units {'higher' if mean1 > mean2 else 'lower'} than {group2}")

                                st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"Statistical Test Results: {result}"})
                            else:
                                # Format result nicely instead of raw display
                                st.subheader("Result:")
                                if isinstance(result, dict):
                                    # Format dictionary results properly
                                    for key, value in result.items():
                                        if isinstance(value, dict):
                                            # Handle nested dictionaries (like group statistics)
                                            st.markdown(f"**{safe_key_format(key)}:**")
                                            for sub_key, sub_value in value.items():
                                                if isinstance(sub_value, (int, float)):
                                                    st.write(f"  • {safe_key_format(sub_key)}: {sub_value:.4f}")
                                                else:
                                                    st.write(f"  • {safe_key_format(sub_key)}: {sub_value}")
                                        elif isinstance(value, (int, float)):
                                            st.metric(safe_key_format(key), f"{value:.4f}")
                                        else:
                                            st.write(f"**{safe_key_format(key)}:** {value}")
                                else:
                                    st.write(result)
                                st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"Result: {result}"})
                            found = True



                        # Also check for 'results' variable (plural, commonly used in statistical tests)
                        if 'results' in exec_env and not found:
                            results = exec_env['results']
                            if isinstance(results, dict):
                                # Handle statistical test results with interpretations
                                st.subheader("Statistical Test Results:")

                                # Separate descriptive statistics from test statistics
                                test_stats = {}
                                descriptive_stats = {}

                                for key, value in results.items():
                                    key_str = str(key).lower() if not isinstance(key, str) else str(key).lower()
                                    if any(stat in key_str for stat in ['mean', 'std', 'sample_size', 'count', 'size']):
                                        descriptive_stats[key] = value
                                    else:
                                        test_stats[key] = value

                                # Display descriptive statistics first if available
                                if descriptive_stats:
                                    st.subheader("Group Statistics:")

                                    # Create columns for group comparisons
                                    groups = {}
                                    for key, value in descriptive_stats.items():
                                        group_name = key.split('_')[0] if '_' in key else 'Group'
                                        stat_type = '_'.join(key.split('_')[1:]) if '_' in key else key

                                        if group_name not in groups:
                                            groups[group_name] = {}
                                        groups[group_name][stat_type] = value

                                    if len(groups) > 1:
                                        cols = st.columns(len(groups))
                                        for i, (group_name, stats) in enumerate(groups.items()):
                                            with cols[i]:
                                                st.markdown(f"**{group_name}**")
                                                for stat_name, stat_value in stats.items():
                                                    if isinstance(stat_value, (int, float)):
                                                        if 'mean' in stat_name.lower():
                                                            st.metric(f"Mean", f"{stat_value:.2f}")
                                                        elif 'std' in stat_name.lower():
                                                            st.metric(f"Std Dev", f"{stat_value:.2f}")
                                                        elif 'size' in stat_name.lower() or 'count' in stat_name.lower():
                                                            st.metric(f"Sample Size", f"{int(stat_value)}")
                                                        else:
                                                            st.metric(safe_key_format(stat_name), f"{stat_value:.3f}")
                                    else:
                                        # Single group or general descriptive stats
                                        for key, value in descriptive_stats.items():
                                            if isinstance(value, (int, float)):
                                                st.metric(safe_key_format(key), f"{value:.3f}")

                                # Display test statistics
                                if test_stats:
                                    st.subheader("Test Statistics:")
                                    for key, value in test_stats.items():
                                        if isinstance(value, (int, float)):
                                            st.metric(safe_key_format(key), f"{value:.4f}")
                                        else:
                                            st.write(f"**{safe_key_format(key)}:** {value}")

                                # Add interpretations for common statistical tests
                                st.subheader("Interpretation:")
                                if 'p_value' in results or 'P-value' in results:
                                    p_val = results.get('p_value', results.get('P-value', 0))
                                    if p_val < 0.001:
                                        st.success("**Highly Significant** (p < 0.001): Very strong evidence of a difference between groups")
                                    elif p_val < 0.01:
                                        st.success("**Significant** (p < 0.01): Strong evidence of a difference between groups") 
                                    elif p_val < 0.05:
                                        st.warning("**Marginally Significant** (p < 0.05): Moderate evidence of a difference between groups")
                                    else:
                                        st.info("**Not Significant** (p ≥ 0.05): No significant difference found between groups")

                                if 't_statistic' in results or 'T-statistic' in results:
                                    t_val = results.get('t_statistic', results.get('T-statistic', 0))
                                    if abs(t_val) > 2:
                                        st.info(f"**Large effect size** (|t| = {abs(t_val):.2f}): Substantial practical difference")
                                    else:
                                        st.info(f"**Small to moderate effect size** (|t| = {abs(t_val):.2f}): Limited practical difference")

                                # Add practical significance interpretation if means are available
                                if any('mean' in str(key).lower() for key in descriptive_stats.keys()):
                                    mean_keys = [k for k in descriptive_stats.keys() if 'mean' in str(k).lower()]
                                    if len(mean_keys) == 2:
                                        mean1, mean2 = descriptive_stats[mean_keys[0]], descriptive_stats[mean_keys[1]]
                                        diff = abs(mean1 - mean2)
                                        group1 = str(mean_keys[0]).replace('_mean', '').replace('_', ' ').title()
                                        group2 = str(mean_keys[1]).replace('_mean', '').replace('_', ' ').title()
                                        st.info(f"**Practical Difference**: {group1} average is {diff:.2f} units {'higher' if mean1 > mean2 else 'lower'} than {group2}")

                                st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"Statistical Test Results: {results}"})
                                found = True
                            else:
                                # Format results nicely instead of showing raw dictionary
                                st.subheader("Results:")
                                if isinstance(results, dict):
                                    # Format dictionary results properly
                                    for key, value in results.items():
                                        if isinstance(value, dict):
                                            # Handle nested dictionaries (like group statistics)
                                            st.markdown(f"**{safe_key_format(key)}:**")
                                            for sub_key, sub_value in value.items():
                                                if isinstance(sub_value, (int, float)):
                                                    st.write(f"  • {safe_key_format(sub_key)}: {sub_value:.4f}")
                                                else:
                                                    st.write(f"  • {safe_key_format(sub_key)}: {sub_value}")
                                        elif isinstance(value, (int, float)):
                                            st.metric(safe_key_format(key), f"{value:.4f}")
                                        else:
                                            st.write(f"**{safe_key_format(key)}:** {value}")
                                else:
                                    st.write(results)
                                st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"Results: {results}"})
                                found = True

                        # Handle errors from self-repair system
                        if obs.get("errors"):
                            st.error(f"Execution error: {obs['errors']}")
                        if obs.get("max_retries_exceeded"):
                            st.error(f"Self-repair failed after {obs.get('repair_attempts', 0) + 1} attempts")
                            st.info("💡 The AI attempted to fix the code automatically multiple times but was unsuccessful")
                            if obs.get("repair_history"):
                                with st.expander("🔧 View Repair Attempts"):
                                    for attempt in obs["repair_history"]:
                                        st.write(f"**Attempt {attempt['attempt']}:** {attempt['error']}")
                        elif obs.get("repair_attempts"):
                            st.info(f"🔧 Code was automatically repaired after {obs['repair_attempts']} attempts")

                        #if found:
                        # Generate narrative summary after successful code execution
                        #    try:
                        #       narrative_summary = generate_narrative_from_results(exec_env, query, api_key, endpoint)
                        #       if narrative_summary:
                        #          st.subheader("📋 Executive Summary:")
                        #          st.write(narrative_summary)
                        #          st.session_state.messages.append({"role": "assistant", "content": narrative_summary, "type": "text"})
                        #  except Exception as e:
                        #        st.warning(f"Could not generate summary: {e}")
                        #    st.success("✅ Analysis complete with agentic superpowers!")
                        #else:
                        #    if not obs.get("errors") and not obs.get("repair_error"):
                        #        st.info("Code executed successfully")

                    except Exception as e:
                        st.error(f"Self-repair system error: {e}")
                    
        # ========== ALWAYS-ON FEEDBACK (ADD THIS BLOCK) ==========
        last_query = st.session_state.get("last_query")
        last_intent = st.session_state.get("last_intent")

        if last_query and last_intent:

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

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    # No files uploaded - enable knowledge base mode
    st.session_state.has_data = False
    
    st.info("No files uploaded - Ask any general questions!")
    st.markdown("""
    **Available Modes:**
    - **Knowledge Base**: Ask any general questions (What is machine learning?, Explain statistics, etc.)
    - **Data Analysis**: Upload CSV files for advanced analysis with visualizations
    - **Business Insights**: Get professional analytical perspectives
    
    **Sample Questions to Try:**
    - "What is the difference between correlation and causation?"
    - "Explain machine learning in simple terms"  
    - "What are the best practices for data visualization?"
    - "How do I interpret statistical significance?"
    """)
    
    # Initialize chat history for knowledge mode
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat History Section (collapsed by default)
    if st.session_state.messages:
        with st.expander(f"📜 Chat History ({len(st.session_state.messages)} messages)", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Clear History", type="secondary", key="clear_history_btn_no_data"):
                    st.session_state.messages = []
                    st.rerun()
            
            # Display chat messages
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    if message.get("type") == "text":
                        st.write(message["content"])
                    else:
                        st.write(message["content"])

    # User query input for knowledge mode
    query = st.chat_input(
        "Ask any question about data science, statistics, business analysis...",
        key="query_input_no_data"
    )

    # Handle knowledge queries when no data is uploaded
    if query:

        if not all([api_key, endpoint]):
            st.error("Azure OpenAI configuration missing. Please check your environment variables.")
        else:
            # Knowledge base mode - answer any general questions
            knowledge_prompt = f'''You are a Principal Data Scientist with 10+ years of experience and a seasoned Business Analyst with extensive real-world expertise. Answer this question clearly and professionally: "{query}"

                            Provide a helpful, accurate explanation using your expertise in:
                            - Data science and machine learning
                            - Statistical analysis and interpretation  
                            - Business intelligence and analytics
                            - Data visualization best practices
                            - Strategic business insights

                            Use simple, professional language and include practical examples when relevant. Structure your response to be informative and actionable.'''

            knowledge_client = AzureChatOpenAI(
                    openai_api_key=api_key,
                    azure_endpoint=endpoint,
                    openai_api_version="2024-10-21",
                    max_tokens=1500,  # Increased from 800 to handle larger datasets
                    temperature=0
                        )
            
            with st.spinner("Analyzing your question..."):
                response = knowledge_client.invoke(knowledge_prompt)
                knowledge_text = response.content
                
            st.subheader("Expert Answer:")
            st.write(knowledge_text)

            # Save to chat history
            st.session_state.messages.append({"role": "user", "content": query, "type": "text"})
            st.session_state.messages.append({"role": "assistant", "content": knowledge_text, "type": "text"})
