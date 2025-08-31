import streamlit as st
import pandas as pd
import io
import traceback
from data_agent import DataAgent
from code_executor import CodeExecutor
from utils import display_dataframe, create_download_link

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

# Initialize agents
data_agent = DataAgent()
code_executor = CodeExecutor()

st.title("🤖 Agentic AI Data Manipulation & Analysis")
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
    
    # Global controls - always available
    st.subheader("🔧 Global Actions")
    
    if st.button("🔄 Reset Everything", type="secondary"):
        # Complete reset of all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Everything has been reset!")
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
            st.session_state.current_phase = 'analysis'
            st.success("Dataset approved! Moving to analysis phase.")
            st.rerun()
        
        if st.button("🔄 Reset to Original"):
            if st.session_state.original_df is not None:
                st.session_state.current_df = st.session_state.original_df.copy()
                st.session_state.manipulation_history = []
                st.success("Reset to original dataset!")
                st.rerun()
    
    elif st.session_state.current_phase == 'analysis':
        st.subheader("🎯 Analysis Actions")
        if st.button("🔧 Back to Manipulation"):
            st.session_state.current_phase = 'manipulation'
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
            st.success(f"✅ Started working with {selected_file}")
            st.rerun()
        
        # Preview datasets
        st.subheader("📊 Dataset Previews")
        for filename, df in st.session_state.uploaded_files.items():
            with st.expander(f"Preview: {filename} ({len(df)} rows, {len(df.columns)} columns)"):
                st.dataframe(df.head(5), use_container_width=True)

elif st.session_state.current_phase == 'manipulation':
    st.header("🔧 Data Manipulation Phase")
    
    # Show currently selected file
    if st.session_state.selected_file:
        st.info(f"📂 Working with: **{st.session_state.selected_file}**")
    
    # Display current dataset info
    if st.session_state.current_df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(st.session_state.current_df))
        with col2:
            st.metric("Columns", len(st.session_state.current_df.columns))
        with col3:
            st.metric("Operations", len(st.session_state.manipulation_history))
    
    # Data type modification section
    st.subheader("🔧 Column Data Types")
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
- Use 'df' as the dataframe variable
- Handle potential errors and edge cases
- Clean/preprocess the data if needed before conversion
- Use appropriate pandas methods for the conversion
- Only return the Python code, no explanations

Example approaches:
- For datetime: handle different date formats, clean strings first
- For numeric: remove non-numeric characters, handle special values
- For boolean: map text values to True/False
- For category: handle missing values appropriately"""

                                generated_code = data_agent.generate_manipulation_code(
                                    df_temp, 
                                    conversion_prompt,
                                    chat_history=[],
                                    unique_categories={}
                                )
                                
                                if generated_code:
                                    # Execute the AI-generated conversion code
                                    result_df = code_executor.execute_manipulation(df_temp, generated_code)
                                    
                                    if result_df is not None:
                                        df_temp = result_df
                                        success = True
                                        st.success("✅ AI-powered conversion successful!")
                                    else:
                                        st.error("❌ AI-generated code failed to execute properly")
                                else:
                                    st.error("❌ Could not generate conversion code")
                                    
                        except Exception as ai_error:
                            st.error(f"❌ AI conversion also failed: {str(ai_error)}")
                    
                    # Apply changes if successful
                    if success:
                        st.session_state.current_df = df_temp
                        
                        # Also update approved_df if it exists (user is in analysis phase)
                        if st.session_state.approved_df is not None:
                            st.session_state.approved_df = df_temp.copy()
                            st.info("🔄 Approved dataset has been updated with the new data type change")
                        
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
                    
                    if generated_code:
                        st.subheader("🔍 Generated Code")
                        st.code(generated_code, language="python")
                        
                        # Execute the code
                        with st.container():
                            st.write(f"🔍 **Debug**: Executing manipulation on dataset with {len(st.session_state.current_df)} rows")
                            
                            # Show available columns for debugging
                            if 'Region' in st.session_state.current_df.columns:
                                unique_regions = st.session_state.current_df['Region'].unique()
                                region_counts = st.session_state.current_df['Region'].value_counts()
                                st.write(f"🔍 **All unique regions**: {list(unique_regions)}")
                                st.write(f"🔍 **Region counts**: {dict(region_counts.head(10))}")
                            else:
                                # Check for similar column names
                                region_like_cols = [col for col in st.session_state.current_df.columns if 'region' in col.lower()]
                                st.write(f"🔍 **Available columns**: {list(st.session_state.current_df.columns)}")
                                if region_like_cols:
                                    st.write(f"🔍 **Region-like columns found**: {region_like_cols}")
                                    for col in region_like_cols[:1]:  # Show values from first region-like column
                                        sample_values = st.session_state.current_df[col].unique()[:10]
                                        st.write(f"🔍 **Values in {col}**: {list(sample_values)}")
                            
                            result_df = code_executor.execute_manipulation(
                                st.session_state.current_df, 
                                generated_code
                            )
                        
                        if result_df is not None:
                            # Show before/after counts  
                            original_rows = len(st.session_state.current_df)
                            new_rows = len(result_df)
                            
                            # Check if any actual change occurred
                            if new_rows == original_rows:
                                st.warning(f"⚠️ **No rows were filtered!** The filter criteria didn't match any data. Check the exact region names above.")
                                st.info("💡 **Tip**: Copy the exact region names from the list above and use them in your filter.")
                                
                                # Try to suggest correct region names
                                if 'Region' in st.session_state.current_df.columns:
                                    unique_regions = st.session_state.current_df['Region'].unique()
                                    # Look for regions that might match what user wanted
                                    north_like = [r for r in unique_regions if 'north' in str(r).lower()]
                                    if north_like:
                                        st.info(f"🔍 **Found similar regions**: {north_like}")
                                        suggested_filter = f"df = df[df['Region'].isin({north_like})]"
                                        st.code(suggested_filter, language="python")
                                        st.write("Try copying this code and asking: 'apply this filter code'")
                            
                            # Update current dataframe only if there was a meaningful change
                            if new_rows != original_rows or not result_df.equals(st.session_state.current_df):
                                st.session_state.current_df = result_df
                                
                                # Also update approved_df if it exists (user is in analysis phase)
                                if st.session_state.approved_df is not None:
                                    st.session_state.approved_df = result_df.copy()
                                    st.info("🔄 Approved dataset has been updated with the manipulation")
                            
                            # Add to history
                            st.session_state.manipulation_history.append({
                                'query': query,
                                'code': generated_code,
                                'timestamp': pd.Timestamp.now()
                            })
                            
                            st.success(f"✅ Operation completed! Dataset updated: {original_rows} → {new_rows} rows")
                            
                            # Show immediate changes
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original Rows", original_rows)
                            with col2:
                                st.metric("New Rows", new_rows)
                            with col3:
                                st.metric("Change", new_rows - original_rows, 
                                        delta=new_rows - original_rows)
                            
                            st.info("🔄 Dataset has been updated. Scroll up to see the updated dataset in the 'Current Dataset' section.")
                            
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
                    st.write(f"**{chat['type'].title()}** - {chat['timestamp'].strftime('%H:%M:%S')}")
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
        # Ensure we have a valid query string
        final_query = analysis_query or quick_action_query or ""
        if not final_query.strip():
            st.error("No valid query provided")
            st.stop()
            
        with st.spinner("🤖 Analyzing with advanced AI capabilities..."):
            try:
                # Use intelligent routing to determine approach
                intent = data_agent.route_query_intelligently(final_query)
                
                # Store last query and intent for feedback system
                st.session_state.last_query = final_query
                st.session_state.last_intent = intent
                
                if intent == "insight":
                    # Generate enhanced business insights
                    st.info("🧠 Generating strategic business insights...")
                    if st.session_state.approved_df is not None:
                        insights = data_agent.generate_enhanced_insights(
                            st.session_state.approved_df,
                            final_query
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
                            'query': final_query,
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
                            summary = generate_insight_narrative_summary(insights, final_query)
                            if summary:
                                st.subheader("🎯 Key Takeaways")
                                st.write(summary)
                        except Exception:
                            pass
                        
                        # Add feedback section after insight response
                        st.markdown("---")
                        st.markdown("**Was this routing correct?**")
                        cols_fb = st.columns([1, 1, 8])
                        with cols_fb[0]:
                            if st.button("👍", key="thumbs_up_insight", help="Insight generation was correct"):
                                try:
                                    data_agent.save_feedback_decision(final_query, "insight", True, None)
                                    st.toast("Feedback saved ✅")
                                except Exception as e:
                                    st.error(f"Error saving feedback: {e}")
                        with cols_fb[1]:
                            if st.button("👎", key="thumbs_down_insight", help="Should have generated code instead"):
                                try:
                                    data_agent.save_feedback_decision(final_query, "insight", False, None)
                                    st.toast("Feedback saved ✅")
                                except Exception as e:
                                    st.error(f"Error saving feedback: {e}")
                    else:
                        st.error("❌ Failed to generate insights. Please try rephrasing your query.")
                        
                else:
                    # Generate and execute analysis code with self-repair
                    st.info("⚙️ Generating and executing analysis code...")
                    generated_code = data_agent.generate_analysis_code(
                        st.session_state.approved_df, 
                        final_query,
                        chat_history=st.session_state.chat_history,
                        unique_categories=st.session_state.unique_categories
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'type': 'analysis', 
                        'query': final_query,
                        'timestamp': pd.Timestamp.now()
                    })
                    
                    if generated_code:
                        # Execute with enhanced capabilities and self-repair
                        analysis_result = code_executor.execute_analysis(
                            st.session_state.approved_df, 
                            generated_code,
                            final_query  # Pass query for self-repair
                        )
                        
                        if analysis_result:
                            # Add to history
                            st.session_state.analysis_history.append({
                                'query': final_query,
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
                            
                            # Add feedback section after code response
                            st.markdown("---")
                            st.markdown("**Was this routing correct?**")
                            cols_fb = st.columns([1, 1, 8])
                            with cols_fb[0]:
                                if st.button("👍", key="thumbs_up_code", help="Code generation was correct"):
                                    try:
                                        data_agent.save_feedback_decision(analysis_query, "code", True, None)
                                        st.toast("Feedback saved ✅")
                                    except Exception as e:
                                        st.error(f"Error saving feedback: {e}")
                            with cols_fb[1]:
                                if st.button("👎", key="thumbs_down_code", help="Should have generated insights instead"):
                                    try:
                                        data_agent.save_feedback_decision(analysis_query, "code", False, None)
                                        st.toast("Feedback saved ✅")
                                    except Exception as e:
                                        st.error(f"Error saving feedback: {e}")
                            
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

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🤖 Powered by OpenAI GPT-4o | Built with Streamlit
    </div>
    """, 
    unsafe_allow_html=True
)
