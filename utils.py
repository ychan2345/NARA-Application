import streamlit as st
import pandas as pd
import base64
from io import StringIO

def display_dataframe(df, max_rows=100, unique_key=""):
    """Display dataframe with proper formatting and pagination."""
    
    if df is None or df.empty:
        st.warning("📋 No data to display")
        return
    
    # Create a copy to avoid modifying the original
    df_display = df.copy()
    
    # Fix mixed-type columns that cause Arrow conversion issues
    for col in df_display.columns:
        if df_display[col].dtype == 'object':
            # Convert all values to strings to avoid mixed type issues
            df_display[col] = df_display[col].astype(str)
    
    # Show basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df_display))
    with col2:
        st.metric("Columns", len(df_display.columns))
    with col3:
        memory_usage = df_display.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")
    
    # Display data types
    with st.expander("📊 Column Information"):
        col_info = pd.DataFrame({
            'Column': df_display.columns,
            'Data Type': [str(dtype) for dtype in df_display.dtypes],
            'Non-Null Count': df_display.count(),
            'Null Count': df_display.isnull().sum(),
            'Unique Values': [df_display[col].nunique() for col in df_display.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    # Display the actual dataframe
    if len(df_display) > max_rows:
        st.warning(f"⚠️ Dataset has {len(df_display)} rows. Showing first {max_rows} rows.")
        st.dataframe(df_display.head(max_rows), use_container_width=True)
        
        # Option to show more
        if st.button("📄 Show All Rows", key=f"show_all_{unique_key}"):
            st.dataframe(df_display, use_container_width=True)
    else:
        st.dataframe(df_display, use_container_width=True)
    
    # Basic statistics for numeric columns (use original df for accurate numeric detection)
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        with st.expander("📈 Numeric Column Statistics"):
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

def create_download_link(df, filename="data.csv", link_text="Download CSV"):
    """Create a download link for dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def format_code_for_display(code):
    """Format code for better display in Streamlit."""
    if not code:
        return "No code generated"
    
    # Remove any markdown formatting
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    
    return code.strip()

def validate_dataframe(df):
    """Validate that the dataframe is safe to use."""
    if df is None:
        return False, "Dataframe is None"
    
    if not isinstance(df, pd.DataFrame):
        return False, "Object is not a pandas DataFrame"
    
    if df.empty:
        return False, "Dataframe is empty"
    
    # Check for reasonable size (prevent memory issues)
    if len(df) > 1000000:  # 1 million rows
        return False, "Dataframe is too large (>1M rows)"
    
    if len(df.columns) > 1000:  # 1000 columns
        return False, "Dataframe has too many columns (>1000)"
    
    return True, "Dataframe is valid"

def get_sample_queries():
    """Return sample queries for user guidance."""
    
    manipulation_queries = [
        "Remove all rows where the age column is less than 18",
        "Create a new column called 'age_group' that categorizes ages into Young (18-30), Middle (31-50), and Senior (51+)",
        "Filter the dataset to only include records from the last 6 months",
        "Remove duplicate rows based on email address",
        "Fill missing values in the salary column with the median salary",
        "Convert the date column to datetime format",
        "Group by department and calculate the average salary",
        "Sort the dataset by date in descending order",
        "Remove outliers in the price column using the IQR method",
        "Create a new column that combines first_name and last_name"
    ]
    
    analysis_queries = [
        "Show the correlation between all numeric columns",
        "Create a histogram showing the distribution of ages",
        "Perform a statistical summary of sales by month",
        "Find outliers in the price column and visualize them",
        "Create a scatter plot of salary vs experience",
        "Show the top 10 most frequent values in the category column",
        "Perform a t-test to compare salaries between two departments",
        "Create a box plot showing salary distribution by department",
        "Calculate and visualize the trend in sales over time",
        "Perform clustering analysis on customer data"
    ]
    
    return {
        'manipulation': manipulation_queries,
        'analysis': analysis_queries
    }

def show_help_section():
    """Display help information for users."""
    
    with st.expander("❓ Help & Examples"):
        st.markdown("""
        ### 🔧 Data Manipulation Examples:
        - **Filtering**: "Remove rows where age is less than 25"
        - **New Columns**: "Create a new column for age groups: Young (18-30), Adult (31-60), Senior (60+)"
        - **Data Cleaning**: "Fill missing values in the salary column with the median"
        - **Grouping**: "Group by department and calculate average salary"
        - **Sorting**: "Sort the data by date in descending order"
        
        ### 📊 Data Analysis Examples:
        - **Statistics**: "Show correlation matrix between all numeric columns"
        - **Visualization**: "Create a histogram of age distribution"
        - **Comparison**: "Compare average salaries between departments using a box plot"
        - **Trends**: "Show sales trend over time with a line chart"
        - **Outliers**: "Find and visualize outliers in the price column"
        
        ### 💡 Tips:
        - Be specific about column names in your requests
        - Mention the type of visualization you want for analysis
        - Use clear, descriptive language
        - Start with simple operations and build complexity
        """)
