import streamlit as st
import os
import json
from openai import OpenAI, AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import pandas as pd
from advanced_analysis import (
    compute_advanced_metrics, 
    self_check_and_rewrite_insight,
    generate_insight_narrative_summary
)
from dotenv import load_dotenv
import altair as alt
import dataiku
from datetime import datetime

load_dotenv()

# Get credentials from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class DataAgent:
    def __init__(self):
        # Initialize OpenAI client based on available configuration
        self.client = self._initialize_openai_client()
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client based on available environment variables"""
            
        client = AzureChatOpenAI(
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version="2024-10-21",
                #max_tokens=1500,  # Increased from 800 to handle larger datasets
                temperature=0
                )

        return client
    
    def save_feedback_decision(self, query, predicted_intent, is_correct, comment=None):
        """Save user feedback on routing decisions to improve future routing accuracy."""
        feedback_file = "intention_feedback.json"
        
        # Load existing feedback data
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            # Initialize with the base examples
            feedback_data = {
                "code": [
                    "show distribution of age",
                    "create a histogram", 
                    "find correlations",
                    "show correlation matrix",
                    "visualize key trends",
                    "perform t-test",
                    "detect outliers",
                    "calculate market share",
                    "what is the share of X",
                    "compute percentage",
                    "calculate ratio",
                    "show the distribution of amount of sale",
                    "In territory 1, what category (contacts segment) has the highest overall dollar growth in 2024 vs 2023",
                    "which Territory is losing the most overall Johnson and Johnson share to the market in 2024 vs 2023",
                    "In region Northeast, compute the market share by territory."
                ],
                "insight": [
                    "what are the key insights",
                    "summarize the data", 
                    "comprehensive summary of dataset",
                    "explain this data pattern",
                    "what does this tell us about business performance",
                    "provide a comprehensive summary of this dataset"
                ]
            }
        
        # Add the query to the appropriate category based on feedback
        if is_correct:
            # User confirmed the routing was correct
            if predicted_intent == "code" and query not in feedback_data["code"]:
                feedback_data["code"].append(query)
            elif predicted_intent == "insight" and query not in feedback_data["insight"]:
                feedback_data["insight"].append(query)
        else:
            # User said routing was wrong, add to opposite category
            correct_intent = "insight" if predicted_intent == "code" else "code"
            if correct_intent == "code" and query not in feedback_data["code"]:
                feedback_data["code"].append(query)
            elif correct_intent == "insight" and query not in feedback_data["insight"]:
                feedback_data["insight"].append(query)
        
        # Save updated feedback data
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return feedback_data
    
    def generate_manipulation_code(self, df, query, chat_history=None, unique_categories=None):
        """Generate Python code for data manipulation based on natural language query."""
        
        # Get dataframe info for context
        df_info = self._get_dataframe_info(df)
        
        # Prepare chat history context
        context_info = ""
        if chat_history and len(chat_history) > 0:
            recent_queries = [item['query'] for item in chat_history[-5:] if 'query' in item]
            if recent_queries:
                context_info = f"Recent conversation context:\n" + "\n".join([f"- {q}" for q in recent_queries])
        
        # Prepare unique categories context  
        categories_info = ""
        if unique_categories:
            categories_info = "Unique categorical values in dataset:\n"
            for col, values in unique_categories.items():
                categories_info += f"- {col}: {', '.join(str(v) for v in values[:10])}{'...' if len(values) > 10 else ''}\n"
            categories_info += "\nNote: When users mention abbreviated names (like 'jnj'), match them to the full names in the data (like 'Johnson and Johnson')."
        
        system_prompt = """You are an expert Python data scientist. Generate safe, efficient pandas code for data manipulation tasks.

IMPORTANT RULES:
1. Only use pandas, numpy, and basic Python operations
2. The dataframe variable is always called 'df'
3. Always assign your final result back to the variable 'df'. For example: df = df[...] or df['new_col'] = ...
4. Do NOT create new DataFrame variables like 'filtered_df', 'df_new', or 'output_df'
5. Do NOT include 'return' statements - just modify 'df' in place
6. The modified 'df' will be automatically used as the result
7. Do not use any file I/O operations
8. Do not use any dangerous operations like eval(), exec(), or __import__()
9. Focus on common data manipulation: filtering, grouping, sorting, creating columns, etc.
10. Include comments explaining the operations
11. Handle potential errors gracefully
12. Preserve data types when possible

CRITICAL FILTERING RULE FOR CONVERTED NUMERIC COLUMNS:
If a column is marked as "object (NUMERIC VALUES: [1, 2, 3...])" in the data info, it contains numeric values stored as objects.
- ALWAYS use numeric values (not strings) for filtering: df[df['col'] == 1] NOT df[df['col'] == '1']
- For multiple values: df[df['col'].isin([1, 2, 3])] NOT df[df['col'].isin(['1', '2', '3'])]
- These columns were converted from int/float to object but still contain numeric data

Respond with only the Python code, no explanations or markdown formatting."""

        user_prompt = f"""
Dataset Information:
{df_info}

{context_info}

{categories_info}

User Request: {query}

Generate pandas code to perform this data manipulation:
"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.client.invoke(messages)

            code = response.content
            if code is None:
                return None
            code = code.strip()
            
            # Clean up the code (remove markdown formatting if present)
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            print(f"Error generating manipulation code: {e}")
            return None
    
    def generate_analysis_code(self, df, query, chat_history=None, unique_categories=None):
        """Generate Python code for data analysis based on natural language query."""
        
        # Get dataframe info for context
        df_info = self._get_dataframe_info(df)
        
        # Prepare chat history context
        context_info = ""
        if chat_history and len(chat_history) > 0:
            recent_queries = [item['query'] for item in chat_history[-5:] if 'query' in item]
            if recent_queries:
                context_info = f"Recent conversation context:\n" + "\n".join([f"- {q}" for q in recent_queries])
        
        # Prepare unique categories context  
        categories_info = ""
        if unique_categories:
            categories_info = "Unique categorical values in dataset:\n"
            for col, values in unique_categories.items():
                categories_info += f"- {col}: {', '.join(str(v) for v in values[:10])}{'...' if len(values) > 10 else ''}\n"
            categories_info += "\nNote: When users mention abbreviated names (like 'jnj'), match them to the full names in the data (like 'Johnson and Johnson')."
        
        system_prompt = """You are an expert data scientist. Generate Python code for data analysis and visualization.

IMPORTANT RULES:
1. Use pandas, numpy, plotly (px, go, ff), and scipy for analysis
2. The dataframe variable is always called 'df'
3. PREFER Plotly for visualizations - use plotly.express (px) for simple plots, plotly.graph_objects (go) for complex ones
4. Include statistical analysis where appropriate
5. Store text insights in a variable called 'analysis_summary'
6. Store any result dataframes in a variable called 'result' (IMPORTANT: use 'result', not 'result_df')
7. Store plotly figures in a variable called 'fig' OR 'plotly_fig'
8. ALWAYS print the final result/analysis_summary so it appears in output
8. Do not use any file I/O operations
9. Handle missing data appropriately
10. Include comments explaining the analysis steps
11. Do NOT call .show() on plots - just assign the figure to fig/plotly_fig

CRITICAL RULES:
- For correlation analysis, ONLY use numeric columns: df.select_dtypes(include=['number']).columns
- Never include text/categorical columns in correlation calculations
- For data summaries, use df.describe() and include both numeric and categorical insights
- Always use proper variable names: 'fig' or 'plotly_fig' for plots, 'result' for data, 'analysis_summary' for text

CRITICAL FILTERING RULE FOR CONVERTED NUMERIC COLUMNS:
If a column is marked as "object (NUMERIC VALUES: [1, 2, 3...])" in the data info, it contains numeric values stored as objects.
- ALWAYS use numeric values (not strings) for filtering: df[df['col'] == 1] NOT df[df['col'] == '1']
- For multiple values: df[df['col'].isin([1, 2, 3])] NOT df[df['col'].isin(['1', '2', '3'])]
- These columns were converted from int/float to object but still contain numeric data

Code structure should be:
- Perform analysis
- Create visualizations if needed using Plotly
- Store insights in analysis_summary
- Store result data in result_df (if applicable)
- Store plotly figure in fig or plotly_fig

Example patterns:
- Correlation: 
  numeric_cols = df.select_dtypes(include=['number']).columns
  if len(numeric_cols) > 1:
      corr_matrix = df[numeric_cols].corr()
      fig = px.imshow(corr_matrix, 
                      text_auto=True, 
                      labels=dict(x="Variables", y="Variables", color="Correlation"),
                      title='Correlation Heatmap')
      result = corr_matrix

- Data Summary:
  analysis_summary = f"Dataset has {len(df)} rows and {len(df.columns)} columns"
  result = df.describe()
- Box plot: fig = px.box(df, x='category', y='value')
- Histogram: fig = px.histogram(df, x='column')

Respond with only the Python code, no explanations or markdown formatting."""

        user_prompt = f"""
Dataset Information:
{df_info}

NOTE: This dataset has already been pre-filtered and prepared based on previous user instructions. DO NOT apply the same filtering logic again (e.g., if prior steps filtered rows where country='United States', do NOT repeat this).

{context_info}

{categories_info}

Analysis Request: {query}

Generate Python code to perform this data analysis:
"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.client.invoke(messages)
            code = response.content
            if code is None:
                return None
            code = code.strip()
            
            # Clean up the code (remove markdown formatting if present)
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            print(f"Error generating analysis code: {e}")
            return None
    
    def _get_dataframe_info(self, df):
        """Get comprehensive information about the dataframe for context."""
        
        info_parts = []
        
        # Basic info
        info_parts.append(f"Shape: {df.shape}")
        info_parts.append(f"Columns: {list(df.columns)}")
        
        # Data types with special handling for converted numeric columns
        dtypes_info = []
        converted_numeric_cols = []
        
        for col, dtype in df.dtypes.items():
            # Check if object column contains only numeric values (converted from int/float)
            if dtype == 'object' and not df[col].empty:
                # Sample non-null values to check if they're all numeric
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    try:
                        # Try to convert sample back to numeric to detect converted columns
                        pd.to_numeric(sample_values, errors='raise')
                        # If successful, this was likely converted from numeric
                        converted_numeric_cols.append(col)
                        unique_vals = sorted(df[col].dropna().unique())
                        dtypes_info.append(f"{col}: {dtype} (NUMERIC VALUES: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''})")
                    except (ValueError, TypeError):
                        # Regular object column
                        dtypes_info.append(f"{col}: {dtype}")
            else:
                dtypes_info.append(f"{col}: {dtype}")
        
        info_parts.append(f"Data Types:\n" + "\n".join(dtypes_info))
        
        # Add special warning about converted numeric columns
        if converted_numeric_cols:
            warning_msg = f"""
            IMPORTANT FILTERING NOTE:
            Columns {converted_numeric_cols} contain numeric values stored as objects.
            When filtering these columns, use NUMERIC values (not strings):
            - CORRECT: df[df['{converted_numeric_cols[0]}'] == 1]
            - WRONG: df[df['{converted_numeric_cols[0]}'] == '1']
            - CORRECT: df[df['{converted_numeric_cols[0]}'].isin([1, 2, 3])]
            - WRONG: df[df['{converted_numeric_cols[0]}'].isin(['1', '2', '3'])]"""
        
            info_parts.append(warning_msg)
        
        # Sample data (first few rows)
        info_parts.append(f"Sample Data (first 3 rows):\n{df.head(3).to_string()}")
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info_parts.append(f"Numeric Columns Summary:\n{df[numeric_cols].describe().to_string()}")
        
        # Missing values
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            missing_df = missing_info[missing_info > 0]
            info_parts.append(f"Missing Values:\n{missing_df.to_string()}")
        else:
            info_parts.append("Missing Values: None")
        
        return "\n\n".join(info_parts)
           
    def route_query_intelligently(self, query: str, code_feedback_format, insight_feedback_format) -> str:
        """
        Intelligently route query to either 'code' or 'insight' based on learned examples
        """
        try:
            intent_prompt = f'''You are a Principal Data Scientist with 10+ years of experience. The user asked: "{query}"

As an expert analyst, determine the most appropriate approach and respond with exactly one word:
- "insight" - if asking for data summaries, business insights, executive summaries, high-level analysis, data overviews, key statistics summaries, or strategic interpretation
- "code" - if asking for specific visualizations, charts, graphs, heatmaps, statistical calculations, or data manipulations that require code execution

IMPORTANT EXAMPLES:
- Generate a comprehensive data summary → insight
- Data summary with key statistics → insight 
- Show correlations with heatmap → code
- Create visualization → code

Below are the examples to help you decide:\n
{code_feedback_format} \n
{insight_feedback_format}

Only respond with one word: insight or code'''

            messages = [HumanMessage(content=intent_prompt)]
            response = self.client.invoke(messages)
            content = response.content
            intent = content.strip().lower() if content else 'code'
            return intent if intent in ['code', 'insight'] else 'code'
            #return intent_prompt
                        
        except Exception:
            # Default to code for safety
            return 'code'
                
    def generate_enhanced_insights(self, df: pd.DataFrame, query: str) -> str:
        """
        Generate enhanced business insights using fast pre-computed statistics (no code execution)
        """
        try:
            # Pre-compute statistics rapidly 
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Build comprehensive analysis summary
            analysis_summary = f"""COMPLETE DATASET ANALYSIS:

OVERVIEW:
- Total Records: {len(df):,} rows
- Variables: {len(df.columns)} columns
- Numeric: {len(numeric_cols)} ({', '.join(numeric_cols[:4])}{'...' if len(numeric_cols) > 4 else ''})
- Categorical: {len(categorical_cols)} ({', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''})
- Completeness: {((len(df) * len(df.columns) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100):.1f}% complete"""

            # Add statistics for numeric columns (fast pandas operations)
            if numeric_cols:
                try:
                    # Limit to first 4 numeric columns for performance
                    key_cols = numeric_cols[:4]
                    stats_df = df[key_cols].describe().round(2)
                    analysis_summary += f"\n\nKEY STATISTICS (Full {len(df):,} Records):\n{stats_df.to_string()}"
                    
                    # Add correlations if multiple numeric columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df[key_cols].corr().round(3)
                        # Find strongest correlations
                        strong_corr = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                                corr_val = corr_matrix.iloc[i, j]
                                if abs(corr_val) > 0.4:  # Only meaningful correlations
                                    strong_corr.append((col1, col2, corr_val))
                        
                        if strong_corr:
                            analysis_summary += "\n\nSTRONG CORRELATIONS:"
                            for col1, col2, corr_val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)[:3]:
                                strength = "Very Strong" if abs(corr_val) > 0.8 else "Strong" if abs(corr_val) > 0.6 else "Moderate"
                                direction = "positive" if corr_val > 0 else "negative"
                                analysis_summary += f"\n- {col1} ↔ {col2}: {strength} {direction} relationship ({corr_val:.3f})"
                                
                except Exception:
                    analysis_summary += "\n\nStatistical calculations encountered an issue."
            
            # Add categorical insights
            if categorical_cols:
                analysis_summary += "\n\nCATEGORICAL DATA:"
                for col in categorical_cols[:3]:
                    try:
                        unique_count = df[col].nunique()
                        most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                        pct_most = (df[col].value_counts().iloc[0] / len(df) * 100) if len(df[col].value_counts()) > 0 else 0
                        analysis_summary += f"\n- {col}: {unique_count} categories, '{most_common}' most frequent ({pct_most:.1f}%)"
                    except Exception:
                        continue

            # Generate business insights with single fast API call
            insights_prompt = f'''You are a Principal Data Scientist with 10+ years of experience. Generate strategic business insights based on this complete dataset analysis.

User Query: "{query}"

{analysis_summary}

Based on this COMPLETE analysis of {len(df):,} records, provide professional business insights that directly answer: "{query}"

Use the EXACT statistics, correlations, and percentages shown above. Reference specific calculated values.

Framework:
1. EXECUTIVE SUMMARY: Key findings directly answering the query
2. DATA EVIDENCE: Reference specific numbers, correlations, and calculated values above  
3. BUSINESS IMPLICATIONS: Real-world meaning of these findings
4. RECOMMENDATIONS: Actionable steps based on the data
5. CONSIDERATIONS: Limitations or areas needing investigation

Provide a structured report grounded in the specific calculated values.'''
                       
            client = AzureChatOpenAI(
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                openai_api_version="2024-10-21",
                #max_tokens=1500,  # Increased from 800 to handle larger datasets
                temperature=0.1
                )
            
            response = client.invoke(insights_prompt)
            return response.content if isinstance(response.content, str) else str(response.content)
                
        except Exception as e:
            return f"Error generating insights: {str(e)}"
