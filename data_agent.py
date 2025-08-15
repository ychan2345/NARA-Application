import os
import json
from openai import OpenAI, AzureOpenAI
import pandas as pd
from advanced_analysis import (
    compute_advanced_metrics, 
    self_check_and_rewrite_insight,
    generate_insight_narrative_summary
)

class DataAgent:
    def __init__(self):
        # Initialize OpenAI client based on available configuration
        self.client, self.model = self._initialize_openai_client()
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client based on available environment variables"""
        # Check for Azure OpenAI configuration first
        if all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_API_VERSION")
        ]):
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            print(f"Using Azure OpenAI - Endpoint: {endpoint}, Deployment: {deployment}, API Version: {api_version}")
            
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version,
                azure_endpoint=endpoint
            )
            return client, deployment
        
        # Fallback to regular OpenAI
        elif os.getenv("OPENAI_API_KEY"):
            print("Using regular OpenAI configuration")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            model = "gpt-4o"
            return client, model
        
        else:
            raise ValueError("No valid OpenAI configuration found. Please set either AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_VERSION or OPENAI_API_KEY")
    
    def save_feedback_decision(self, query, predicted_intent, is_correct, comment=None):
        """Save user feedback on routing decisions to improve future routing accuracy."""
        feedback_file = "routing_feedback.json"
        
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
        
        system_prompt = """You are an expert Python data analyst. Generate safe, efficient pandas code for data manipulation tasks.

IMPORTANT RULES:
1. Only use pandas, numpy, and basic Python operations
2. The dataframe variable is always called 'df'
3. Do NOT include 'return' statements - just modify 'df' in place
4. The modified 'df' will be automatically used as the result
5. Do not use any file I/O operations
6. Do not use any dangerous operations like eval(), exec(), or __import__()
7. Focus on common data manipulation: filtering, grouping, sorting, creating columns, etc.
8. Include comments explaining the operations
9. Handle potential errors gracefully
10. Preserve data types when possible

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            code = response.choices[0].message.content
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
6. Store any result dataframes in a variable called 'result_df'
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

{context_info}

{categories_info}

Analysis Request: {query}

Generate Python code to perform this data analysis:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            code = response.choices[0].message.content
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
        
        # Data types
        dtypes_info = []
        for col, dtype in df.dtypes.items():
            dtypes_info.append(f"{col}: {dtype}")
        info_parts.append(f"Data Types:\n" + "\n".join(dtypes_info))
        
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
    
    def load_learned_examples(self):
        """Load learned examples from feedback JSON file for intelligent routing"""
        try:
            with open('routing_feedback.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
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
                    "show the distribution of amount of sale"
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
    
    def route_query_intelligently(self, query: str) -> str:
        """
        Intelligently route query to either 'code' or 'insight' based on learned examples
        """
        try:
            examples = self.load_learned_examples()
            
            # Build examples text from learned examples
            examples_text = "Examples to help you decide:"
            
            # Code examples
            for example in examples.get("code", []):
                examples_text += f'\n- "{example}" → code'
            
            # Insight examples  
            for example in examples.get("insight", []):
                examples_text += f'\n- "{example}" → insight'
            
            intent_prompt = f'''You are a Principal Data Scientist with 10+ years of experience. The user asked: "{query}"

As an expert analyst, determine the most appropriate approach and respond with exactly one word:
- "insight" - if asking for data summaries, business insights, executive summaries, high-level analysis, data overviews, key statistics summaries, or strategic interpretation
- "code" - if asking for specific visualizations, charts, graphs, heatmaps, statistical calculations, or data manipulations that require code execution

IMPORTANT EXAMPLES:
- "Generate a comprehensive data summary" → insight
- "Data summary with key statistics" → insight 
- "Show correlations with heatmap" → code
- "Create visualization" → code

{examples_text}

Only respond with one word: insight or code'''

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            intent = content.strip().lower() if content else 'code'
            return intent if intent in ['code', 'insight'] else 'code'
            
        except Exception:
            # Default to code for safety
            return 'code'
    
    def generate_enhanced_insights(self, df: pd.DataFrame, query: str) -> str:
        """
        Generate enhanced business insights with computed metrics and self-checking
        """
        try:
            # Compute advanced metrics
            computed_metrics = compute_advanced_metrics(df)
            
            # Get basic column information
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Build enhanced data summary with computed metrics
            data_summary = f"""Dataset Overview:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Numeric columns: {numeric_cols[:5]}
- Categorical columns: {categorical_cols[:5]}
- Missing values: {df.isnull().sum().sum()} total across all columns"""

            # Add computed insights to data summary
            if computed_metrics:
                data_summary += "\n\nComputed Insights:"

                # Add correlation insights
                if 'correlations' in computed_metrics and computed_metrics['correlations']:
                    data_summary += "\n- Key Correlations:"
                    for col1, col2, corr_val in computed_metrics['correlations'][:3]:
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

            # Add statistical summary for numeric columns
            if numeric_cols:
                stats_summary = df[numeric_cols[:3]].describe().round(2)
                data_summary += f"\n\nKey Statistics:\n{stats_summary.to_string()}"

            insights_prompt = f'''You are a Principal Data Scientist with 10+ years of experience and a seasoned Business Analyst with extensive real-world business expertise. Your role is to provide professional, strategic insights that bridge technical analysis with actionable business recommendations.

The user asked: "{query}"

Drawing from your extensive experience, provide a comprehensive analysis that demonstrates deep understanding of both technical analysis and business strategy.

{data_summary}

Sample data:
{df.head(3).to_string()}

IMPORTANT: I have already performed key calculations under the hood. Use the computed insights above to provide data-driven evidence with specific numbers, correlations, and statistical findings. Reference these exact calculated values in your analysis.

Professional Analysis Framework:
1. Executive Summary: Start with key findings that directly answer: "{query}" - use the computed metrics above
2. Data-Driven Evidence: Use the specific calculated values, correlations, and statistical trends from the computed insights section
3. Business Context: Interpret the calculated findings through the lens of real-world business implications
4. Strategic Insights: Provide actionable recommendations based on the numerical evidence from calculations
5. Risk Assessment: Identify potential limitations in the calculated metrics or areas requiring further investigation

Provide your analysis as a structured, professional report that executives and business stakeholders would find valuable. Ground every insight in the specific calculated values provided above.'''

            # Create Azure OpenAI client with temperature=0 (as specified in attached file)
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
            
            if not all([azure_endpoint, api_key, deployment_name]):
                return "Azure OpenAI configuration is incomplete. Please check your environment variables."
            
            from langchain_openai import AzureChatOpenAI
            
            insights_client = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,  # Pass as string directly
                api_version="2023-05-15",
                azure_deployment=deployment_name,
                temperature=0  # Exact temperature from attached file
            )
            response = insights_client.invoke(insights_prompt)
            insights_text = response.content

            # Self-check & possible rewrite for insights
            try:
                from advanced_analysis import self_check_and_rewrite_insight
                metrics_text_for_check = data_summary  # this already includes computed numbers/correlations
                checked_insights = self_check_and_rewrite_insight(insights_text, metrics_text_for_check)
                if checked_insights != insights_text:
                    insights_text = checked_insights  # replace with corrected version if provided
                    print("🔍 Analysis was automatically reviewed and enhanced for accuracy")
            except Exception as e:
                print(f"Insight self-check skipped: {e}")
            
            return insights_text
                
        except Exception as e:
            return f"Error generating insights: {str(e)}"
