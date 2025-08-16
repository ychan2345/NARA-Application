"""
Advanced Analysis Module
Integrates enhanced AI capabilities for data analysis with self-repair and vision analysis
"""

import json
import os
import sys
import re
import base64
import io
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime


def get_openai_client():
    """Get LangChain ChatOpenAI client using available configuration (Azure or regular OpenAI)"""
    # Check for Azure OpenAI configuration first
    if all([
        os.environ.get("AZURE_OPENAI_API_KEY"),
        os.environ.get("AZURE_OPENAI_ENDPOINT"),
        os.environ.get("AZURE_OPENAI_API_VERSION")
    ]):
        return AzureChatOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            temperature=0.1
        )
    
    # Fallback to regular OpenAI
    elif os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=0.1
        )
    
    else:
        raise ValueError("No valid OpenAI configuration found")


def get_model_name():
    """Get the appropriate model name based on configuration"""
    if all([
        os.environ.get("AZURE_OPENAI_API_KEY"),
        os.environ.get("AZURE_OPENAI_ENDPOINT"),
        os.environ.get("AZURE_OPENAI_API_VERSION")
    ]):
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    else:
        return "gpt-4o"


def self_check_and_rewrite_insight(insights_text: str, metrics_text: str) -> str:
    """
    Self-check and rewrite insights for accuracy using Azure OpenAI
    """
    try:
        client = get_openai_client()
        model_name = get_model_name()
        
        check_prompt = f"""You are a senior data scientist reviewing an analysis for accuracy. 

Original insights:
{insights_text}

Available data metrics and calculations:
{metrics_text}

Please review the insights and check if:
1. All numerical claims are supported by the provided metrics
2. All correlations mentioned are backed by the calculated values
3. All statistics are accurate based on the computed insights
4. Business interpretations are reasonable given the data

If the insights are accurate, return them unchanged. If there are inaccuracies, provide a corrected version that uses only the verified metrics and calculations."""

        messages = [HumanMessage(content=check_prompt)]
        response = client.invoke(messages)
        
        return response.content or insights_text
        
    except Exception as e:
        print(f"Self-check error: {e}")
        return insights_text


def safe_key_format(key):
    """Format dictionary keys for display"""
    return str(key).replace('_', ' ').title()


def run_code_with_self_repair(code: str, exec_env: dict, query: str, max_tries: int = 3):
    """
    Execute code with self-repair capability using GPT-4o.
    Returns (observation_dict, final_code or None on failure).
    """
    current_code = code
    repair_history = []

    for attempt in range(max_tries):
        # Clear previous results  
        exec_env.pop("fig", None)
        exec_env.pop("result", None)
        exec_env.pop("results", None)
        obs = {}
        
        # Remove .show() calls to prevent display issues
        cleaned = re.sub(r'\w*\.show\(\)', '', current_code)

        # Execute current code with stdout capture
        try:
            from contextlib import redirect_stdout, redirect_stderr
            import io
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            original_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(100)
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(cleaned, exec_env)
                
            # Capture the printed output
            stdout_output = stdout_capture.getvalue().strip()
            if stdout_output:
                obs["stdout"] = stdout_output
                
        except Exception as e:
            obs["errors"] = f"{type(e).__name__}: {e}"
        finally:
            sys.setrecursionlimit(original_limit)

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
            return obs, current_code if current_code else ""

        # Record this attempt for history
        repair_history.append({
            "attempt": attempt + 1,
            "error": obs["errors"],
            "code": current_code[:200] + "..." if len(current_code) > 200 else current_code
        })

        # Generate fix using GPT-4o
        fix_prompt = f"""You wrote Python code for this data analysis task and it errored on attempt {attempt + 1}/{max_tries}. Fix it and return ONLY the corrected code.

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
- Use exact column names from the dataset.
"""

        try:
            client = get_openai_client()
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages = [HumanMessage(content=fix_prompt)]
            response = client.invoke(messages)
            
            fixer = response.content
            if fixer and "```" in fixer:
                code_blocks = fixer.split("```")
                if len(code_blocks) >= 3:
                    fixer = code_blocks[-2]
                if fixer and fixer.startswith("python"):
                    fixer = fixer[len("python"):].strip()
            current_code = fixer.strip() if fixer else current_code
            
        except Exception as repair_error:
            obs["repair_generation_error"] = str(repair_error)
            return obs, current_code if current_code else ""

    return obs, current_code if current_code else ""


def compute_advanced_metrics(df: pd.DataFrame) -> dict:
    """
    Compute advanced metrics for enhanced insights.
    Returns dictionary with computed statistics and correlations.
    """
    computed_metrics = {}
    
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            # Basic statistics
            stats_summary = df[numeric_cols[:3]].describe().round(2)
            computed_metrics['basic_stats'] = stats_summary.to_dict()

            # Correlations if multiple numeric columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().round(3)
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.3:  # Only meaningful correlations
                            corr_pairs.append((col1, col2, corr_val))
                computed_metrics['correlations'] = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]

            # Distribution insights
            for col in numeric_cols[:3]:  # Limit to first 3
                col_data = df[col].dropna()
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
            for col in categorical_cols[:2]:  # Limit to first 2
                value_counts = df[col].value_counts()
                computed_metrics[f'{col}_distribution'] = {
                    'unique_count': len(value_counts),
                    'top_categories': value_counts.head(3).to_dict(),
                    'missing_count': df[col].isnull().sum()
                }

    except Exception as e:
        st.warning(f"Some advanced calculations could not be completed: {e}")
    
    return computed_metrics


def self_check_and_rewrite_insight(insights_text: str, metrics_text: str) -> str:
    """
    Ask GPT-4o to check alignment with computed metrics and rewrite if needed.
    Returns the final (possibly rewritten) insight text.
    """
    try:
        check_prompt = f"""Review this analysis against the provided metrics.
- Flag unsupported claims, contradictions, or numbers not grounded in metrics.
- If issues exist, REWRITE a corrected version below your critique.
- Keep it concise, executive-level, and grounded in the metrics only.

Analysis:
{insights_text}

Metrics:
{metrics_text}
"""
        
        client = get_openai_client()
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
        messages = [HumanMessage(content=check_prompt)]
        response = client.invoke(messages)
        
        resp = response.content
        
        # Heuristic: if it contains "rewrite" or looks revised, use the response
        if resp and ("rewrite" in resp.lower() or "revised" in resp.lower()):
            return resp  # show critique + rewrite
        return insights_text
        
    except Exception as e:
        st.warning(f"Insight self-check failed: {e}")
        return insights_text


def analyze_chart_with_vision(fig, query: str) -> str:
    """
    Use GPT-4o vision to analyze a chart and provide insights.
    """
    try:
        # Try to convert figure to image
        img_bytes = fig.to_image(format="png", width=800, height=600)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        vision_prompt = f"""As a Principal Data Scientist, analyze this chart created for the query: "{query}"

Please provide:
• 2-3 key patterns or trends you observe in the visualization
• Business implications of these findings  
• Actionable recommendations based on what the chart reveals
• Any notable outliers, correlations, or insights

Focus on what a business executive would find valuable and actionable. Use simple, professional language."""

        client = get_openai_client()
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
        # For vision with LangChain, we need to use HumanMessage with image content
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    }
                ]
            )
        ]
        response = client.invoke(messages)

        return response.content

    except Exception as e:
        # Fallback to chart structure analysis
        return analyze_chart_structure(fig, query)


def analyze_chart_structure(fig, query: str) -> str:
    """
    Analyze chart structure and data without requiring image conversion.
    """
    try:
        chart_info = []

        # Get chart type
        if hasattr(fig, 'data') and fig.data:
            trace_types = [trace.type if hasattr(trace, 'type') else 'scatter' for trace in fig.data]
            chart_info.append(f"Chart type: {', '.join(set(trace_types))}")

            # Get data dimensions
            for i, trace in enumerate(fig.data):
                if hasattr(trace, 'x') and hasattr(trace, 'y'):
                    x_len = len(trace.x) if trace.x is not None else 0
                    y_len = len(trace.y) if trace.y is not None else 0
                    chart_info.append(f"Data series {i+1}: {max(x_len, y_len)} data points")

        # Get axis labels
        if hasattr(fig, 'layout'):
            if hasattr(fig.layout, 'xaxis') and hasattr(fig.layout.xaxis, 'title'):
                if fig.layout.xaxis.title and hasattr(fig.layout.xaxis.title, 'text'):
                    chart_info.append(f"X-axis: {fig.layout.xaxis.title.text}")
            if hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title'):
                if fig.layout.yaxis.title and hasattr(fig.layout.yaxis.title, 'text'):
                    chart_info.append(f"Y-axis: {fig.layout.yaxis.title.text}")

        if chart_info:
            chart_description = "; ".join(chart_info)

            structure_prompt = f'''Based on this chart structure created for the query "{query}":

Chart Details: {chart_description}

As a Principal Data Scientist, provide 2-3 insights about:
• What this visualization type reveals about the data patterns
• Business implications of the chart structure and data dimensions
• Actionable recommendations based on this analysis approach

Focus on executive-level insights using simple, professional language.'''

            client = get_openai_client()
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages = [HumanMessage(content=structure_prompt)]
            response = client.invoke(messages)

            return response.content

    except Exception as e:
        return f"Chart analysis unavailable: {str(e)}"

    return "Chart analysis unavailable"


def generate_insight_narrative_summary(insights_text: str, query: str) -> str:
    """
    Generate concise summary from detailed insights.
    """
    try:
        summary_prompt = f'''From this detailed analysis for the query "{query}":

{insights_text}

Extract 2-4 key actionable takeaways as bullet points. Focus on:
- The most important business implications
- Clear, simple language
- Actionable recommendations
- Avoid repeating the full analysis

Format as bullet points starting with •'''

        client = get_openai_client()
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
        messages = [HumanMessage(content=summary_prompt)]
        response = client.invoke(messages)

        return response.content

    except Exception as e:
        st.warning(f"Could not generate narrative summary: {e}")
        return None


def generate_narrative_from_results(exec_env: dict, query: str) -> str:
    """
    Generate narrative summary from execution results, including vision analysis of charts.
    """
    try:
        summary_points = []
        chart_analysis = None

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

        # Check for figure/chart and analyze with vision if available
        if 'fig' in exec_env:
            summary_points.append("Visualization created")
            
            # Try to analyze the chart with vision
            chart_analysis = analyze_chart_with_vision(exec_env['fig'], query)
            if chart_analysis:
                return chart_analysis

        # If we have meaningful numerical results but no chart analysis, generate text-based narrative
        if summary_points and not chart_analysis:
            results_text = "; ".join(summary_points)

            narrative_prompt = f'''Based on these analysis results for the query "{query}":

Results: {results_text}

As a Principal Data Scientist, provide 2-4 concise bullet-point insights or a 2-3 sentence executive summary that:
- Highlights the key business implications
- Uses simple, professional language  
- Focuses on actionable insights
- Avoids technical jargon

Format as bullet points starting with • or as a brief executive summary.'''

            client = get_openai_client()
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages = [HumanMessage(content=narrative_prompt)]
            response = client.invoke(messages)

            return response.content

    except Exception as e:
        st.warning(f"Could not generate narrative: {e}")
        return None

    return None
