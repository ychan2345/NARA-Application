import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy import stats
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import warnings
from advanced_analysis import (
    run_code_with_self_repair, 
    generate_narrative_from_results,
    analyze_chart_with_vision
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CodeExecutor:
    def __init__(self):
        # Set up safe execution environment
        self.safe_builtins = {
            '__builtins__': {
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'object': object,
                'type': type,
                'sum': sum,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'print': print,
                '__import__': __import__
            }
        }
    
    def execute_manipulation(self, df, code):
        """Execute data manipulation code safely and return the result dataframe."""
        
        try:
            # Clean the code to remove any return statements
            code = self._clean_manipulation_code(code)
            
            # Create safe namespace
            namespace = {
                'df': df.copy(),  # Work with a copy to preserve original
                'pd': pd,
                'np': np,
                **self.safe_builtins
            }
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, namespace)
            
            # Get the modified dataframe - check multiple possible variable names
            result_df = None
            
            # First try to get the 'df' variable (original dataframe)
            if 'df' in namespace and isinstance(namespace['df'], pd.DataFrame):
                result_df = namespace['df']
            
            # If df wasn't modified, look for other dataframe variables that were created
            if result_df is None or result_df.equals(df):
                # Look for any DataFrame variables in the namespace
                for var_name, var_value in namespace.items():
                    if (isinstance(var_value, pd.DataFrame) and 
                        var_name not in ['df', 'pd'] and  # Skip original df and pandas module
                        not var_name.startswith('__')):   # Skip built-in variables
                        result_df = var_value
                        break
            
            if result_df is not None and isinstance(result_df, pd.DataFrame):
                return result_df
            else:
                print("Error: Code did not return a valid dataframe")
                return None
                
        except Exception as e:
            print(f"Execution error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def execute_analysis(self, df, code, query=""):
        """Execute analysis code with self-repair capability and enhanced results."""
        
        try:
            # Create safe namespace for analysis
            exec_env = {
                'df': df.copy(),
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'px': px,
                'go': go,
                'ff': ff,
                'make_subplots': make_subplots,
                'stats': stats,
                'analysis_summary': None,
                'result_df': None,
                'plotly_fig': None,
                'fig': None,
                'result': None,
                **self.safe_builtins
            }
            
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            
            # Set up matplotlib for better plots
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['font.size'] = 10
            
            # Use self-repair execution if query is provided
            if query:
                obs, final_code = run_code_with_self_repair(code, exec_env, query)
                
                # If self-repair succeeded, use its results
                if not obs.get("errors"):
                    result = {
                        'text_output': obs.get('stdout', ''),  # Capture stdout from self-repair
                        'dataframe': None,
                        'plot': None,
                        'plotly_fig': obs.get('fig'),
                        'self_repair_used': obs.get('repair_attempts', 0) > 0
                    }
                    
                    # Check for dataframe results
                    if obs.get('result') is not None:
                        if hasattr(obs.get('result'), 'shape'):
                            result['dataframe'] = obs.get('result')
                        elif isinstance(obs.get('result'), dict):
                            # If result is a dict, it might contain analysis results
                            result['text_output'] = str(obs.get('result'))
                    
                    # Skip narrative generation for code responses
                    # Only return raw code execution results
                    
                    return result
                else:
                    # Self-repair failed, fall back to normal execution
                    print(f"Self-repair failed after {obs.get('repair_attempts', 0)} attempts")
            
            # Fallback to normal execution
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_env)
            
            # Collect results more comprehensively
            result = {
                'text_output': '',
                'dataframe': None,
                'plot': None,
                'plotly_fig': None
            }
            
            # Check for text output - try multiple variable names
            text_output = None
            if exec_env.get('analysis_summary'):
                text_output = exec_env.get('analysis_summary')
            elif exec_env.get('summary'):
                text_output = exec_env.get('summary')
            elif exec_env.get('result_text'):
                text_output = exec_env.get('result_text')
            
            if text_output:
                result['text_output'] = str(text_output)
            
            # Check for dataframe results
            if exec_env.get('result_df') is not None and hasattr(exec_env.get('result_df'), 'shape'):
                result['dataframe'] = exec_env.get('result_df')
            elif exec_env.get('result') is not None and hasattr(exec_env.get('result'), 'shape'):
                result['dataframe'] = exec_env.get('result')
            
            # Check for figure results with multiple possible variable names
            fig_candidates = ['plotly_fig', 'fig', 'correlation_heatmap', 'chart', 'visualization']
            for var_name in fig_candidates:
                fig_obj = exec_env.get(var_name)
                if fig_obj is not None:
                    # Validate it's a proper plotly figure
                    if hasattr(fig_obj, 'data') and hasattr(fig_obj, 'layout'):
                        result['plotly_fig'] = fig_obj
                        break
            
            # Fallback to matplotlib
            if result['plotly_fig'] is None and plt.get_fignums():
                result['plot'] = plt.gcf()
            
            # Get any printed output and ensure it's captured
            printed_output = stdout_capture.getvalue().strip()
            
            # Prioritize printed output for text results
            if printed_output:
                result['text_output'] = printed_output
                print(f"Captured and set printed output: {printed_output}")
            elif text_output:
                result['text_output'] = str(text_output)
                print(f"Using variable text_output: {text_output}")
            
            return result
            
        except Exception as e:
            print(f"Analysis execution error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def validate_code_safety(self, code):
        """Basic validation to ensure code doesn't contain dangerous operations."""
        
        dangerous_patterns = [
            'import os',
            'import sys', 
            'import subprocess',
            'open(',
            'file(',
            'eval(',
            'exec(',
            '__import__',
            'globals(',
            'locals(',
            'getattr(',
            'setattr(',
            'delattr(',
            'hasattr(',
            'compile(',
            'input(',
            'raw_input('
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False, f"Potentially dangerous operation detected: {pattern}"
        
        return True, "Code appears safe"
    
    def _clean_manipulation_code(self, code):
        """Clean manipulation code to remove return statements and other issues."""
        if not code:
            return code
            
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            # Skip return statements
            if stripped_line.startswith('return '):
                continue
            # Skip standalone 'return'
            if stripped_line == 'return':
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
