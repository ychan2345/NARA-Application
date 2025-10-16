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
        """Execute analysis code with self-repair capability and enhanced results.
        Always returns a dict with keys:
        - ok: bool
        - text_output: str | ""
        - dataframe: pd.DataFrame | None
        - plot: matplotlib Figure | None
        - plotly_fig: go.Figure | None
        - self_repair_used: bool
        - repair_attempts: int
        - error: str | ""
        - traceback: str | ""
        - final_code: str | None
        """
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

            # Set up matplotlib
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['font.size'] = 10

            # ---------- Try self-repair engine first ----------
            if query:
                obs, final_code = run_code_with_self_repair(code, exec_env, query)

                if not obs.get("errors"):
                    # Detect whether obs['fig'] is Plotly or Matplotlib/Seaborn
                    plotly_obj = None
                    mpl_obj = None
                    if "fig" in obs and obs["fig"] is not None:
                        f = obs["fig"]
                        if hasattr(f, "data") and hasattr(f, "layout"):
                            plotly_obj = f            # Plotly Figure
                        else:
                            mpl_obj = f               # Matplotlib/Seaborn Figure

                    result = {
                        'ok': True,
                        'text_output': obs.get('stdout', '') or '',
                        'dataframe': None,
                        'plot': mpl_obj,
                        'plotly_fig': plotly_obj,
                        'self_repair_used': obs.get('repair_attempts', 0) > 0,
                        'repair_attempts': int(obs.get('repair_attempts', 0)),
                        'error': '',
                        'traceback': '',
                        'final_code': final_code
                    }
                    # If a dataframe-like result is present
                    if obs.get('result') is not None:
                        if hasattr(obs.get('result'), 'shape'):
                            result['dataframe'] = obs.get('result')
                        elif isinstance(obs.get('result'), dict):
                            result['text_output'] = str(obs.get('result'))
                        elif isinstance(obs.get('result'), (int, float, str, bool)) or np.isscalar(obs.get('result')):
                            scalar_val = obs.get('result')
                            # show as text
                            result['text_output'] = str(scalar_val)
                            # and also as a tiny table so the UI can render it if needed
                            try:
                                import pandas as pd
                                result['dataframe'] = pd.DataFrame({'value': [scalar_val]})
                            except Exception:
                                pass

                    if result.get('dataframe') is None:
                        table = CodeExecutor._find_table_in_namespace(exec_env)
                        if table is not None:
                            result['dataframe'] = table

                    return result

            # ---------- Fallback: normal exec (no self-repair) ----------
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_env)

            result = {
                'ok': True,
                'text_output': '',
                'dataframe': None,
                'plot': None,
                'plotly_fig': None,
                'self_repair_used': False,
                'repair_attempts': 0,
                'error': '',
                'traceback': '',
                'final_code': code
            }

            # Text output from variables
            for var in ('analysis_summary', 'summary', 'result_text'):
                if exec_env.get(var):
                    result['text_output'] = str(exec_env.get(var))
                    break

            # DataFrame results
            if exec_env.get('result_df') is not None and hasattr(exec_env.get('result_df'), 'shape'):
                result['dataframe'] = exec_env.get('result_df')
            elif exec_env.get('result') is not None and hasattr(exec_env.get('result'), 'shape'):
                result['dataframe'] = exec_env.get('result')
            
            if result.get('dataframe') is None:
                table = CodeExecutor._find_table_in_namespace(exec_env)
                if table is not None:
                    result['dataframe'] = table

            # Plotly figures (several common names)
            for var_name in ['plotly_fig', 'fig', 'correlation_heatmap', 'chart', 'visualization']:
                fig_obj = exec_env.get(var_name)
                if fig_obj is not None and hasattr(fig_obj, 'data') and hasattr(fig_obj, 'layout'):
                    result['plotly_fig'] = fig_obj
                    break

            # Matplotlib fallback
            if result['plotly_fig'] is None and plt.get_fignums():
                result['plot'] = plt.gcf()

            # Printed output
            printed_output = stdout_capture.getvalue().strip()
            if printed_output:
                result['text_output'] = printed_output
                
            if not result['text_output']:
                r = exec_env.get('result')
                if r is not None and (isinstance(r, (int, float, str, bool)) or np.isscalar(r)):
                    result['text_output'] = str(r)
                    try:
                        import pandas as pd
                        result['dataframe'] = pd.DataFrame({'value': [r]})
                    except Exception:
                        pass

            return result

        except Exception as e:
            return {
                'ok': False,
                'text_output': '',
                'dataframe': None,
                'plot': None,
                'plotly_fig': None,
                'self_repair_used': False,
                'repair_attempts': 0,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'final_code': code
            }

    
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

    @staticmethod
    def _find_table_in_namespace(ns):
        """Heuristically find a table-like object in an exec namespace."""
        import pandas as pd  # safe if not at top
        candidates = []

        SKIP = {'df','pd','np','plt','sns','px','go','ff','make_subplots','stats','analysis_summary',
                'plotly_fig','fig','result','result_df','results','__builtins__'}

        for k, v in ns.items():
            if k in SKIP or k.startswith('__'):
                continue
            # DataFrame
            if isinstance(v, pd.DataFrame):
                candidates.append((k, v))
                continue
            # Series -> DataFrame
            if hasattr(v, 'to_frame') and callable(getattr(v, 'to_frame', None)):
                try:
                    df = v.to_frame()
                    candidates.append((k, df))
                    continue
                except Exception:
                    pass
            # list of dicts
            if isinstance(v, list) and v and isinstance(v[0], dict):
                try:
                    import pandas as pd
                    df = pd.DataFrame(v)
                    candidates.append((k, df))
                    continue
                except Exception:
                    pass
            # dict -> DataFrame
            if isinstance(v, dict):
                try:
                    import pandas as pd
                    df = pd.DataFrame(v)
                    # avoid 1xN dicts turning into a single-row
                    if df.shape[0] >= 1 and df.shape[1] >= 1:
                        candidates.append((k, df))
                except Exception:
                    pass

        if not candidates:
            return None

        # prefer the largest table (by cells)
        _, best = max(candidates, key=lambda kv: kv[1].shape[0] * kv[1].shape[1])
        return best
