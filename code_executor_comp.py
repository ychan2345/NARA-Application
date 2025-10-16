import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        """Execute analysis code with self-repair and robust output capture.
        Always returns a dict with keys:
        ok, text_output, dataframe, plot, plotly_fig, self_repair_used, repair_attempts, error, traceback, final_code
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go

        # ---------- always start with a "safe default" result ----------
        result = {
            'ok': False,
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

        try:
            # Optional seaborn
            try:
                import seaborn as sns  # may not exist; that's fine
            except Exception:
                sns = None

            exec_env = {
                'df': df.copy(),
                'pd': pd,
                'np': np,
                'plt': plt,
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
                **({'sns': sns} if sns is not None else {}),  # only expose seaborn if available
                **self.safe_builtins
            }

            # Clear plots
            plt.clf(); plt.close('all')
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['font.size'] = 10

            # ---------- 1) Self-repair path (guarded) ----------
            obs = None
            final_code = code
            if query:
                try:
                    obs, final_code = run_code_with_self_repair(code, exec_env, query)
                except Exception as e:
                    # if the self-repair engine crashes, we proceed to normal exec
                    obs = {'errors': [str(e)], 'repair_attempts': 0}
                    final_code = code

            if obs and not obs.get("errors"):
                # Figure (plotly or mpl)
                plotly_obj, mpl_obj = None, None
                if obs.get("fig") is not None:
                    f = obs["fig"]
                    if hasattr(f, "data") and hasattr(f, "layout"):
                        plotly_obj = f
                    else:
                        mpl_obj = f

                result.update({
                    'ok': True,  # assume ok unless we find nothing at all below
                    'text_output': (obs.get('stdout') or '')[:200000],  # cap just in case
                    'plot': mpl_obj,
                    'plotly_fig': plotly_obj,
                    'self_repair_used': int(obs.get('repair_attempts', 0)) > 0,
                    'repair_attempts': int(obs.get('repair_attempts', 0)),
                    'final_code': final_code
                })

                # Primary result
                if obs.get('result') is not None:
                    r = obs['result']
                    if hasattr(r, 'shape'):                 # DataFrame / ndarray
                        result['dataframe'] = r
                    elif isinstance(r, dict):               # dict -> text
                        result['text_output'] = str(r)
                    elif isinstance(r, (int, float, str, bool)) or np.isscalar(r):
                        # scalar -> text + tiny table
                        result['text_output'] = str(r)
                        try:
                            result['dataframe'] = pd.DataFrame({'value': [r]})
                        except Exception:
                            pass

                # If still no table, try to find one in namespace
                if result.get('dataframe') is None:
                    table = CodeExecutor._find_table_in_namespace(exec_env)
                    if table is not None:
                        result['dataframe'] = table

                # Mark ok=false only if literally nothing to show
                if not result['text_output'] and result['dataframe'] is None and result['plot'] is None and result['plotly_fig'] is None:
                    result['ok'] = False
                    result['error'] = 'No output produced by analysis.'
                return result

            # ---------- 2) Fallback: normal exec ----------
            stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_env)

            # Collect figures (plotly first)
            for var in ['plotly_fig', 'fig', 'correlation_heatmap', 'chart', 'visualization']:
                fig_obj = exec_env.get(var)
                if fig_obj is not None and hasattr(fig_obj, 'data') and hasattr(fig_obj, 'layout'):
                    result['plotly_fig'] = fig_obj
                    break
            if result['plotly_fig'] is None and plt.get_fignums():
                result['plot'] = plt.gcf()

            # Text outputs from variables
            for var in ('analysis_summary', 'summary', 'result_text'):
                if exec_env.get(var):
                    result['text_output'] = str(exec_env.get(var))
                    break

            # DataFrame-like from well-known names
            if result.get('dataframe') is None:
                if exec_env.get('result_df') is not None and hasattr(exec_env['result_df'], 'shape'):
                    result['dataframe'] = exec_env['result_df']
                elif exec_env.get('result') is not None and hasattr(exec_env['result'], 'shape'):
                    result['dataframe'] = exec_env['result']

            # Try to salvage a table from namespace
            if result.get('dataframe') is None:
                table = CodeExecutor._find_table_in_namespace(exec_env)
                if table is not None:
                    result['dataframe'] = table

            # Printed output wins for text
            printed_output = stdout_capture.getvalue().strip()
            if printed_output:
                result['text_output'] = printed_output

            # Accept scalar assigned to exec_env['result']
            if not result['text_output']:
                r = exec_env.get('result')
                if r is not None and (isinstance(r, (int, float, str, bool)) or np.isscalar(r)):
                    result['text_output'] = str(r)
                    try:
                        result['dataframe'] = pd.DataFrame({'value': [r]})
                    except Exception:
                        pass

            # Capture bare last-line expression if nothing else yet
            if not result['text_output'] and result['dataframe'] is None:
                try:
                    lines = [ln for ln in (code or '').splitlines() if ln.strip()]
                    if lines:
                        last = lines[-1]
                        stmt_prefixes = ('def ', 'class ', 'for ', 'while ', 'if ', 'try', 'with ', 'import ', 'from ')
                        if not last.lstrip().startswith(stmt_prefixes):
                            val = eval(compile(last, '<expr>', 'eval'), exec_env)
                            if hasattr(val, 'to_frame'):
                                val = val.to_frame()
                            if hasattr(val, 'shape'):
                                result['dataframe'] = val
                            elif isinstance(val, (int, float, str, bool)) or np.isscalar(val):
                                result['text_output'] = str(val)
                                try:
                                    result['dataframe'] = pd.DataFrame({'value': [val]})
                                except Exception:
                                    pass
                except Exception:
                    # swallow: expression capture is best-effort
                    pass

            # Final ok/error
            if result['text_output'] or result['dataframe'] is not None or result['plot'] is not None or result['plotly_fig'] is not None:
                result['ok'] = True
            else:
                result['ok'] = False
                # surface stderr to help upstream show something useful
                errtxt = stderr_capture.getvalue().strip()
                if errtxt:
                    result['error'] = errtxt
            return result

        except Exception as e:
            # absolutely never raise; upstream treats exceptions as "analysis failed"
            result.update({
                'ok': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })
            return result
    
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
