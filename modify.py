1) code_executor.py — add a table finder and use it
A) Add this helper (near the top of the file, after imports or inside the class as a @staticmethod)
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

B) In execute_analysis(...), after you’ve executed the code and built the result dict, add a fallback if result['dataframe'] is still None.

You have two paths in your executor:

Self-repair path (when query passed; you already read obs).

Normal exec path.

Add the salvage block in both.

In the self-repair success branch (right after you set result and before return result):
# If a dataframe-like result is present (you already try obs['result'])
if obs.get('result') is not None:
    if hasattr(obs.get('result'), 'shape'):
        result['dataframe'] = obs.get('result')
    elif isinstance(obs.get('result'), dict):
        result['text_output'] = str(obs.get('result'))

# ⬇️ NEW: salvage any table from the exec env if result['dataframe'] is still empty
if result.get('dataframe') is None:
    table = _find_table_in_namespace(exec_env)
    if table is not None:
        result['dataframe'] = table

In the normal exec branch (after you populate text_output/dataframe and before return result):
# Existing: try result_df/result etc...
if exec_env.get('result_df') is not None and hasattr(exec_env.get('result_df'), 'shape'):
    result['dataframe'] = exec_env.get('result_df')
elif exec_env.get('result') is not None and hasattr(exec_env.get('result'), 'shape'):
    result['dataframe'] = exec_env.get('result')

# ⬇️ NEW: salvage any table from the exec env if still missing
if result.get('dataframe') is None:
    table = _find_table_in_namespace(exec_env)
    if table is not None:
        result['dataframe'] = table


That’s it for the executor. Now, even if the LLM forgets to assign the table to result, you’ll still surface one.

2) (Optional but recommended) advanced_analysis.py — make self-repair set result when missing

If you’re using run_code_with_self_repair(...), add the same salvage right after exec(cleaned, exec_env) and before you copy vars to obs:

# After exec, before building obs
if "result" not in exec_env:
    try:
        from code_executor import _find_table_in_namespace
        tbl = _find_table_in_namespace(exec_env)
        if tbl is not None:
            exec_env["result"] = tbl
    except Exception:
        pass


(If _find_table_in_namespace is inside the class, duplicate the minimal logic here instead of importing.)

Why this fixes your symptom

The LLM often names tables summary_df, table, pivot, etc.

Your Streamlit app shows tables only when the executor returns dataframe (or result convertible to a DataFrame).

With the “table salvage” block, any reasonable tabular output in the namespace will be found and returned—so the “Table” section won’t be empty when the AI text talks about a summary table.

No changes are needed in your Streamlit file you pasted above (you already display result['dataframe']).
