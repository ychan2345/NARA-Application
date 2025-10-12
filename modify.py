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
    return result
