@staticmethod
def _find_table_in_namespace(ns):
    import pandas as pd, types
    candidates = []
    SKIP = {'df','pd','np','plt','px','go','ff','make_subplots','stats',
            'analysis_summary','plotly_fig','fig','result','result_df','results','__builtins__'}
    for k, v in ns.items():
        if k in SKIP or str(k).startswith('__') or isinstance(v, types.ModuleType):
            continue
        if isinstance(v, pd.DataFrame):
            candidates.append((k, v)); continue
        if hasattr(v, 'to_frame') and callable(getattr(v, 'to_frame', None)):
            try: candidates.append((k, v.to_frame())); continue
            except Exception: pass
        if isinstance(v, list) and v and isinstance(v[0], dict):
            try:
                candidates.append((k, pd.DataFrame(v))); continue
            except Exception: pass
        if isinstance(v, dict):
            try:
                df = pd.DataFrame(v)
                if df.shape[0] >= 1 and df.shape[1] >= 1:
                    candidates.append((k, df))
            except Exception: pass
    if not candidates: return None
    _, best = max(candidates, key=lambda kv: kv[1].shape[0] * kv[1].shape[1])
    return best
