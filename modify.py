✅ Edit 1 — Add a general coercion helper inside the class

Where: Inside class CodeExecutor, put this helper just above execute_analysis (or anywhere inside the class).

    @staticmethod
    def _coerce_to_displayables(obj):
        """Return a dict like {'text': str|None, 'dataframe': pd.DataFrame|None} from arbitrary Python objects."""
        import pandas as pd
        import numpy as np
        import numbers

        out_text = None
        out_df = None

        # Unwrap numpy 0-D arrays / scalars
        if isinstance(obj, np.ndarray) and obj.ndim == 0:
            obj = obj.item()

        # Pandas DataFrame
        if hasattr(obj, "shape") and getattr(obj, "shape", None) is not None and len(obj.shape) == 2:
            out_df = obj
            return {"text": out_text, "dataframe": out_df}

        # Pandas Series -> DataFrame (or scalar if length 1)
        if "pandas" in str(type(obj)).lower() and hasattr(obj, "to_frame"):
            try:
                import pandas as pd  # local import safe
                if getattr(obj, "shape", (0,))[0] == 1:
                    # single value series → scalar
                    try:
                        obj = obj.iloc[0]
                        out_text = str(obj)
                        out_df = pd.DataFrame({"value": [obj]})
                        return {"text": out_text, "dataframe": out_df}
                    except Exception:
                        pass
                out_df = obj.to_frame()  # best effort
                return {"text": out_text, "dataframe": out_df}
            except Exception:
                pass

        # Numpy 1-D arrays → DataFrame
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            import pandas as pd
            out_df = pd.DataFrame({"value": obj})
            return {"text": out_text, "dataframe": out_df}

        # Numbers / strings / booleans / numpy scalars → scalar
        if isinstance(obj, (numbers.Number, str, bool)) or (hasattr(np, "isscalar") and np.isscalar(obj)):
            import pandas as pd
            out_text = str(obj)
            out_df = pd.DataFrame({"value": [obj]})
            return {"text": out_text, "dataframe": out_df}

        # dict → DataFrame (best effort)
        if isinstance(obj, dict):
            try:
                import pandas as pd
                df = pd.DataFrame(obj)
                # prefer non-empty tables
                if df.shape[0] >= 1 and df.shape[1] >= 1:
                    out_df = df
                else:
                    out_text = str(obj)
                return {"text": out_text, "dataframe": out_df}
            except Exception:
                out_text = str(obj)
                return {"text": out_text, "dataframe": None}

        # list of dicts → DataFrame
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            try:
                import pandas as pd
                out_df = pd.DataFrame(obj)
                return {"text": out_text, "dataframe": out_df}
            except Exception:
                pass

        # Fallback: stringify if nothing else
        try:
            out_text = str(obj)
        except Exception:
            out_text = None
        return {"text": out_text, "dataframe": out_df}

✅ Edit 2 — In the self-repair success branch, use the coercer

Where: inside execute_analysis, in the block:

if obs and not obs.get("errors"):
    ...
    if obs.get('result') is not None:
        r = obs['result']
        if hasattr(r, 'shape'): ...
        elif isinstance(r, dict): ...
        elif isinstance(r, (int, float, str, bool)) or np.isscalar(r): ...


Replace that entire inner if obs.get('result')... block with:

                if obs.get('result') is not None:
                    co = CodeExecutor._coerce_to_displayables(obs['result'])
                    if co.get("dataframe") is not None:
                        result['dataframe'] = co["dataframe"]
                    if co.get("text"):
                        # append to stdout-derived text if present
                        if result['text_output']:
                            result['text_output'] += ("\n" + co["text"])
                        else:
                            result['text_output'] = co["text"]


(This single call handles DataFrames, Series, scalars, arrays, dicts—no special-casing.)

Leave the rest of that branch as-is (including your namespace table finder and plots).

✅ Edit 3 — In the fallback (normal exec) branch, add “new variables” scan + use the coercer

We’ll do three small insertions:

3A) Snapshot variable names before exec

Find:

stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
    exec(code, exec_env)


Insert this line just before the with ... exec(...) block:

pre_vars = set(exec_env.keys())


So it becomes:

pre_vars = set(exec_env.keys())
stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
    exec(code, exec_env)

3B) Keep your current printed/series/dataframe handling (as-is), then add a new variables scan just before the bare-expression capture:

Find this part (near the end, before you capture the bare last line):

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


Immediately after that block, insert this “scan new vars” block:

# If still nothing, look at NEW variables created by the code and pick a usable one
if not result['text_output'] and result['dataframe'] is None:
    post_vars = set(exec_env.keys())
    new_vars = [k for k in (post_vars - pre_vars) if not str(k).startswith('__')]
    # Heuristic priority: DataFrame/Series/ndarray -> numeric/scalar -> dict/list-of-dicts -> string
    picked = None
    # 1) Prefer frames/series/arrays
    for k in new_vars:
        v = exec_env.get(k)
        co = CodeExecutor._coerce_to_displayables(v)
        if co.get("dataframe") is not None:
            picked = co; break
    # 2) Otherwise choose first that gives a scalar/text
    if picked is None:
        for k in new_vars:
            v = exec_env.get(k)
            co = CodeExecutor._coerce_to_displayables(v)
            if co.get("text") or co.get("dataframe") is not None:
                picked = co; break
    if picked is not None:
        if picked.get("dataframe") is not None:
            result['dataframe'] = picked["dataframe"]
        if picked.get("text") and not result['text_output']:
            result['text_output'] = picked["text"]

3C) Keep your existing bare last-line expression capture (you already have it). It will only run if the above didn’t find anything, making the whole thing robust.
✅ (Optional but recommended) Use the coercer for eval(...) capture too

Inside your “bare last-line expression” block, where you set val = eval(...) and then do a bunch of if/else checks, you can replace that with the coercer:

Replace inside that try-block:

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


With:

val = eval(compile(last, '<expr>', 'eval'), exec_env)
co = CodeExecutor._coerce_to_displayables(val)
if co.get("dataframe") is not None:
    result['dataframe'] = co["dataframe"]
if co.get("text") and not result['text_output']:
    result['text_output'] = co["text"]
