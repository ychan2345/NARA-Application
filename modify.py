# Printed output
printed_output = stdout_capture.getvalue().strip()
if printed_output:
    result['text_output'] = printed_output

# Accept scalar assigned to exec_env['result']
if not result['text_output']:
    r = exec_env.get('result')
    if r is not None and (isinstance(r, (int, float, str, bool)) or np.isscalar(r)):
        result['text_output'] = str(r)
        try:
            import pandas as pd
            result['dataframe'] = pd.DataFrame({'value': [r]})
        except Exception:
            pass

# ✅ NEW: capture a bare expression (e.g., df['amount'].mean())
if not result['text_output'] and result['dataframe'] is None:
    try:
        # take the last non-empty line and try to eval it
        lines = [ln for ln in code.strip().splitlines() if ln.strip()]
        if lines:
            last = lines[-1]
            # avoid eval on statements (def, import, for, if, etc.)
            stmt_keywords = ('def ', 'class ', 'for ', 'while ', 'if ', 'try:', 'with ', 'import ', 'from ')
            is_statement = last.lstrip().startswith(stmt_keywords)
            if not is_statement:
                val = eval(compile(last, '<expr>', 'eval'), exec_env)
                # Coerce to outputs
                if hasattr(val, 'to_frame'):
                    val = val.to_frame()
                if hasattr(val, 'shape'):  # DataFrame/ndarray
                    result['dataframe'] = val
                elif isinstance(val, (int, float, str, bool)) or np.isscalar(val):
                    result['text_output'] = str(val)
                    try:
                        import pandas as pd
                        result['dataframe'] = pd.DataFrame({'value': [val]})
                    except Exception:
                        pass
    except Exception:
        # swallow; just means the last line wasn't a pure expression
        pass

return result
