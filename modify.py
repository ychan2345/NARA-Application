1) Accept scalars after self-repair succeeds

File: code_executor_comp.py
Block: inside execute_analysis, in the self-repair success branch, right after the existing elif isinstance(obs.get('result'), dict): line. 

code_executor_comp

Add:

                        elif isinstance(obs.get('result'), (int, float, str, bool)) or np.isscalar(obs.get('result')):
                            scalar_val = obs.get('result')
                            result['text_output'] = str(scalar_val)
                            try:
                                import pandas as pd
                                result['dataframe'] = pd.DataFrame({'value': [scalar_val]})
                            except Exception:
                                pass


Then fix the helper call a few lines below:

Change:

                    if result.get('dataframe') is None:
                        table = _find_table_in_namespace(exec_env)


To:

                    if result.get('dataframe') is None:
                        table = CodeExecutor._find_table_in_namespace(exec_env)


code_executor_comp

2) Accept scalars & capture bare expressions in the fallback path

File: code_executor_comp.py
Block: still execute_analysis, fallback branch. After you read printed_output, before the return result. 

code_executor_comp

Replace this tail section:

# Printed output
printed_output = stdout_capture.getvalue().strip()
if printed_output:
    result['text_output'] = printed_output

return result


With this:

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

# Capture a bare last-line expression like: df['amount'].mean()
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
                        import pandas as pd
                        result['dataframe'] = pd.DataFrame({'value': [val]})
                    except Exception:
                        pass
    except Exception:
        pass

return result


(If you already added a scalar block, keep it; just ensure this expression-capture block is present too.)

Also fix the earlier helper call in fallback:

Change:

table = _find_table_in_namespace(exec_env)


To:

table = CodeExecutor._find_table_in_namespace(exec_env)


code_executor_comp

3) Only inject seaborn if available (optional but safer)

File: code_executor_comp.py
Block: where you build exec_env. Replace the hard 'sns': sns with a conditional dict expansion. 

code_executor_comp

Change:

'sns': sns,


To:

**({'sns': sns} if 'sns' in globals() else {}),


(Or preface the import with a try/except at the top and check sns is not None.)

4) Make sure the consumer treats scalar/text as “success”

Your UI only marks the whole run as success when the executor returns ok=True. That happens in your code path; the “analysis failed after 3 attempts” banner only shows when result is falsy or ok is False. 

NARA_Comp_V2

 

NARA_Comp_V2

If you still see failures after the edits above, it means the executor returned ok=False (likely due to an exception). Grab error/traceback from your executor’s return and surface it in the UI for debugging (you already have a commented line for this):

# Optional, for debugging:
# st.caption(result.get('error', ''))
