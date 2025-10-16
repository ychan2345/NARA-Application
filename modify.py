Edit 1 — Self-repair success path (handle obs['result'] if it’s a scalar)

Where: Inside execute_analysis(...), in the block that assembles result after run_code_with_self_repair.
Insert this right after the existing dict branch (after the line that handles elif isinstance(obs.get('result'), dict): ...). You’ll see this context: 

code_executor_comp

 and 

code_executor_comp

                    # If a dataframe-like result is present
                    if obs.get('result') is not None:
                        if hasattr(obs.get('result'), 'shape'):
                            result['dataframe'] = obs.get('result')
                        elif isinstance(obs.get('result'), dict):
                            result['text_output'] = str(obs.get('result'))
                        # 🔽 ADD THIS BLOCK
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

Edit 2 — Normal exec fallback (handle exec_env['result'] scalars)

Where: Same execute_analysis(...), in the “Fallback: normal exec” section, after the “Printed output” logic and before the return result. You’ll see the printed output lines and then an immediate return result; insert between them. Context here: printed output lines and return are shown at 

code_executor_comp

 and a fuller block at 

code_executor_comp

            # Printed output
            printed_output = stdout_capture.getvalue().strip()
            if printed_output:
                result['text_output'] = printed_output

            # 🔽 ADD THIS BLOCK (scalar 'result' -> text + tiny table)
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


Why these two spots?

The self-repair success path is where we read obs['result'] and currently only handle DataFrame/dict; adding the scalar case there covers successful self-repair runs. 

code_executor_comp

The fallback path is used when we run exec(code, exec_env) directly; adding scalar handling for exec_env['result'] ensures plain expressions work even without prints or DataFrames.
