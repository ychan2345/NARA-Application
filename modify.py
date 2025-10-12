A) Make execute_analysis return structured failure info

Update your CodeExecutor.execute_analysis to always return a dict that includes success/failure flags and the last code used, so the app can retry intelligently.

Patch for code_executor.py

Replace your current execute_analysis with this version (only this method — everything else unchanged):

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

            # Success via self-repair
            if not obs.get("errors"):
                result = {
                    'ok': True,
                    'text_output': obs.get('stdout', '') or '',
                    'dataframe': None,
                    'plot': None,
                    'plotly_fig': obs.get('fig'),
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

            # Self-repair failed: return structured failure so caller can retry
            return {
                'ok': False,
                'text_output': '',
                'dataframe': None,
                'plot': None,
                'plotly_fig': None,
                'self_repair_used': False,
                'repair_attempts': int(obs.get('repair_attempts', 0)),
                'error': "\n".join(obs.get('errors', [])),
                'traceback': obs.get('traceback', ''),
                'final_code': final_code
            }

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


What changed

Returns a dict on failure instead of None, with ok=False, error, traceback, final_code, and repair_attempts.

On self-repair success, includes self_repair_used and final_code.

Normal exec still works and returns ok=True.

This lets the app know when first-pass repair failed and gives it the code/error to ask the agent for a corrected version.

B) Add a light-weight retry loop in the app

In your Streamlit app’s analysis branch (the “code” path), wrap execution with a small orchestrator that:

generates code,

calls execute_analysis(...),

if ok is False, asks the agent to fix using the error + final_code, and re-runs (up to 2–3 tries).

Drop-in function (put near your other helpers)
def run_with_retry(base_df, user_query, data_agent, code_executor, unique_categories, chat_history, max_attempts=3):
    """
    Returns (result_dict, attempts_used)
    result_dict respects CodeExecutor's schema and includes 'ok' boolean.
    """
    attempts = 0

    # 1) First code generation
    code = data_agent.generate_analysis_code(
        base_df,
        user_query,
        chat_history=chat_history,
        unique_categories=unique_categories
    )
    if not code:
        return ({'ok': False, 'error': 'Code generation failed', 'text_output': ''}, attempts)

    while attempts < max_attempts:
        attempts += 1
        res = code_executor.execute_analysis(base_df, code, user_query)
        if res and res.get('ok'):
            # Mark self-repair usage if attempts > 1
            if attempts > 1:
                res['self_repair_used'] = True
            return (res, attempts)

        # Prepare repair prompt using error + final_code from executor
        err = (res or {}).get('error', '')
        tb = (res or {}).get('traceback', '')
        last_code = (res or {}).get('final_code', code)

        repair_prompt = (
            f"{user_query}\n\n"
            f"The previous Python code failed with this error:\n{err}\n\n"
            f"Traceback:\n{tb}\n\n"
            f"Here is the failing code:\n\n{last_code}\n\n"
            "Return a fully corrected Python script ONLY (no markdown). "
            "It must be robust to missing columns and data types, avoid inplace pitfalls, "
            "and execute without errors."
        )
        code = data_agent.generate_analysis_code(
            base_df,
            repair_prompt,
            chat_history=chat_history,
            unique_categories=unique_categories
        )
        if not code:
            break

    return (res or {'ok': False, 'error': 'Exhausted retries'}, attempts)

Use it in your analysis handler (code path)

Replace your current single-shot generation/execution with:

st.info("⚙️ Generating and executing analysis code (with auto-repair)...")
base_df = st.session_state.approved_df if st.session_state.approved_df is not None else st.session_state.current_df

result, attempts_used = run_with_retry(
    base_df=base_df,
    user_query=analysis_query,
    data_agent=data_agent,
    code_executor=code_executor,
    unique_categories=st.session_state.unique_categories,
    chat_history=st.session_state.chat_history,
    max_attempts=3
)

if result and result.get('ok'):
    st.session_state.analysis_history.append({
        'query': analysis_query,
        'code': None,  # keep code hidden per your preference
        'result': result,
        'intent': 'code',
        'timestamp': pd.Timestamp.now()
    })
    msg = "✅ Analysis completed successfully!"
    if result.get('self_repair_used'):
        msg += f" (auto-repaired in {attempts_used} attempt(s))"
    st.success(msg)

    # Show ONLY the AI text (per your previous requirement)
    st.subheader("📊 AI Response")
    text = result.get('text_output')
    if text:
        st.write(text)
    else:
        st.info("No written response was produced.")
else:
    st.error("❌ Analysis failed after automatic repair attempts.")
    # Optional, for debugging:
    # st.caption(result.get('error', ''))


You can keep the insight branch unchanged; it’s not code-execution heavy. Just ensure you still call _save_chat_message('analysis', analysis_query) so history renders on session switch.
