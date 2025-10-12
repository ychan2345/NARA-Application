1) Add a tiny helper to normalize table-like results

Put this helper near your other utils (right after apply_manual_conversion_fallback(...) is a good spot):

def _normalize_to_dataframe(obj):
    """Turn common table-shaped outputs into a DataFrame."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # list of dicts
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        # dict of lists/scalars
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    return None

2) Show a table when the model returns one

Find this block (inside the analysis branch where intent != "insight" and after run_with_retry(...)):

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


Replace it with:

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

    # --- AI text (unchanged) ---
    st.subheader("📊 AI Response")
    text = result.get('text_output')
    if text:
        st.write(text)
    else:
        st.info("No written response was produced.")

    # --- NEW: show a table if provided ---
    # Primary path: CodeExecutor fills 'dataframe' when code assigns to result/result_df
    df_out = result.get('dataframe')

    # Fallback: if your executor ever returns a raw 'result' payload, normalize it
    if df_out is None:
        df_out = _normalize_to_dataframe(result.get('result'))

    if df_out is not None:
        st.subheader("📋 Table")
        # unique key to avoid re-render collisions
        unique_key = f"analysis_result_{len(st.session_state.analysis_history)}"
        display_dataframe(df_out, unique_key=unique_key)
