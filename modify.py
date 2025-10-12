1) Add two helpers (put them right under _normalize_to_dataframe(...))
def _should_show_table(user_query: str, df: pd.DataFrame) -> bool:
    """
    Show the full table if the user explicitly asks for it OR
    if it's reasonably small and there isn't an obvious chart to show.
    """
    if df is None:
        return False

    q = (user_query or "").lower()
    explicit_keywords = [
        "table", "tabular", "as a table", "show table", "list out", "summary table",
        "pivot", "cross-tab", "crosstab"
    ]
    if any(k in q for k in explicit_keywords):
        return True

    # Heuristic: small enough to be readable
    return (df.shape[0] <= 200 and df.shape[1] <= 20)

def _render_table(df: pd.DataFrame, key: str):
    """Nice-looking Streamlit table with number formatting and hidden index."""
    pretty = df.copy()

    # Prefer common time columns first
    preferred_order = [c for c in ["year", "month", "date"] if c in pretty.columns] + \
                      [c for c in pretty.columns if c not in ["year", "month", "date"]]
    pretty = pretty[preferred_order]

    # Ensure numeric formatting
    numeric_cols = pretty.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        pretty[c] = pd.to_numeric(pretty[c], errors="coerce")

    col_cfg = {c: st.column_config.NumberColumn(c, format="%,.2f") for c in numeric_cols}

    st.dataframe(pretty, hide_index=True, use_container_width=True, column_config=col_cfg, key=key)

2) Replace your current “show table” block in the code branch where analysis succeeds

Find this part in your analysis phase (inside the if result and result.get('ok'): block):

# --- NEW: show a table if provided ---
df_out = result.get('dataframe')
if df_out is None:
    df_out = _normalize_to_dataframe(result.get('result'))

if df_out is not None:
    st.subheader("📋 Table")
    unique_key = f"analysis_result_{len(st.session_state.analysis_history)}"
    display_dataframe(df_out, unique_key=unique_key)


Replace it with this logic that only shows the table when appropriate:

# --- Table handling (conditional) ---
df_out = result.get('dataframe')
if df_out is None:
    df_out = _normalize_to_dataframe(result.get('result'))

if df_out is not None:
    if _should_show_table(analysis_query, df_out):
        st.subheader("📋 Table")
        _render_table(df_out, key=f"analysis_result_{len(st.session_state.analysis_history)}")
    else:
        # Keep it lightweight in the UI but make it available
        with st.expander("📋 Table available (preview first rows)"):
            _render_table(df_out.head(20), key=f"analysis_result_preview_{len(st.session_state.analysis_history)}")
        # Optional: download link if you want to allow export
        try:
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full table as CSV",
                data=csv_bytes,
                file_name="analysis_result.csv",
                mime="text/csv",
                key=f"dl_{len(st.session_state.analysis_history)}"
            )
        except Exception:
            pass
