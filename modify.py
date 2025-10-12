1) Add two tiny helpers (near your other utils)

Put these under your existing _render_table helper (or anywhere in “Utils”):

def _parse_output_intent(user_query: str):
    q = (user_query or "").lower()
    wants_table = any(k in q for k in [
        "table", "tabular", "as a table", "show table", "list out", "summary table",
        "pivot", "cross-tab", "crosstab", "dataframe", "table format", "grid"
    ])
    wants_both  = any(k in q for k in [
        "both", "table and chart", "chart and table", "show both"
    ])
    return wants_table, wants_both

def _render_chart_from_result(result: dict, key_prefix: str = "") -> bool:
    """Render a chart if present; return True if one was rendered."""
    if result.get('plotly_fig') is not None:
        st.plotly_chart(result['plotly_fig'], use_container_width=True,
                        key=f"{key_prefix}plotly_{int(pd.Timestamp.now().value)}")
        return True
    if result.get('plot') is not None:
        st.pyplot(result['plot'])
        return True
    return False

2) Replace your current “table + visualization” block with a single chooser

Find this section (inside the if result and result.get('ok'): branch):

# --- Table handling (conditional) ---
df_out = result.get('dataframe')
...
# 2) Visualization (if any)
if result.get('plotly_fig') is not None:
    st.plotly_chart(...)
elif result.get('plot') is not None:
    st.pyplot(...)


Replace that whole block with this unified logic:

# --- Unified output chooser: show ONE primary output unless user asks for both ---
wants_table, wants_both = _parse_output_intent(analysis_query)

# Gather possible outputs
df_out = result.get('dataframe')
if df_out is None:
    df_out = _normalize_to_dataframe(result.get('result'))

has_table = df_out is not None
has_chart = (result.get('plotly_fig') is not None) or (result.get('plot') is not None)

# Decide what to show
if wants_both and (has_table or has_chart):
    # Show both (table first), only when explicitly requested
    if has_table:
        st.subheader("📋 Table")
        # full table only if it's small or explicitly asked; otherwise preview + download
        if _should_show_table(analysis_query, df_out):
            _render_table(df_out, key=f"analysis_result_{len(st.session_state.analysis_history)}")
        else:
            with st.expander("📋 Table available (preview first rows)"):
                _render_table(df_out.head(20), key=f"analysis_result_preview_{len(st.session_state.analysis_history)}")
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

    if has_chart:
        st.subheader("📈 Chart")
        _render_chart_from_result(result, key_prefix=f"analysis_chart_")

else:
    # Show a single primary output
    if wants_table and has_table:
        st.subheader("📋 Table")
        if _should_show_table(analysis_query, df_out):
            _render_table(df_out, key=f"analysis_result_{len(st.session_state.analysis_history)}")
        else:
            with st.expander("📋 Table available (preview first rows)"):
                _render_table(df_out.head(20), key=f"analysis_result_preview_{len(st.session_state.analysis_history)}")
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

    elif has_chart:
        st.subheader("📈 Chart")
        _render_chart_from_result(result, key_prefix=f"analysis_chart_")

    elif has_table:
        # No chart but we do have a table → show it
        st.subheader("📋 Table")
        if _should_show_table(analysis_query, df_out):
            _render_table(df_out, key=f"analysis_result_{len(st.session_state.analysis_history)}")
        else:
            with st.expander("📋 Table available (preview first rows)"):
                _render_table(df_out.head(20), key=f"analysis_result_preview_{len(st.session_state.analysis_history)}")
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
