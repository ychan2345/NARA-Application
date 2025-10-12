def _render_table(df: pd.DataFrame, key: str):
    """Nice-looking Streamlit table with number formatting and hidden index.
    Ensures index (e.g., Year) is visible as a column."""
    pretty = df.copy()

    # 1) If the index is named or not a simple RangeIndex, bring it back as a column
    if not isinstance(pretty.index, pd.RangeIndex) or getattr(pretty.index, "name", None):
        pretty = pretty.reset_index()

    # 2) Flatten any MultiIndex columns
    if isinstance(pretty.columns, pd.MultiIndex):
        pretty.columns = ["_".join([str(x) for x in tup if x != ""]) for tup in pretty.columns.values]

    # 3) Case-insensitive preferred ordering (Year/Month/Date first if present)
    cols = list(pretty.columns)
    lower_map = {c.lower(): c for c in cols}
    preferred = [lower_map[k] for k in ["year", "month", "date"] if k in lower_map]
    rest = [c for c in cols if c not in preferred]
    pretty = pretty[preferred + rest] if preferred else pretty

    # 4) Ensure numeric formatting
    numeric_cols = pretty.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        pretty[c] = pd.to_numeric(pretty[c], errors="coerce")

    col_cfg = {c: st.column_config.NumberColumn(c, format="%,.2f") for c in numeric_cols}

    st.dataframe(pretty, hide_index=True, use_container_width=True, column_config=col_cfg, key=key)
