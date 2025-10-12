1) Show charts in the main result (Streamlit app)

File: NARA_Comp_V2.py
Where: In the analysis success block (inside elif st.session_state.current_phase == 'analysis': → code path intent != "insight"), you currently have:

st.subheader("📊 AI Response")
text = result.get('text_output')
if text:
    st.write(text)
else:
    st.info("No written response was produced.")


Replace that with this (adds chart rendering, still hides tables/metrics):

st.subheader("📊 AI Response")

# 1) Narrative text
text = result.get('text_output')
if text:
    st.write(text)
else:
    st.info("No written response was produced.")

# 2) Visualization (if any)
# Prefer Plotly; otherwise show Matplotlib/Seaborn
if result.get('plotly_fig') is not None:
    st.plotly_chart(result['plotly_fig'], use_container_width=True, key=f"analysis_chart_{int(pd.Timestamp.now().value)}")
elif result.get('plot') is not None:
    st.pyplot(result['plot'])


Nothing else in your app needs to change for visuals to show in the main response.

Note: You already render visuals inside “Previous Analysis Conversations”, so no change needed there unless you want to remove them.

2) Make sure the repair path actually hands back a figure

If your analysis runs through the self-repair engine, you need that helper to return a figure object under the key fig. You’ve already adapted code_executor.py to route Plotly vs. Matplotlib correctly based on whether obs['fig'] looks like a Plotly object (.data & .layout) or not.

If you sometimes still get no chart even though the code draws one, it’s usually because the repair helper didn’t put any figure into obs['fig']. To make that robust:

File: advanced_analysis.py
Where: inside run_code_with_self_repair(...), right after you execute the candidate code each attempt and assemble obs (stdout/errors/etc.), add a fallback to find a figure automatically if the user code didn’t assign to a variable named fig.

Add something like this after executing the code (where you already capture stdout/result):

# After executing candidate code and before returning obs for this attempt:
# Ensure we return a figure when the code created one, even if it wasn't named 'fig'
if 'fig' not in locals_dict or locals_dict.get('fig') is None:
    detected_fig = None

    # 1) Try to find a Plotly figure in locals
    for name, val in locals_dict.items():
        try:
            if hasattr(val, 'data') and hasattr(val, 'layout'):  # heuristic for Plotly Figure
                detected_fig = val
                break
        except Exception:
            pass

    # 2) If none, try to find a Matplotlib figure
    if detected_fig is None:
        try:
            import matplotlib.pyplot as _plt
            import matplotlib.figure as _mplfig
            for name, val in locals_dict.items():
                if isinstance(val, _mplfig.Figure):
                    detected_fig = val
                    break
            # If still none but there is a current figure, grab it
            if detected_fig is None and _plt.get_fignums():
                detected_fig = _plt.gcf()
        except Exception:
            pass

    if detected_fig is not None:
        obs['fig'] = detected_fig


A few notes:

locals_dict should be whatever mapping you’re using to read back variables from the executed code environment (often the same dict you pass to exec(...)).

You don’t need to change your prompts. This simply makes the repair loop tolerant if the model used a different variable name for the chart or relied on implicit Matplotlib state.
