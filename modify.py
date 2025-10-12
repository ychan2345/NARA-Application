1) Show the code in the success UI (retry path)

Where: analysis phase → non-insight branch → after run_with_retry(...), inside the if result and result.get('ok'): block.

Add this right after you write the AI response text:

# --- Show the generated / repaired Python code ---
final_code = result.get('final_code')
if final_code:
    with st.expander("🔍 View Generated Python Code", expanded=False):
        st.code(final_code, language="python")


This pulls the code from result['final_code'] (which your CodeExecutor.execute_analysis() already returns, including the repaired version when auto-repair was used).

2) Save the code into analysis history (so it appears in “Previous Analysis”)

Find this block (right after a successful result):

st.session_state.analysis_history.append({
    'query': analysis_query,
    'code': None,  # keep code hidden per your preference
    'result': result,
    'intent': 'code',
    'timestamp': pd.Timestamp.now()
})


Change it to:

st.session_state.analysis_history.append({
    'query': analysis_query,
    'code': result.get('final_code'),  # <-- store code so history expander shows it
    'result': result,
    'intent': 'code',
    'timestamp': pd.Timestamp.now()
})


Your “Previous Analysis Conversations” section already shows code when analysis['code'] exists, so this makes past runs display their script too.
