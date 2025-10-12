1) Add a tiny helper to save chat messages (works for both branches)

Insert anywhere after your imports / state init (e.g., right below your other small helpers):

# --- helper: persist chat history uniformly ---
def _save_chat_message(kind: str, text: str):
    """Append a chat message to session state and DB."""
    if 'chat_history' not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({
        'type': kind,
        'query': text,
        'timestamp': pd.Timestamp.now()
    })
    try:
        persistence.save_chat_message(st.session_state.session_id, text, kind)
    except Exception:
        pass
# --- end helper ---

2) Record the user’s question for BOTH insight & code paths

Find this line in the analysis handler (inside elif st.session_state.current_phase == 'analysis':):

intent = data_agent.route_query_intelligently(analysis_query, code_feedback_format, insight_feedback_format)


Immediately after it, insert:

# record the user's question in chat history (works for both branches)
_save_chat_message('analysis', analysis_query)


Now remove any old duplicates lower down (you likely have these in the code branch):

# DELETE these two lines if present (the helper replaces them)
st.session_state.chat_history.append({'type': 'analysis', 'query': analysis_query, 'timestamp': pd.Timestamp.now()})
persistence.save_chat_message(st.session_state.session_id, analysis_query, 'analysis')


Keep everything else the same. This guarantees the message is saved for both “insight” and “code” flows.

3) Show chat history on the Analysis page

Place this block near the top of the Analysis page UI (e.g., right BEFORE “Quick Actions”):

# ---- Show chat history (previous questions) ----
if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
    with st.expander(f"💬 Chat History ({len(st.session_state.chat_history)} questions)", expanded=False):
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            query_text = chat.get('query', '')
            ts = chat.get('timestamp', '')
            try:
                time_str = ts if isinstance(ts, str) else ts.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = str(ts)
            st.markdown(f"📊 **Q{len(st.session_state.chat_history)-i}:** {query_text}")
            if time_str:
                st.caption(f"🕐 {time_str}")
            if i < len(st.session_state.chat_history) - 1:
                st.divider()


Good insertion point example:

Right after your inline “Fix Column Data Types” expander finishes, and

Right before:

st.write("**Quick Actions:**")
