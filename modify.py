if st.button("➕ New Session", key="create_new_session"):
    # Create a fresh session id but keep current dataset & phase
    new_session_id = persistence.create_session(
        session_name=f"New Session {pd.Timestamp.now().strftime('%H:%M')}",
        browser_id=st.session_state.browser_id
    )
    st.session_state.session_id = new_session_id
    cookies['session_id'] = new_session_id
    cookies.save()

    # Keep the dataset and phase as-is (stay in analysis mode)
    # DO NOT reset original_df/current_df/approved_df/selected_file/uploaded_files/unique_categories
    # Only clear chat history
    st.session_state.chat_history = []

    # (Optional) If you also want to clear the analysis history, uncomment the next line:
    # st.session_state.analysis_history = []

    # Make sure we persist the current dataset & phase into the new session row
    current_phase = 'analysis' if st.session_state.approved_df is not None else st.session_state.current_phase
    persistence.save_session_state(
        st.session_state.session_id,
        original_df=st.session_state.original_df,
        current_df=st.session_state.current_df,
        approved_df=st.session_state.approved_df,
        phase=current_phase
    )

    st.success("New session created — kept current dataset & analysis mode, cleared chat history.")
    st.rerun()
