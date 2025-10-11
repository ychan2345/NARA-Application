# Handle both string and datetime timestamp formats
                    timestamp = chat.get('timestamp', '')
                    if isinstance(timestamp, str):
                        time_str = timestamp
                    else:
                        try:
                            time_str = timestamp.strftime('%H:%M:%S')
                        except:
                            time_str = str(timestamp)
                    
                    st.write(f"**{chat['type'].title()}** - {time_str}")
                    st.write(f"Query: {chat['query']}")
                    st.divider()

Found
# Apply changes if successful
if success:
    st.session_state.current_df = df_temp
    
    # Also update approved_df if it exists (user is in analysis phase)
    if st.session_state.approved_df is not None:
        st.session_state.approved_df = df_temp.copy()
        st.info("🔄 Approved dataset has been updated with the new data type change")
    
    st.session_state.manipulation_history.append(f"Changed {column_to_modify} data type to {new_dtype}")
    st.success(f"✅ Successfully changed {column_to_modify} to {dtype_options[new_dtype]}")
    st.rerun()


Replace:
# Apply changes if successful
if success:
    st.session_state.current_df = df_temp
    st.session_state.manipulation_history.append(f"Changed {column_to_modify} data type to {new_dtype}")
    st.success(f"✅ Successfully changed {column_to_modify} to {dtype_options[new_dtype]}")
    st.rerun()


Found:
# Update current dataframe only if there was a meaningful change
if new_rows != original_rows or not result_df.equals(st.session_state.current_df):
    st.session_state.current_df = result_df
    
    # Also update approved_df if it exists (user is in analysis phase)
    if st.session_state.approved_df is not None:
        st.session_state.approved_df = result_df.copy()
        st.info("🔄 Approved dataset has been updated with the manipulation")

Replace:
# Update current dataframe only if there was a meaningful change
if new_rows != original_rows or not result_df.equals(st.session_state.current_df):
    st.session_state.current_df = result_df

Found
if st.button("✅ Approve Dataset for Analysis", type="primary"):
    st.session_state.approved_df = st.session_state.current_df.copy()
    st.session_state.current_phase = 'analysis'
    ...

Replace
if st.button("✅ Approve Dataset for Analysis", type="primary"):
    st.session_state.approved_df = st.session_state.current_df.copy()
    st.session_state.approved_source = "current"  # <-- add this line
    st.session_state.current_phase = 'analysis'
    ...

Found
if st.button("🔄 Reset to Original"):
    if st.session_state.original_df is not None:
        st.session_state.current_df = st.session_state.original_df.copy()
        st.session_state.manipulation_history = []
        # Clear approved_df since we've reset to original - must re-approve
        st.session_state.approved_df = None
        ...

Replace
if st.button("🔄 Reset to Original"):
    if st.session_state.original_df is not None:
        st.session_state.current_df = st.session_state.original_df.copy()
        st.session_state.manipulation_history = []
        # Clear approved_df since we've reset to original - must re-approve
        st.session_state.approved_df = None
        st.session_state.approved_source = None  # <-- add this line
        ...

Found
if st.button("🔧 Back to Manipulation"):
    st.session_state.current_phase = 'manipulation'
    # Clear approved_df so it must be re-approved after any changes
    st.session_state.approved_df = None
    ...

Replace
if st.button("🔧 Back to Manipulation"):
    st.session_state.current_phase = 'manipulation'
    # Clear approved_df so it must be re-approved after any changes
    st.session_state.approved_df = None
    st.session_state.approved_source = None  # <-- add this line
    ...


Found
# Debug info to track dataset state
if st.session_state.approved_df is not None and st.session_state.original_df is not None:
    if len(st.session_state.approved_df) == len(st.session_state.original_df):
        st.info(f"✅ Using ORIGINAL dataset ({len(st.session_state.approved_df)} rows)")
    else:
        st.warning(f"⚠️ Using FILTERED dataset ({len(st.session_state.approved_df)} rows, original had {len(st.session_state.original_df)} rows)")

Replace
# Debug info to track dataset state
if st.session_state.approved_df is not None and st.session_state.original_df is not None:
    if st.session_state.approved_df.equals(st.session_state.original_df):
        st.info(f"✅ Using ORIGINAL dataset ({len(st.session_state.approved_df)} rows)")
    else:
        src = st.session_state.get("approved_source", "modified")
        st.warning(
            f"⚠️ Using {src.upper()} dataset "
            f"({len(st.session_state.approved_df)} rows, original had {len(st.session_state.original_df)} rows)"
        )


Optional
# Prevent accidental edits when an approved dataset exists
if st.session_state.get("approved_df") is not None:
    st.warning("You currently have an approved dataset. Click **'🔧 Back to Manipulation'** in the sidebar to modify data.")
    st.stop()

