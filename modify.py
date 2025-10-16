1) Strengthen the self-repair prompt (so repaired code always surfaces scalars)

Where: Inside run_code_with_self_repair(...), find the big fix_prompt = f"""... block.
Edit: Replace the trailing “Rules:” section with this version (add the OUTPUT REQUIREMENTS):

        fix_prompt = f"""You wrote Python code for this data analysis task and it errored on attempt {attempt + 1}/{max_tries}. Fix it and return ONLY the corrected code.

Task: {query}
Current Error: {obs['errors']}
Current code:
{current_code}

Previous repair attempts: {len(repair_history)}
{chr(10).join([f"Attempt {h['attempt']}: {h['error']}" for h in repair_history[-2:]]) if len(repair_history) > 0 else ""}

RULES (MUST FOLLOW EXACTLY):
- Use only variables/libraries already in the environment: df, pd, px, np, stats, go, make_subplots, sns, plt.
- Do not call .show().
- Assign ANY final answer (scalar, Series, or DataFrame) to a variable named result.
- Also print(result) so stdout contains the answer.
- Scalars are VALID final outputs. Do NOT wrap scalars in a DataFrame.
- For distinct counts, use: df[col].nunique(dropna=True).
- For mean/median/std on non-numeric columns, coerce first:
    s = pd.to_numeric(df[col], errors="coerce")
    result = float(s.mean())   # or s.median(), s.std()
- Use exact column names from the dataset.
"""


Why this helps: even when the “average/median/distinct” code evaluates to a scalar, the repaired snippet will set result and also print(result). Your executor already treats stdout/result as success, so the self-repair loop will stop retrying.

2) Make sure you actually instantiate the client for repair calls

Right now, in run_code_with_self_repair, you call client.invoke(messages) but there’s no client = get_openai_client() in that function scope. Add this line right before you build/send messages:

        try:
            messages = [HumanMessage(content=fix_prompt)]
            client = get_openai_client()  # ← ADD THIS
            response = client.invoke(messages)


(Without that, the repair step can fail for reasons unrelated to your analysis code.)
