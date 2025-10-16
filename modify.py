1) Add a tiny coercion helper (inside the class)

Place this inside class CodeExecutor, anywhere above execute_manipulation:

    @staticmethod
    def _coerce_to_df_or_text(obj):
        """Return (df_or_none, text_or_none) for arbitrary Python objects."""
        import pandas as pd, numpy as np, numbers

        # numpy 0-D → scalar
        if isinstance(obj, np.ndarray) and obj.ndim == 0:
            obj = obj.item()

        # DataFrame
        if hasattr(obj, "shape") and getattr(obj, "shape", None) is not None and len(obj.shape) == 2:
            return obj, None

        # Pandas Series → DataFrame (or scalar if len==1)
        if "pandas" in str(type(obj)).lower() and hasattr(obj, "to_frame"):
            try:
                if getattr(obj, "shape", (0,))[0] == 1:
                    v = obj.iloc[0]
                    return pd.DataFrame({"value": [v]}), str(v)
                return obj.to_frame(), None
            except Exception:
                pass

        # numpy 1-D array → DataFrame
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return pd.DataFrame({"value": obj}), None

        # scalar (number/str/bool or numpy scalar)
        if isinstance(obj, (numbers.Number, str, bool)) or (hasattr(np, "isscalar") and np.isscalar(obj)):
            return pd.DataFrame({"value": [obj]}), str(obj)

        # dict → DataFrame (best effort) or text
        if isinstance(obj, dict):
            try:
                df = pd.DataFrame(obj)
                if df.shape[0] >= 1 and df.shape[1] >= 1:
                    return df, None
            except Exception:
                pass
            return None, str(obj)

        # list of dicts → DataFrame
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            try:
                return pd.DataFrame(obj), None
            except Exception:
                pass

        # fallback: stringify
        try:
            return None, str(obj)
        except Exception:
            return None, None

2) Upgrade execute_manipulation to surface non-DataFrame results
A) After exec(code, namespace), keep your stdout/stderr capture. Then, when you fail to find a DataFrame, try to coerce something useful.

Find this block near the end of execute_manipulation:

            if result_df is not None and isinstance(result_df, pd.DataFrame):
                return result_df
            else:
                print("Error: Code did not return a valid dataframe")
                return None


Replace it with:

            if result_df is not None and isinstance(result_df, pd.DataFrame):
                return result_df
            else:
                # Try to salvage a usable result (scalar/series/dict/array/print)
                # 1) Printed output → tiny table
                printed = stdout_capture.getvalue().strip()
                if printed:
                    try:
                        import pandas as pd
                        return pd.DataFrame({"output": [printed]})
                    except Exception:
                        pass

                # 2) Common variable names
                for cand in ("result", "results", "out", "output"):
                    if cand in namespace:
                        df2, text2 = CodeExecutor._coerce_to_df_or_text(namespace[cand])
                        if df2 is not None:
                            return df2
                        if text2:
                            try:
                                import pandas as pd
                                return pd.DataFrame({"value": [text2]})
                            except Exception:
                                pass

                # 3) Scan newly created variables (ignore builtins/modules)
                import types
                for k, v in namespace.items():
                    if k in ("df", "pd", "np") or k.startswith("__") or isinstance(v, types.ModuleType):
                        continue
                    df2, text2 = CodeExecutor._coerce_to_df_or_text(v)
                    if df2 is not None:
                        return df2
                    if text2:
                        try:
                            import pandas as pd
                            return pd.DataFrame({"value": [text2]})
                        except Exception:
                            pass

                # 4) Evaluate bare last line if it looks like an expression
                try:
                    lines = [ln for ln in (code or "").splitlines() if ln.strip()]
                    if lines:
                        last = lines[-1]
                        stmt_prefixes = ("def ", "class ", "for ", "while ", "if ", "try", "with ", "import ", "from ")
                        if not last.lstrip().startswith(stmt_prefixes):
                            val = eval(compile(last, "<expr>", "eval"), namespace)
                            df2, text2 = CodeExecutor._coerce_to_df_or_text(val)
                            if df2 is not None:
                                return df2
                            if text2:
                                import pandas as pd
                                return pd.DataFrame({"value": [text2]})
                except Exception:
                    pass

                # If truly nothing, return the original df so UI doesn't crash
                print("Warning: No explicit table produced; returning original df.")
                return df.copy()


Now, if the router sends a scalar op like df['model year'].nunique() to manipulation, you’ll still get a 1-row DataFrame to render, instead of None.
