from __future__ import annotations

import os
from pathlib import Path

def get_google_api_key() -> str:
    """
    Robust API key loader that works on:
    - Streamlit Cloud (st.secrets)
    - Local dev with .streamlit/secrets.toml
    - Local dev with .env (python-dotenv)
    - Local dev with exported environment variables
    """
    # 1) Try Streamlit secrets (only if Streamlit is available)
    try:
        import streamlit as st  # type: ignore

        # st.secrets behaves like a mapping; avoid hard KeyError
        key = st.secrets.get("GOOGLE_API_KEY", None)
        if key and str(key).strip():
            return str(key).strip()
    except Exception:
        # Not running in Streamlit (or Streamlit not installed yet)
        pass

    # 2) Try environment variable
    key = os.getenv("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()

    # 3) Try .env (local only)
    try:
        from dotenv import load_dotenv  # type: ignore

        # Only attempt if a .env exists (avoid confusing behavior in Cloud)
        if Path(".env").exists():
            load_dotenv(override=False)
            key = os.getenv("GOOGLE_API_KEY")
            if key and key.strip():
                return key.strip()
    except Exception:
        pass

    raise RuntimeError(
        "GOOGLE_API_KEY not found.\n\n"
        "Fix:\n"
        "• Streamlit Cloud: add GOOGLE_API_KEY in App → Settings → Secrets\n"
        "• Local: set GOOGLE_API_KEY env var, or put it in .env, or .streamlit/secrets.toml"
    )
