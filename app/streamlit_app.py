from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "model"))

from inference import generate_rewrite, load_model  # noqa: E402


st.set_page_config(page_title="Professional Email Rewriter")

st.title("Professional Email Rewriter")


@st.cache_resource
def cached_model():
    return load_model()


rough_email = st.text_area("Rough email", height=180)

if st.button("Rewrite"):
    if rough_email.strip():
        with st.spinner("Rewriting..."):
            tokenizer, model, device = cached_model()
            rewritten = generate_rewrite(rough_email, tokenizer, model, device)
        st.text_area("Professional email", value=rewritten, height=180)
    else:
        st.warning("Enter an email to rewrite.")
