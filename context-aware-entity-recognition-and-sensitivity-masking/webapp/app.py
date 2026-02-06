from __future__ import annotations

import io
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from src.ner_masker.inference import NERMasker
from src.ner_masker.utils import highlight_spans


@st.cache_resource
def load_masker(model_dir: str) -> NERMasker:
    return NERMasker(model_dir)


def main():
    st.set_page_config(page_title="Context-Aware NER & Masking", layout="wide")
    st.title("Context-Aware Entity Recognition and Sensitivity Masking")
    st.caption("Fine-tuned transformer for BIO tagging and masking")

    with st.sidebar:
        st.header("Settings")
        model_dir = st.text_input("Model directory", value="dslim/bert-base-NER")
        mask_token = st.text_input("Mask token", value="[MASK]")
        if st.button("Load / Reload Model"):
            st.cache_resource.clear()
        st.divider()
        st.markdown("Upload .txt files for batch processing below")
        uploaded = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

    # Try to load model; if it fails, show an actionable error. Avoid warning for hub model IDs.
    try:
        masker = load_masker(model_dir)
    except Exception as e:
        st.error(f"Failed to load model '{model_dir}'. If this is a local path, ensure it exists and contains a saved tokenizer and model. If it's a hub ID, check network access.\n\nDetails: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["Single Text", "Batch Processing"])

    with tab1:
        default_text = (
            "Technical support is available via UID: 8829-X or help.desk@corp.net. "
            "Visit our portal at https://user-profile.com/john_doe."
        )
        text = st.text_area("Enter text", value=default_text, height=160)
        if st.button("Analyze & Mask", key="analyze_single"):
            spans = masker.predict_entities(text)
            masked = masker.mask_text(text, mask_token=mask_token)
            st.subheader("Masked Output")
            st.code(masked)
            st.subheader("Highlights")
            st.markdown(highlight_spans(text, spans), unsafe_allow_html=True)

    with tab2:
        if uploaded:
            rows = []
            for uf in uploaded:
                content = uf.getvalue().decode("utf-8", errors="ignore")
                masked = masker.mask_text(content, mask_token=mask_token)
                rows.append({"filename": uf.name, "masked": masked})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="masked_outputs.csv", mime="text/csv")


if __name__ == "__main__":
    main()
