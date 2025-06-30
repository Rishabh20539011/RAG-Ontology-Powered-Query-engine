from __future__ import annotations

"""Streamlit front‑end for the RAG‑powered ontology explorer.

*Note- Make sure to update all the secrets in **config.yaml** to avoid any error in UI.
"""

import logging
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

import streamlit as st
from llama_index.core import SimpleDirectoryReader, Document

from ontology_builder import build_ontology
from config_loader import CONFIG

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _save_uploaded_files(files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> Path:
    upload_dir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
    st.info(f"Saving {len(files)} file(s) to **{upload_dir}** …")
    for f in files:
        dest = upload_dir / f.name
        dest.write_bytes(f.getbuffer())
        log.info("Saved %s", dest)
    return upload_dir


def _extract_zip_to_tmp(zip_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    zip_tmp = Path(tempfile.mkdtemp()) / f"{uuid4().hex}.zip"
    zip_tmp.write_bytes(zip_file.getbuffer())
    extract_dir = zip_tmp.parent / f"unzipped_{uuid4().hex}"
    st.info(f"Extracting **{zip_file.name}** …")
    with zipfile.ZipFile(zip_tmp, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _load_files_into_llama(path: Path) -> list[Document]:
    reader = SimpleDirectoryReader(input_dir=str(path), recursive=True)
    return reader.load_data()

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RAG Ontology Explorer", layout="wide")
page = st.session_state.get("page", "upload")

# Ensure the Groq API key exists in config
if not CONFIG["groq"].get("api_key"):
    st.error(
        "Groq API key not found in *config.yaml*. Please add it there before running the app."
    )
    st.stop()

if page == "upload":
    st.title("RAG‑powered Ontology Builder 🧠➡️📊")
    st.markdown("Upload one or more text / markdown / PDF files **or** a ZIP archive.")

    uploaded_files = st.file_uploader(
        "Choose file(s)",
        accept_multiple_files=True,
        type=["txt", "md", "pdf", "zip"],
    )

    if st.button("Build Ontology") and uploaded_files:
        project_id = uuid4().hex  # unique per build
        st.session_state.project_id = project_id

        with st.spinner("Preparing input files …"):
            if len(uploaded_files) == 1 and uploaded_files[0].name.endswith(".zip"):
                source_dir = _extract_zip_to_tmp(uploaded_files[0])
            else:
                source_dir = _save_uploaded_files(uploaded_files)
            docs = _load_files_into_llama(source_dir)

        # Build ontology – *api_key* read internally from CONFIG
        html_path, qe, retriever = build_ontology(
            docs,
            st_module=st,
            project_id=project_id,
        )

        st.session_state.ontology_html_path = str(html_path)
        st.session_state.query_engine = qe
        st.session_state.retriever = retriever
        st.session_state.page = "ontology"
        st.rerun()

# ---------------------------------------------------------------------------
# Page 2 – View & Query
# ---------------------------------------------------------------------------

if page == "ontology":
    st.title("Ontology View & Query")

    html_path = st.session_state.get("ontology_html_path")
    if not html_path:
        st.error("Ontology not found – please rebuild.")
        st.stop()

    with open(html_path, "r", encoding="utf‑8") as fh:
        st.components.v1.html(fh.read(), height=600, scrolling=True)

    st.markdown("---")
    st.header("Ask a question about your knowledge graph:")

    query = st.text_input("Your question")
    if st.button("Run Query") and query:
        qe = st.session_state.get("query_engine")
        retriever = st.session_state.get("retriever")
        if qe is None or retriever is None:
            st.error("Query utilities not available. Please rebuild the ontology.")
            st.stop()

        with st.spinner("Thinking …"):
            response = qe.query(query)
            nodes = retriever.retrieve(query)

        st.markdown(f"**Answer:** {response.response}")

        if nodes:
            with st.expander("Show retrieved graph nodes"):
                for node in nodes:
                    st.code(node.text, language="text")

    if st.button("🔄 Start over"):
        st.session_state.clear()
        st.rerun()