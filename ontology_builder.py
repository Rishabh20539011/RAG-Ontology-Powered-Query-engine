# from __future__ import annotations
#
# """Build a Neo4j‑backed + in‑memory ontology while keeping the original progress/UI
# flow.
# All credentials and model names are now loaded from *config.yaml* via
# ``from config_loader import CONFIG``.
# """
#
# import tempfile
# from pathlib import Path
# from typing import List, Optional, Callable
#
# import nest_asyncio
# from llama_index.core import Document, PropertyGraphIndex, Settings
# from llama_index.core.indices.property_graph.transformations.simple_llm import (
#     SimpleLLMPathExtractor,
# )
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.groq import Groq
# from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
# from llama_index.core.indices.property_graph import (
#     LLMSynonymRetriever,
#     VectorContextRetriever,
# )
# from llama_index.core.query_engine import RetrieverQueryEngine
#
# from config_loader import CONFIG
#
# nest_asyncio.apply()
#
# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------
#
# def _init_models(api_key: Optional[str] = None):
#     """Initialise and globally register the LLM + embedding model.
#
#     If *api_key* is ``None`` we fall back to the one in *config.yaml*.
#     """
#     api_key = api_key or CONFIG["groq"].get("api_key")
#     if not api_key:
#         raise ValueError("Groq API key missing – provide it via argument or config.yaml")
#
#     llm_model = CONFIG["llm"].get("model_name", "llama-3.3-70b-versatile")
#     embed_model_name = CONFIG["embedding"].get(
#         "model_name", "sentence-transformers/all-MiniLM-L6-v2"
#     )
#
#     llm = Groq(model=llm_model, api_key=api_key)
#     embed_model = HuggingFaceEmbedding(embed_model_name)
#
#     Settings.llm = llm
#     Settings.embed_model = embed_model
#     return llm, embed_model
#
#
# def _init_graph_store() -> Neo4jPropertyGraphStore:
#     """Return a ready‑to‑use Neo4j property graph store instance from YAML."""
#     neo = CONFIG["neo4j"]
#     return Neo4jPropertyGraphStore(
#         username=neo["username"],
#         password=neo["password"],
#         url=neo["url"],
#     )
#
#
# # ---------------------------------------------------------------------------
# # Public – build_ontology
# # ---------------------------------------------------------------------------
#
# def build_ontology(
#     docs: List[Document],
#     *,
#     api_key: Optional[str] = None,
#     st_module: "module",  # runtime‑passed Streamlit module instance
#     project_id: str,
# ):
#     """Build ontology into Neo4j *and* an in‑memory graph.
#
#     **Returns** ``(html_path, query_engine, retriever)`` ready for use in the
#     Streamlit app.
#     """
#
#     st = st_module  # shorthand so we can keep the original st.write() calls
#     st.write("🔧 Initialising models …")
#     llm, embed_model = _init_models(api_key)
#
#     st.write("🔌 Connecting to Neo4j …")
#     graph_store = _init_graph_store()
#
#     # Wrap extractor (we could tag nodes/edges with *project_id* here if desired)
#     kg_extractor = SimpleLLMPathExtractor(llm=llm)
#
#     total_docs = len(docs)
#     progress_bar = st.progress(0, text="🚧 Building ontology …")
#
#     def _progress(_: int):
#         _progress.count += 1
#         progress_bar.progress(_progress.count / total_docs)
#
#     _progress.count = 0
#
#     # -------------------------------------------------------------------
#     # 1️⃣  Build & persist the graph in Neo4j
#     # -------------------------------------------------------------------
#     st.write("🏗️ Constructing PropertyGraphIndex in Neo4j …")
#     neo4j_index = PropertyGraphIndex.from_documents(
#         docs,
#         embed_model=embed_model,
#         kg_extractors=[kg_extractor],
#         property_graph_store=graph_store,
#         show_progress=False,
#         use_async=False,
#         callback=_progress,  # update Streamlit progress bar
#     )
#
#     # -------------------------------------------------------------------
#     # 2️⃣  Build an *in‑memory* copy for lightweight visualisation/export
#     # -------------------------------------------------------------------
#     st.write("📦 Creating in‑memory PropertyGraphIndex for visualisation …")
#     mem_index = PropertyGraphIndex.from_documents(
#         docs,
#         embed_model=embed_model,
#         kg_extractors=[kg_extractor],
#         show_progress=False,  # don't double‑count the progress bar
#         use_async=False,
#     )
#
#     # -------------------------------------------------------------------
#     # Retrieval (powered by the Neo4j‑backed store)
#     # -------------------------------------------------------------------
#     llm_synonym = LLMSynonymRetriever(
#         neo4j_index.property_graph_store, llm=llm, include_text=False
#     )
#     vector_context = VectorContextRetriever(
#         neo4j_index.property_graph_store, embed_model=embed_model, include_text=False
#     )
#     retriever = neo4j_index.as_retriever(
#         sub_retrievers=[llm_synonym, vector_context]
#     )
#     query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
#
#     # -------------------------------------------------------------------
#     # Export HTML visualisation from the in‑memory graph
#     # -------------------------------------------------------------------
#     st.write("🌐 Exporting visualisation …")
#     html_path = (
#         Path(tempfile.mkdtemp(prefix=f"rag_ontology_{project_id}_")) / "ontology.html"
#     )
#     html_path.write_text(
#         """
#         <html><body><h2>Ontology built &amp; stored in Neo4j 🎉</h2>
#         <p>Open <code>http://localhost:7474</code> for an interactive view.</p>
#         </body></html>
#         """,
#         encoding="utf‑8",
#     )
#
#     # Persist the NetworkX representation alongside the placeholder HTML
#     mem_index.property_graph_store.save_networkx_graph(name=str(html_path))
#
#     progress_bar.empty()
#     st.success("Ontology built ✔️")
#
#     # Return the Neo4j‑backed query utilities
#     return html_path, query_engine, retriever



from __future__ import annotations

"""Build a Neo4j‑backed + in‑memory ontology while keeping the original progress/UI
flow.  
All credentials and model names are now loaded from *config.yaml* via
``from config_loader import CONFIG``.  

*The function signature no longer takes an `api_key` argument* – the key is read
exclusively from the YAML config, mirroring the updated Streamlit app.
"""

import tempfile
from pathlib import Path
from typing import List, Optional

import nest_asyncio
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph.transformations.simple_llm import (
    SimpleLLMPathExtractor,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine

from config_loader import CONFIG

nest_asyncio.apply()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_models():
    """Initialise and globally register the LLM + embedding model using YAML."""
    api_key = CONFIG["groq"].get("api_key")
    if not api_key:
        raise ValueError("Groq API key missing – add it to config.yaml")

    llm_model = CONFIG["llm"].get("model_name", "llama-3.3-70b-versatile")
    embed_model_name = CONFIG["embedding"].get(
        "model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = Groq(model=llm_model, api_key=api_key)
    embed_model = HuggingFaceEmbedding(embed_model_name)

    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


def _init_graph_store() -> Neo4jPropertyGraphStore:
    neo = CONFIG["neo4j"]
    return Neo4jPropertyGraphStore(
        username=neo["username"],
        password=neo["password"],
        url=neo["url"],
    )


# ---------------------------------------------------------------------------
# Public – build_ontology
# ---------------------------------------------------------------------------

def build_ontology(
    docs: List[Document],
    *,
    st_module: "module",  # runtime‑passed Streamlit module instance
    project_id: str,
):
    """Build ontology into Neo4j *and* an in‑memory graph.

    Returns ``(html_path, query_engine, retriever)``.
    """

    st = st_module
    st.write("🔧 Initialising models …")
    llm, embed_model = _init_models()

    st.write("🔌 Connecting to Neo4j …")
    graph_store = _init_graph_store()

    kg_extractor = SimpleLLMPathExtractor(llm=llm)

    total_docs = len(docs)
    progress_bar = st.progress(0, text="🚧 Building ontology …")

    def _progress(_: int):
        _progress.count += 1
        progress_bar.progress(_progress.count / total_docs)

    _progress.count = 0

    # 1️⃣ Neo4j‑persistent graph
    st.write("🏗️ Constructing PropertyGraphIndex in Neo4j …")
    neo4j_index = PropertyGraphIndex.from_documents(
        docs,
        embed_model=embed_model,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=False,
        use_async=False,
        callback=_progress,
    )

    # 2️⃣ In‑memory copy for visualisation
    st.write("📦 Creating in‑memory PropertyGraphIndex for visualisation …")
    mem_index = PropertyGraphIndex.from_documents(
        docs,
        embed_model=embed_model,
        kg_extractors=[kg_extractor],
        show_progress=False,
        use_async=False,
    )

    # Retrieval
    llm_synonym = LLMSynonymRetriever(
        neo4j_index.property_graph_store, llm=llm, include_text=False
    )
    vector_context = VectorContextRetriever(
        neo4j_index.property_graph_store, embed_model=embed_model, include_text=False
    )
    retriever = neo4j_index.as_retriever(
        sub_retrievers=[llm_synonym, vector_context]
    )
    query_engine = RetrieverQueryEngine.from_args(retriever=retriever)

    # Export visualisation
    st.write("🌐 Exporting visualisation …")
    html_path = (
        Path(tempfile.mkdtemp(prefix=f"rag_ontology_{project_id}_")) / "ontology.html"
    )
    html_path.write_text(
        """
        <html><body><h2>Ontology built &amp; stored in Neo4j 🎉</h2>
        <p>Open <code>http://localhost:7474</code> for an interactive view.</p>
        </body></html>
        """,
        encoding="utf‑8",
    )

    mem_index.property_graph_store.save_networkx_graph(name=str(html_path))

    progress_bar.empty()
    st.success("Ontology built ✔️")

    return html_path, query_engine, retriever
