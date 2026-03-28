"""Streamlit Chat Interface - Memory-Aware Regulatory Agent.

Prerequisites:
    Phoenix running in a separate terminal: python -m phoenix.server.main serve

Run with: streamlit run app/main.py
"""

from __future__ import annotations

import html
import sys
import uuid
from pathlib import Path

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import run_agent  # noqa: E402

# -- Phoenix OTEL tracing (must run before any traced code) ----------------
try:
    from phoenix.otel import register

    register(
        project_name="memory-agent-regulatory",
        endpoint="http://localhost:6006/v1/traces",
    )
except ImportError:
    pass
except Exception:
    pass

from src.observability.tracing import span_set_input, span_set_output, trace_span  # noqa: E402

# -- Page config -----------------------------------------------------------

st.set_page_config(
    page_title="Agente Regulatorio PIX",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    .main .block-container { max-width: 900px; padding-top: 2rem; }

    h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; }

    .citation-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #15803d;
        border-radius: 4px;
        padding: 0.1rem 0.45rem;
        margin: 0.15rem 0.15rem 0.15rem 0;
        font-weight: 500;
    }
    .citation-footer {
        margin-top: 0.6rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e2e8f0;
    }
    .citation-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    .meta-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- Session state ---------------------------------------------------------


def _init_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "citations" not in st.session_state:
        st.session_state.citations = {}  # msg_index -> list of citation strings
    if "memory_log" not in st.session_state:
        st.session_state.memory_log = []
    if "use_memory" not in st.session_state:
        st.session_state.use_memory = False
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}


_init_session_state()

# -- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Configuracoes")

    st.session_state.use_memory = st.toggle(
        "Ativar Memoria", value=st.session_state.use_memory
    )
    st.caption(
        "Quando ativada, o agente utiliza memoria conversacional "
        "via PostgreSQL + Weaviate. Requer Docker."
    )

    st.divider()

    if st.button("Nova Conversa"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.citations = {}
        st.session_state.memory_log = []
        st.session_state.feedback = {}
        st.rerun()

    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")

    st.divider()
    st.header("Atividade de Memoria")

    if not st.session_state.memory_log:
        st.info("Nenhuma operacao de memoria ainda. Inicie a conversa.")
    else:
        for i, entry in enumerate(reversed(st.session_state.memory_log)):
            turn_num = len(st.session_state.memory_log) - i
            with st.expander(f"Turno {turn_num}", expanded=(i == 0)):
                if entry.get("read"):
                    st.markdown("**Leitura de Memoria**")
                    for r in entry["read"]:
                        st.markdown(
                            f"- **{r.get('type', '')}**: "
                            f"{r.get('snippet', 'N/A')}"
                        )
                if entry.get("write"):
                    st.markdown("**Escrita de Memoria**")
                    for w in entry["write"]:
                        st.markdown(
                            f"- **{w.get('type', '')}**: "
                            f"{w.get('snippet', 'N/A')}"
                        )

    st.divider()
    st.markdown(
        "[Painel de Avaliacao](evaluation) | "
        "[Explorador de Memoria](memory_explorer)"
    )


# -- Dependencies ----------------------------------------------------------


def _build_deps() -> dict:
    """Build dependency dict for the agent graph."""
    deps: dict = {}

    try:
        import ollama

        from src.config.settings import get_settings

        settings = get_settings()
        model_name = settings.llm.model

        def llm_fn(messages: list[dict]) -> str:
            resp = ollama.chat(model=model_name, messages=messages)
            return resp["message"]["content"]

        deps["llm_fn"] = llm_fn
    except Exception:
        deps["llm_fn"] = lambda msgs: (
            "Ollama nao disponivel. Instale com: pip install ollama"
        )

    # Build context function with retrieval + citation extraction
    try:
        from src.rag.context_builder import build_citations, build_context
        from src.retrieval.retriever import retrieve

        def build_context_fn(memory_context=None, **kwargs) -> str:
            query = kwargs.get("query", "")
            chunks = []
            if query:
                try:
                    results = retrieve(query, top_k=5)
                    chunks = results
                except Exception:
                    pass

            # Store citations for later UI rendering
            if chunks:
                st.session_state["_last_citations"] = build_citations(chunks)
                st.session_state["_last_chunks"] = chunks
            else:
                st.session_state["_last_citations"] = ""
                st.session_state["_last_chunks"] = []

            return build_context(chunks=chunks, memory_context=memory_context)

        deps["build_context_fn"] = build_context_fn
    except ImportError:
        deps["build_context_fn"] = lambda **kw: ""

    if st.session_state.use_memory:
        try:
            from src.memory.database import get_session
            from src.memory.manager import MemoryManager

            session = get_session()
            import weaviate

            wc = weaviate.connect_to_local()
            mm = MemoryManager(db_session=session, weaviate_client=wc)
            deps["memory_manager"] = mm

            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer("BAAI/bge-m3")
            deps["embed_fn"] = lambda text: _model.encode(text).tolist()
        except Exception as e:
            st.sidebar.warning(f"Memoria indisponivel: {e}")

    return deps


# -- Query execution -------------------------------------------------------


def _run_query(query: str) -> str:
    """Run the agent wrapped in a parent trace span."""
    deps = _build_deps()

    with trace_span(
        "agent_pipeline",
        attributes={
            "thread_id": st.session_state.thread_id,
            "query": query,
        },
        openinference_span_kind="CHAIN",
    ) as span:
        if span and span.is_recording():
            span_set_input(span, query)

        response = run_agent(
            query=query,
            thread_id=st.session_state.thread_id,
            deps=deps,
        )

        if span and span.is_recording():
            span_set_output(span, response or "")

    return response or "(Nenhuma resposta gerada)"


def _save_feedback(msg_index: int, rating: int) -> None:
    """Persist feedback to PostgreSQL."""
    st.session_state.feedback[msg_index] = rating
    messages = st.session_state.messages
    query = ""
    for i in range(msg_index - 1, -1, -1):
        if messages[i]["role"] == "user":
            query = messages[i]["content"]
            break
    response = messages[msg_index]["content"]
    try:
        from src.memory.database import get_session
        from src.memory.models import ResponseFeedback

        session = get_session()
        fb = ResponseFeedback(
            thread_id=st.session_state.thread_id,
            query=query,
            response=response,
            rating=rating,
        )
        session.add(fb)
        session.commit()
    except Exception:
        pass


def _render_citations_html(citation_text: str) -> str:
    """Convert citation footer text to HTML badges."""
    if not citation_text:
        return ""
    # Extract citations between "Fontes consultadas: " and the final period
    import re

    match = re.search(r"Fontes consultadas:\s*(.+?)\.\s*Para", citation_text)
    if not match:
        return ""
    raw = match.group(1)
    parts = [c.strip() for c in raw.split(";") if c.strip()]
    badges = "".join(
        f'<span class="citation-badge">{html.escape(p)}</span>' for p in parts
    )
    return (
        f'<div class="citation-footer">'
        f'<div class="citation-label">Fontes consultadas</div>'
        f"{badges}</div>"
    )


# -- Main Chat UI ---------------------------------------------------------

st.title("Agente Regulatorio PIX")
st.caption(
    "Pergunte sobre regulamentacao PIX: limites, prazos, "
    "disputas, MED e normas do Banco Central."
)

# Display conversation history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show citations for assistant messages
        if msg["role"] == "assistant":
            cit = st.session_state.citations.get(i, "")
            if cit:
                st.markdown(_render_citations_html(cit), unsafe_allow_html=True)
            # Feedback
            prev_rating = st.session_state.feedback.get(i)
            rating = st.feedback(
                "thumbs",
                key=f"fb_{i}",
                disabled=prev_rating is not None,
            )
            if rating is not None and prev_rating is None:
                _save_feedback(i, rating)

# Chat input
if prompt := st.chat_input("Faca sua pergunta sobre regulamentacao PIX..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Reset temp citation storage
    st.session_state["_last_citations"] = ""
    st.session_state["_last_chunks"] = []

    with st.chat_message("assistant"):
        with st.spinner("Consultando documentos..."):
            response = _run_query(prompt)
        st.markdown(response)

        # Show citations
        cit_text = st.session_state.get("_last_citations", "")
        if cit_text:
            st.markdown(_render_citations_html(cit_text), unsafe_allow_html=True)

    # Store message and citations
    msg_idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    if cit_text:
        st.session_state.citations[msg_idx] = cit_text

    # Log memory activity
    memory_entry = {"read": [], "write": []}
    if st.session_state.use_memory:
        memory_entry["read"].append(
            {"type": "conversacional", "snippet": "Historico carregado"}
        )
        memory_entry["write"].append(
            {"type": "conversacional", "snippet": prompt[:50]}
        )
    st.session_state.memory_log.append(memory_entry)
    st.rerun()
