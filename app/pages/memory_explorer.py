"""Memory Explorer - browse stored memories by type and thread."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Memory Explorer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Memory Explorer")
st.caption("Browse stored memories by type and thread")

# ── Connection state ─────────────────────────────────────────────────────────

MEMORY_TYPES = ["conversational", "semantic", "procedural", "summary"]

selected_type = st.selectbox("Memory Type", MEMORY_TYPES)

thread_filter = st.text_input(
    "Filter by Thread ID (optional)", placeholder="e.g. abc123..."
)

st.divider()

# ── Data loading ─────────────────────────────────────────────────────────────


def _load_conversational(thread_filter: str) -> list[dict]:
    """Load conversational memory from PostgreSQL."""
    try:
        from src.memory.conversational import get_history
        from src.memory.database import get_session

        session = get_session()
        if not thread_filter:
            st.info(
                "Enter a thread ID to load conversational memory."
            )
            return []
        turns = get_history(session, thread_filter, limit=50)
        return [
            {
                "id": str(t.id),
                "thread_id": t.thread_id,
                "role": t.role,
                "content": t.content[:200],
                "timestamp": str(t.timestamp),
            }
            for t in turns
        ]
    except Exception as e:
        st.error(f"Cannot connect to PostgreSQL: {e}")
        return []


def _load_semantic(thread_filter: str) -> list[dict]:
    """Load semantic memories from Weaviate."""
    try:
        import weaviate

        client = weaviate.connect_to_local()
        collection = client.collections.get("SemanticMemory")
        response = collection.query.fetch_objects(limit=50)
        results = []
        for obj in response.objects:
            props = obj.properties
            tid = props.get("thread_id", "")
            if thread_filter and thread_filter not in tid:
                continue
            results.append(
                {
                    "id": str(obj.uuid),
                    "content": str(props.get("content", ""))[:200],
                    "source": props.get("source", ""),
                    "thread_id": tid,
                    "timestamp": props.get("timestamp", ""),
                }
            )
        client.close()
        return results
    except Exception as e:
        st.error(f"Cannot connect to Weaviate: {e}")
        return []


def _load_procedural(_thread_filter: str) -> list[dict]:
    """Load procedural patterns from Weaviate."""
    try:
        import weaviate

        client = weaviate.connect_to_local()
        collection = client.collections.get("ProceduralMemory")
        response = collection.query.fetch_objects(limit=50)
        results = [
            {
                "id": str(obj.uuid),
                "trigger": str(
                    obj.properties.get("trigger", "")
                )[:200],
                "action": str(
                    obj.properties.get("action", "")
                )[:200],
                "timestamp": obj.properties.get("timestamp", ""),
            }
            for obj in response.objects
        ]
        client.close()
        return results
    except Exception as e:
        st.error(f"Cannot connect to Weaviate: {e}")
        return []


def _load_summary(thread_filter: str) -> list[dict]:
    """Load session summaries from PostgreSQL."""
    try:
        from src.memory.database import get_session
        from src.memory.summary import get_summary

        session = get_session()
        if not thread_filter:
            st.info("Enter a thread ID to load summaries.")
            return []
        summary = get_summary(session, thread_filter)
        if summary:
            return [
                {
                    "thread_id": thread_filter,
                    "summary": summary[:500],
                }
            ]
        return []
    except Exception as e:
        st.error(f"Cannot connect to PostgreSQL: {e}")
        return []


# ── Display ──────────────────────────────────────────────────────────────────

LOADERS = {
    "conversational": _load_conversational,
    "semantic": _load_semantic,
    "procedural": _load_procedural,
    "summary": _load_summary,
}

loader = LOADERS[selected_type]
data = loader(thread_filter)

if data:
    import pandas as pd

    st.success(f"Found {len(data)} records")
    st.dataframe(pd.DataFrame(data), use_container_width=True)
else:
    st.info(
        "No records found. Make sure Docker services are running "
        "and you have stored memories via the chat interface."
    )

# ── Info ─────────────────────────────────────────────────────────────────────

with st.expander("Memory Type Reference"):
    st.markdown(
        """
| Type | Storage | Description |
|------|---------|-------------|
| **Conversational** | PostgreSQL | Turn-by-turn chat history per thread |
| **Semantic** | Weaviate | Vector-indexed content for similarity search |
| **Procedural** | Weaviate | Trigger-action patterns learned over time |
| **Summary** | PostgreSQL | Compressed session summaries |
"""
    )
