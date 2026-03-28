"""Evaluation Dashboard - displays baseline vs memory-aware comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Evaluation Dashboard")
st.caption("Baseline (stateless) vs Memory-Aware pipeline comparison")

# ── Load results ─────────────────────────────────────────────────────────────

REPORT_JSON = Path("data/evaluation/report.json")
REPORT_MD = Path("data/evaluation/report.md")


@st.cache_data
def _load_json_report() -> dict | None:
    if REPORT_JSON.exists():
        return json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    return None


@st.cache_data
def _load_md_report() -> str | None:
    if REPORT_MD.exists():
        return REPORT_MD.read_text(encoding="utf-8")
    return None


data = _load_json_report()
md_report = _load_md_report()

if data is None:
    st.warning(
        "No evaluation report found. "
        "Run `python scripts/run_evaluation.py --generate` first."
    )
    st.stop()

# ── Overview metrics ─────────────────────────────────────────────────────────

st.header("Metric Overview")

comparison = data.get("comparison", {})

# Extract scalar metrics for columns
scalar_metrics = {}
latency_data = {}
for name, info in comparison.items():
    value = info.get("value", 0)
    if isinstance(value, dict):
        latency_data = value
    else:
        scalar_metrics[name] = value

if scalar_metrics:
    cols = st.columns(len(scalar_metrics))
    for col, (name, value) in zip(cols, scalar_metrics.items(), strict=False):
        label = name.replace("_", " ").title()
        delta = None
        delta_color = "normal"
        if name == "token_efficiency" and value != 0:
            pct = (value - 1.0) * 100
            delta = f"{pct:+.1f}%"
            delta_color = "normal" if pct >= 0 else "inverse"
        elif name == "context_reuse_rate":
            delta = f"{value * 100:.0f}%"
        col.metric(label=label, value=f"{value:.4f}", delta=delta)

# ── Latency comparison ───────────────────────────────────────────────────────

if latency_data:
    st.header("Latency Comparison")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Baseline Avg",
        f"{latency_data.get('baseline_avg', 0):.1f} ms",
    )
    col2.metric(
        "Memory Avg",
        f"{latency_data.get('memory_avg', 0):.1f} ms",
    )
    overhead = latency_data.get("overhead_ms", 0)
    col3.metric(
        "Overhead",
        f"{overhead:+.1f} ms",
        delta=f"{overhead:+.1f} ms",
        delta_color="inverse" if overhead > 0 else "normal",
    )

    # Bar chart
    import pandas as pd

    latency_df = pd.DataFrame(
        {
            "Metric": ["Avg", "P50", "P95"],
            "Baseline": [
                latency_data.get("baseline_avg", 0),
                latency_data.get("baseline_p50", 0),
                latency_data.get("baseline_p95", 0),
            ],
            "Memory": [
                latency_data.get("memory_avg", 0),
                latency_data.get("memory_p50", 0),
                latency_data.get("memory_p95", 0),
            ],
        }
    ).set_index("Metric")

    st.bar_chart(latency_df)

# ── Per-session details ──────────────────────────────────────────────────────

per_session = data.get("per_session", [])
if per_session:
    st.header("Per-Session Breakdown")

    import pandas as pd

    df = pd.DataFrame(per_session)

    # Token comparison chart
    if "baseline_tokens" in df.columns and "memory_tokens" in df.columns:
        st.subheader("Token Usage per Session")
        token_df = df[["session_id", "baseline_tokens", "memory_tokens"]].copy()
        token_df["session_id"] = token_df["session_id"].str[:8]
        token_df = token_df.set_index("session_id")
        st.bar_chart(token_df)

    # Full table
    st.subheader("Details")
    st.dataframe(df, use_container_width=True)

# ── Raw Markdown report ──────────────────────────────────────────────────────

with st.expander("View Raw Markdown Report"):
    if md_report:
        st.markdown(md_report)
    else:
        st.info("Markdown report not available.")

# ── Raw JSON ─────────────────────────────────────────────────────────────────

with st.expander("View Raw JSON Data"):
    st.json(data)
