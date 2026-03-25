"""Phoenix tracing helper — isolates OpenTelemetry/Phoenix dependency."""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_tracer = None


def _get_tracer():
    """Lazy init tracer. Uses global provider (Phoenix if register() was called at startup)."""
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace

        _tracer = trace.get_tracer("memory-agent-regulatory", "0.1.0")
        return _tracer
    except ImportError:
        logger.debug("OpenTelemetry not available, tracing disabled")
        return None


def _truncate(s: str, max_len: int = 4000) -> str:
    """Truncate string for span attributes (OTEL has limits)."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "...[truncated]"


def span_set_input(span: Any, value: Any) -> None:
    """Set span input. Uses Phoenix set_input when available, else input.value attribute."""
    if span is None or not span.is_recording():
        return
    if isinstance(value, (dict, list)):
        value = json.dumps(value) if value else ""
    elif not isinstance(value, str):
        value = str(value)
    value = _truncate(value)
    set_input = getattr(span, "set_input", None)
    if callable(set_input):
        try:
            set_input(value)
        except Exception:
            span.set_attribute("input.value", value)
    else:
        span.set_attribute("input.value", value)


def span_set_output(span: Any, value: Any) -> None:
    """Set span output. Uses Phoenix set_output when available, else output.value attribute."""
    if span is None or not span.is_recording():
        return
    if isinstance(value, (dict, list)):
        value = json.dumps(value) if value else "{}"
    elif not isinstance(value, str):
        value = str(value)
    value = _truncate(value)
    set_output = getattr(span, "set_output", None)
    if callable(set_output):
        try:
            set_output(value)
        except Exception:
            span.set_attribute("output.value", value)
    else:
        span.set_attribute("output.value", value)


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    openinference_span_kind: str | None = None,
) -> Generator[Any, None, None]:
    """
    Context manager for a traced span.

    Yields the span so caller can set output attributes after the operation.
    If tracing is unavailable, yields None.

    Parameters
    ----------
    name : str
        Span name.
    attributes : dict, optional
        Initial span attributes.
    openinference_span_kind : str, optional
        OpenInference span kind (RETRIEVER, CHAIN, PROMPT, LLM, etc.) for Phoenix UI.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return

    attrs = dict(attributes or {})
    if openinference_span_kind and "openinference.span.kind" not in attrs:
        attrs["openinference.span.kind"] = openinference_span_kind.upper()

    span_ctx = tracer.start_as_current_span(name, attributes=attrs)

    with span_ctx as span:
        try:
            yield span
        except Exception as e:
            if span.is_recording():
                try:
                    from opentelemetry import trace as otel_trace

                    span.set_status(
                        otel_trace.Status(otel_trace.StatusCode.ERROR, str(e))
                    )
                except ImportError:
                    pass
            raise
