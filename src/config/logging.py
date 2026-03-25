"""Centralized logging configuration with optional JSON structured output.

Usage in scripts::

    from src.config.logging import setup_logging
    setup_logging()

The log format and level are read from the centralized Pydantic Settings
(config.yaml / env vars LOG_LEVEL, LOG_FORMAT).
"""

import logging
import sys

_configured = False


class _OTelCorrelationFilter(logging.Filter):
    """Inject OpenTelemetry trace_id and span_id into log records.

    When tracing is active, these fields appear in JSON logs so you can
    correlate log lines with Phoenix traces.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.trace_id:
                record.trace_id = format(ctx.trace_id, "032x")  # type: ignore[attr-defined]
                record.span_id = format(ctx.span_id, "016x")  # type: ignore[attr-defined]
            else:
                record.trace_id = ""  # type: ignore[attr-defined]
                record.span_id = ""  # type: ignore[attr-defined]
        except (ImportError, Exception):
            record.trace_id = ""  # type: ignore[attr-defined]
            record.span_id = ""  # type: ignore[attr-defined]
        return True


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure the root logger once.

    Parameters
    ----------
    level : str, optional
        Override log level. If None, reads from settings (LOG_LEVEL env or config.yaml).
    fmt : str, optional
        Override format ("text" or "json"). If None, reads from settings.
    """
    global _configured
    if _configured:
        return
    _configured = True

    from src.config import get_settings

    settings = get_settings()
    log_level = (level or settings.logging.level).upper()
    log_format = fmt or settings.logging.format

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)

    if log_format == "json":
        from pythonjsonlogger.json import JsonFormatter

        formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(_OTelCorrelationFilter())
    else:
        formatter = logging.Formatter(
            fmt="%(levelname)s | %(name)s | %(message)s",
        )
        handler.setFormatter(formatter)

    root.addHandler(handler)


def reset_logging() -> None:
    """Reset logging state. Only intended for tests."""
    global _configured
    _configured = False
    root = logging.getLogger()
    root.handlers.clear()
