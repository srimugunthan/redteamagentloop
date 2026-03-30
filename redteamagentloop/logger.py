"""JSON-structured session logger for RedTeamAgentLoop.

Each session writes to ``reports/logs/{session_id}.log``.
Every log record is a single JSON object on one line (NDJSON format), so logs
can be parsed with ``jq`` or streamed into any log aggregator.

Usage::

    log = get_session_logger(session_id)
    log.debug("node started", extra={"node": "attacker", "iteration": 3, "session_id": session_id})
    log.error("LLM call failed", exc_info=True, extra={"node": "target_caller", "session_id": session_id})
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects (NDJSON)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Include optional structured fields if present on the record.
        for field in ("node", "iteration", "session_id"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["traceback"] = traceback.format_exception(*record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LOG_DIR_DEFAULT = "reports/logs"


def get_session_logger(
    session_id: str,
    log_dir: str = _LOG_DIR_DEFAULT,
) -> logging.Logger:
    """Return a ``logging.Logger`` that writes JSON to *log_dir*/{session_id}.log.

    Calling this multiple times with the same *session_id* returns the same
    logger instance without adding duplicate handlers.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{session_id}.log"

    logger_name = f"redteam.{session_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't bubble up to root logger

    if not logger.handlers:
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)

    return logger
