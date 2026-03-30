"""Target caller node — thin relay to the target LLM.

Phase 9 additions:
- Rate limiter (``target_rate_limiter`` in configurable)
- Circuit breaker: 5 consecutive errors → 60 s pause; 3 pauses → terminate
- Session-structured logging
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from rich.console import Console

from redteamagentloop.logger import get_session_logger

if TYPE_CHECKING:
    from redteamagentloop.agent.state import RedTeamState

# ---------------------------------------------------------------------------
# Circuit-breaker state — keyed by session_id so concurrent runs don't share.
# ---------------------------------------------------------------------------

_cb_state: dict[str, dict] = {}
# schema: {session_id: {"consecutive_errors": int, "pauses": int}}

_CB_ERROR_THRESHOLD = 5    # errors before a pause
_CB_PAUSE_SECONDS = 60     # seconds to pause when threshold hit
_CB_MAX_PAUSES = 3         # pauses before hard termination

_console = Console()


def _get_cb(session_id: str) -> dict:
    if session_id not in _cb_state:
        _cb_state[session_id] = {"consecutive_errors": 0, "pauses": 0}
    return _cb_state[session_id]


async def target_caller_node(state: "RedTeamState", config: RunnableConfig) -> dict:
    if state.get("error") is not None:
        return {}  # pass through upstream error (e.g. max_iterations_reached)

    if not state["current_prompt"]:
        return {"current_response": "", "error": "Empty prompt — nothing to send to target."}

    cfg = config.get("configurable", {})
    app_config = cfg.get("app_config")
    rate_limiter = cfg.get("target_rate_limiter")

    session_id = state["session_id"]
    log = get_session_logger(session_id)
    cb = _get_cb(session_id)

    target_llm = cfg.get("target_llm")
    if target_llm is None:
        tc = app_config.targets[0]
        target_llm = ChatOpenAI(
            model=tc.model,
            base_url=tc.base_url,
            api_key=tc.api_key,
            timeout=tc.timeout_seconds,
            temperature=0.0,
        )

    messages = [
        SystemMessage(content=state["target_system_prompt"]),
        HumanMessage(content=state["current_prompt"]),
    ]

    log.debug(
        "target_caller node started",
        extra={"node": "target_caller", "iteration": state["iteration_count"], "session_id": session_id},
    )

    # Rate-limit before calling the target LLM.
    if rate_limiter is not None:
        await rate_limiter.acquire()

    try:
        response = await target_llm.ainvoke(messages)
        # Success — reset consecutive error counter.
        cb["consecutive_errors"] = 0
        return {"current_response": response.content.strip(), "error": None}

    except httpx.TimeoutException as exc:
        error_msg = f"Target LLM timeout: {exc}"
    except Exception as exc:
        error_msg = f"Target LLM error: {exc}"

    # Increment error counter and check circuit breaker.
    cb["consecutive_errors"] += 1
    log.error(
        error_msg,
        extra={"node": "target_caller", "iteration": state["iteration_count"], "session_id": session_id},
    )

    if cb["consecutive_errors"] >= _CB_ERROR_THRESHOLD:
        cb["consecutive_errors"] = 0
        cb["pauses"] += 1

        if cb["pauses"] > _CB_MAX_PAUSES:
            return {
                "current_response": "",
                "error": f"Circuit breaker: {_CB_MAX_PAUSES} pauses exceeded — session terminated.",
            }

        _console.print(
            f"[yellow]Circuit breaker: {_CB_ERROR_THRESHOLD} consecutive target errors "
            f"(pause {cb['pauses']}/{_CB_MAX_PAUSES}). "
            f"Waiting {_CB_PAUSE_SECONDS}s before resuming…[/yellow]"
        )
        await asyncio.sleep(_CB_PAUSE_SECONDS)

    return {"current_response": "", "error": error_msg}
