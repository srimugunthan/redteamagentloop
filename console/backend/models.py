"""Pydantic models shared between the backend API and SSE stream."""

from __future__ import annotations

from pydantic import BaseModel


class SessionRequest(BaseModel):
    objective: str
    system_prompt: str = ""
    target_tag: str | None = None


class SessionResponse(BaseModel):
    session_id: str


class AttackEvent(BaseModel):
    """A single typed event emitted over the SSE stream.

    The ``event_type`` field drives the frontend state machine:

    iteration_start → prompt_ready → response_ready
        → score_delta (×N, for gauge animation) → score_ready
        → vuln_logged (only if score >= threshold)
        → [repeat per iteration]
        → session_end
    """
    event_type: str          # see docstring above
    session_id: str
    iteration: int | None = None
    strategy: str | None = None
    prompt: str | None = None
    response: str | None = None
    score: float | None = None
    score_delta: float | None = None   # only on score_delta events
    violated_policy: str | None = None
    error: str | None = None
