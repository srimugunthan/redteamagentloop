"""Maps raw LangGraph astream_events output to typed AttackEvents.

LangGraph emits low-level events for every node start/end plus internal
chain bookkeeping.  This module translates the subset we care about into
the AttackEvent schema consumed by the frontend, and discards the rest.

Key mappings
------------
on_chain_start / attacker        → iteration_start
on_chain_end   / attacker        → prompt_ready
on_chain_end   / target_caller   → response_ready
on_chain_end   / judge           → [score_delta × N] + score_ready
on_chain_end   / vuln_logger     → vuln_logged
on_chain_end   / LangGraph       → session_end  (top-level graph)
"""

from __future__ import annotations

from console.backend.models import AttackEvent

_SCORE_DELTA_STEP = 0.5
_GRAPH_NAMES = {"LangGraph", ""}   # LangGraph names the top-level chain "LangGraph"


class EventMapper:
    """Stateful mapper for one session.  Tracks iteration number across events."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._iteration: int = 0
        self._strategy: str = ""
        self._prompt: str = ""

    def map(self, langgraph_event: dict) -> list[AttackEvent]:
        """Return zero or more AttackEvents for a single LangGraph event."""
        ev = langgraph_event.get("event", "")
        name = langgraph_event.get("name", "")
        data = langgraph_event.get("data", {})

        # --- attacker node -------------------------------------------------
        if ev == "on_chain_start" and name == "attacker":
            # self._iteration still holds the PREVIOUS iteration count here.
            # The attacker node will increment it to self._iteration + 1 on
            # on_chain_end, so we emit the upcoming number so the frontend
            # creates the blank row with the right key before prompt_ready arrives.
            return [AttackEvent(
                event_type="iteration_start",
                session_id=self.session_id,
                iteration=self._iteration + 1,
            )]

        if ev == "on_chain_end" and name == "attacker":
            output = data.get("output") or {}
            self._iteration = output.get("iteration_count", self._iteration)
            self._strategy = output.get("current_strategy", "")
            self._prompt = output.get("current_prompt", "")
            error = output.get("error")
            return [AttackEvent(
                event_type="prompt_ready",
                session_id=self.session_id,
                iteration=self._iteration,
                strategy=self._strategy,
                prompt=self._prompt,
                error=error,
            )]

        # --- target_caller node --------------------------------------------
        if ev == "on_chain_end" and name == "target_caller":
            output = data.get("output") or {}
            return [AttackEvent(
                event_type="response_ready",
                session_id=self.session_id,
                iteration=self._iteration,
                response=output.get("current_response", ""),
                error=output.get("error"),
            )]

        # --- judge node ----------------------------------------------------
        if ev == "on_chain_end" and name == "judge":
            output = data.get("output") or {}
            score = float(output.get("score", 0.0))
            events: list[AttackEvent] = []

            # Emit incremental score_delta events to animate the gauge.
            current = 0.0
            while current + _SCORE_DELTA_STEP <= score:
                current = round(current + _SCORE_DELTA_STEP, 1)
                events.append(AttackEvent(
                    event_type="score_delta",
                    session_id=self.session_id,
                    iteration=self._iteration,
                    score=current,
                    score_delta=_SCORE_DELTA_STEP,
                ))

            # Always emit the final score_ready with the exact value.
            events.append(AttackEvent(
                event_type="score_ready",
                session_id=self.session_id,
                iteration=self._iteration,
                score=score,
            ))
            return events

        # --- vuln_logger node ---------------------------------------------
        if ev == "on_chain_end" and name == "vuln_logger":
            output = data.get("output") or {}
            new_attacks = output.get("successful_attacks", [])
            if new_attacks:
                latest = new_attacks[-1]
                return [AttackEvent(
                    event_type="vuln_logged",
                    session_id=self.session_id,
                    iteration=self._iteration,
                    prompt=latest.get("prompt", ""),
                    response=latest.get("response", ""),
                    score=latest.get("score", 0.0),
                    violated_policy=latest.get("score_rationale", ""),
                )]
            return []

        # --- top-level graph end ------------------------------------------
        if ev == "on_chain_end" and name in _GRAPH_NAMES:
            output = data.get("output") or {}
            return [AttackEvent(
                event_type="session_end",
                session_id=self.session_id,
                iteration=output.get("iteration_count", self._iteration),
            )]

        # Everything else (on_llm_start, on_chat_model_start, etc.) → skip
        return []
