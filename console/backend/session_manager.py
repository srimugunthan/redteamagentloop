"""SessionManager — runs graph in background and exposes events via AsyncGenerator.

One Queue per session.  The background task puts AttackEvents into the queue
as the graph streams; the SSE endpoint drains the queue and sends them to the
browser.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from console.backend.event_mapper import EventMapper
from console.backend.models import AttackEvent

if TYPE_CHECKING:
    from redteamagentloop.config import AppConfig

_QUEUE_MAX = 500          # back-pressure limit
_DRAIN_TIMEOUT = 60.0     # seconds to wait for next event before continuing


class SessionManager:
    """Module-level singleton — one instance shared across the FastAPI app."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._mappers: dict[str, EventMapper] = {}

    async def create_session(
        self,
        objective: str,
        app_config: "AppConfig",
        system_prompt: str = "",
        target_tag: str | None = None,
    ) -> str:
        """Build the graph, launch it as a background task, return session_id."""
        session_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        mapper = EventMapper(session_id)

        self._queues[session_id] = queue
        self._mappers[session_id] = mapper
        self._tasks[session_id] = asyncio.create_task(
            self._run_graph(session_id, objective, system_prompt, app_config, target_tag, queue, mapper),
            name=f"session-{session_id[:8]}",
        )
        return session_id

    async def _run_graph(
        self,
        session_id: str,
        objective: str,
        system_prompt: str,
        app_config: "AppConfig",
        target_tag: str | None,
        queue: asyncio.Queue,
        mapper: EventMapper,
    ) -> None:
        from redteamagentloop.agent.graph import build_graph, build_initial_state

        graph = build_graph(app_config)

        # Pick the target; fall back to first if tag not found.
        target = app_config.targets[0]
        if target_tag:
            for t in app_config.targets:
                if t.output_tag == target_tag:
                    target = t
                    break

        initial = build_initial_state(
            app_config,
            target_objective=objective,
            target_system_prompt=system_prompt or target.system_prompt,
        )

        try:
            async for raw_event in graph.astream_events(
                initial,
                config={"configurable": {"app_config": app_config}},
                version="v1",
            ):
                for attack_event in mapper.map(raw_event):
                    await queue.put(attack_event)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await queue.put(AttackEvent(
                event_type="session_end",
                session_id=session_id,
                iteration=mapper._iteration,
                error=str(exc),
            ))
        finally:
            # Always emit a definitive session_end followed by the queue sentinel.
            await queue.put(AttackEvent(
                event_type="session_end",
                session_id=session_id,
                iteration=mapper._iteration,
            ))
            await queue.put(None)  # signals get_stream to stop iterating

    async def get_stream(self, session_id: str) -> AsyncGenerator[AttackEvent, None]:
        """Yield AttackEvents for *session_id* until the session ends."""
        queue = self._queues.get(session_id)
        if queue is None:
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_DRAIN_TIMEOUT)
            except asyncio.TimeoutError:
                # Keep the SSE connection alive with a keepalive comment.
                continue
            if event is None:
                self._cleanup(session_id)
                break
            yield event

    def terminate_session(self, session_id: str) -> None:
        """Cancel the background task and clean up resources."""
        if task := self._tasks.get(session_id):
            task.cancel()
        self._cleanup(session_id)

    def _cleanup(self, session_id: str) -> None:
        self._queues.pop(session_id, None)
        self._tasks.pop(session_id, None)
        self._mappers.pop(session_id, None)


# Module-level singleton consumed by main.py
session_manager = SessionManager()
