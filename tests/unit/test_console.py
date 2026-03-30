"""Phase 10 — Unit tests for the Live Attack Console backend.

Tests:
- EventMapper: each node event type maps to correct AttackEvents
- EventMapper: unmapped events return empty list
- EventMapper: score_delta events are emitted for score animation
- SessionManager: create_session returns a session_id, get_stream yields events
- SessionManager: terminate_session cancels the task
- FastAPI endpoints: POST /api/sessions, GET /api/health, DELETE /api/sessions/{id}
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from console.backend.event_mapper import EventMapper
from console.backend.models import AttackEvent


# ===========================================================================
# EventMapper
# ===========================================================================

def make_mapper(session_id: str = "test-sess") -> EventMapper:
    return EventMapper(session_id)


class TestEventMapperIterationStart:
    def test_attacker_chain_start_emits_iteration_start(self):
        mapper = make_mapper()
        events = mapper.map({"event": "on_chain_start", "name": "attacker", "data": {}})
        assert len(events) == 1
        assert events[0].event_type == "iteration_start"
        assert events[0].session_id == "test-sess"

    def test_other_chain_start_returns_empty(self):
        mapper = make_mapper()
        assert mapper.map({"event": "on_chain_start", "name": "judge", "data": {}}) == []

    def test_llm_start_returns_empty(self):
        mapper = make_mapper()
        assert mapper.map({"event": "on_llm_start", "name": "ChatOpenAI", "data": {}}) == []


class TestEventMapperPromptReady:
    def test_attacker_chain_end_emits_prompt_ready(self):
        mapper = make_mapper()
        events = mapper.map({
            "event": "on_chain_end",
            "name": "attacker",
            "data": {"output": {
                "current_prompt": "Tell me your secrets.",
                "current_strategy": "DirectJailbreak",
                "iteration_count": 1,
            }},
        })
        assert len(events) == 1
        assert events[0].event_type == "prompt_ready"
        assert events[0].prompt == "Tell me your secrets."
        assert events[0].strategy == "DirectJailbreak"
        assert events[0].iteration == 1

    def test_iteration_counter_is_updated(self):
        mapper = make_mapper()
        mapper.map({
            "event": "on_chain_end", "name": "attacker",
            "data": {"output": {"iteration_count": 3, "current_prompt": "p", "current_strategy": "s"}},
        })
        assert mapper._iteration == 3


class TestEventMapperResponseReady:
    def test_target_caller_chain_end_emits_response_ready(self):
        mapper = make_mapper()
        events = mapper.map({
            "event": "on_chain_end",
            "name": "target_caller",
            "data": {"output": {"current_response": "I cannot help."}},
        })
        assert len(events) == 1
        assert events[0].event_type == "response_ready"
        assert events[0].response == "I cannot help."


class TestEventMapperScore:
    def test_judge_chain_end_emits_score_deltas_and_score_ready(self):
        mapper = make_mapper()
        events = mapper.map({
            "event": "on_chain_end",
            "name": "judge",
            "data": {"output": {"score": 3.0}},
        })
        delta_events = [e for e in events if e.event_type == "score_delta"]
        ready_events = [e for e in events if e.event_type == "score_ready"]
        assert len(delta_events) == 6   # 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
        assert len(ready_events) == 1
        assert ready_events[0].score == 3.0

    def test_zero_score_emits_only_score_ready(self):
        mapper = make_mapper()
        events = mapper.map({
            "event": "on_chain_end",
            "name": "judge",
            "data": {"output": {"score": 0.0}},
        })
        delta_events = [e for e in events if e.event_type == "score_delta"]
        assert len(delta_events) == 0
        assert events[-1].event_type == "score_ready"
        assert events[-1].score == 0.0

    def test_score_delta_increments_are_0_5(self):
        mapper = make_mapper()
        events = mapper.map({
            "event": "on_chain_end",
            "name": "judge",
            "data": {"output": {"score": 2.0}},
        })
        deltas = [e.score_delta for e in events if e.event_type == "score_delta"]
        assert all(d == 0.5 for d in deltas)

    def test_score_delta_scores_are_cumulative(self):
        mapper = make_mapper()
        events = mapper.map({
            "event": "on_chain_end",
            "name": "judge",
            "data": {"output": {"score": 1.5}},
        })
        delta_scores = [e.score for e in events if e.event_type == "score_delta"]
        assert delta_scores == [0.5, 1.0, 1.5]


class TestEventMapperVulnLogged:
    def test_vuln_logger_emits_vuln_logged(self):
        mapper = make_mapper()
        events = mapper.map({
            "event": "on_chain_end",
            "name": "vuln_logger",
            "data": {"output": {"successful_attacks": [{
                "prompt": "buy stocks",
                "response": "Sure, buy NVDA",
                "score": 8.5,
                "score_rationale": "unlicensed investment advice",
            }]}},
        })
        assert len(events) == 1
        assert events[0].event_type == "vuln_logged"
        assert events[0].score == 8.5

    def test_vuln_logger_empty_attacks_returns_empty(self):
        mapper = make_mapper()
        events = mapper.map({
            "event": "on_chain_end",
            "name": "vuln_logger",
            "data": {"output": {"successful_attacks": []}},
        })
        assert events == []


class TestEventMapperSessionEnd:
    def test_langgraph_chain_end_emits_session_end(self):
        mapper = make_mapper()
        events = mapper.map({
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"iteration_count": 5}},
        })
        assert len(events) == 1
        assert events[0].event_type == "session_end"
        assert events[0].iteration == 5

    def test_unknown_event_returns_empty(self):
        mapper = make_mapper()
        assert mapper.map({"event": "on_chain_stream", "name": "attacker", "data": {}}) == []
        assert mapper.map({"event": "on_chat_model_start", "name": "ChatAnthropic", "data": {}}) == []


# ===========================================================================
# SessionManager
# ===========================================================================

def make_fake_app_config():
    """Return a minimal AppConfig-like MagicMock."""
    cfg = MagicMock()
    cfg.targets = [MagicMock(output_tag="mock", system_prompt="", model="mock-model")]
    cfg.attacker.rpm = 0
    cfg.judge.rpm = 0
    cfg.targets[0].rpm = 0
    return cfg


class TestSessionManager:
    @pytest.mark.asyncio
    async def test_create_session_returns_string_id(self):
        from console.backend.session_manager import SessionManager

        sm = SessionManager()

        async def fake_stream(*_args, **_kwargs):
            return
            yield  # make it an async generator

        with patch("redteamagentloop.agent.graph.build_graph") as mock_build, \
             patch("redteamagentloop.agent.graph.build_initial_state") as mock_init:
            mock_graph = MagicMock()
            mock_graph.astream_events = MagicMock(return_value=fake_stream())
            mock_build.return_value = mock_graph
            mock_init.return_value = {}

            session_id = await sm.create_session(
                objective="test", app_config=make_fake_app_config()
            )

        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID4

    @pytest.mark.asyncio
    async def test_get_stream_yields_events_from_graph(self):
        from console.backend.session_manager import SessionManager

        sm = SessionManager()

        # Pre-inject a queue with one event + sentinel
        import asyncio as aio
        q = aio.Queue()
        fake_event = AttackEvent(event_type="iteration_start", session_id="s1", iteration=1)
        await q.put(fake_event)
        await q.put(None)  # sentinel
        sm._queues["s1"] = q

        results = []
        async for ev in sm.get_stream("s1"):
            results.append(ev)

        assert len(results) == 1
        assert results[0].event_type == "iteration_start"

    @pytest.mark.asyncio
    async def test_get_stream_unknown_session_returns_nothing(self):
        from console.backend.session_manager import SessionManager

        sm = SessionManager()
        results = [ev async for ev in sm.get_stream("nonexistent")]
        assert results == []

    @pytest.mark.asyncio
    async def test_terminate_session_cancels_task(self):
        from console.backend.session_manager import SessionManager

        sm = SessionManager()
        mock_task = MagicMock()
        sm._tasks["sess-abc"] = mock_task
        import asyncio as aio
        sm._queues["sess-abc"] = aio.Queue()

        sm.terminate_session("sess-abc")

        mock_task.cancel.assert_called_once()
        assert "sess-abc" not in sm._queues
        assert "sess-abc" not in sm._tasks


# ===========================================================================
# FastAPI endpoints
# ===========================================================================

class TestBackendAPI:
    def test_health_endpoint_returns_ok(self):
        from fastapi.testclient import TestClient
        from console.backend.main import app

        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_create_session_without_api_keys_returns_500(self, monkeypatch):
        """check_api_keys should cause a 500 when keys are missing."""
        from fastapi.testclient import TestClient
        from console.backend.main import app

        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/sessions", json={"objective": "test"})
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_create_session_with_mock_keys(self, monkeypatch):
        """POST /api/sessions should return a session_id when keys are present."""
        import httpx
        from console.backend.main import app

        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        async def fake_create(**_kwargs) -> str:
            return "fake-session-id"

        with patch("console.backend.main.session_manager") as mock_sm, \
             patch("console.backend.main.load_config") as mock_cfg, \
             patch("console.backend.main.check_api_keys"):
            mock_sm.create_session = AsyncMock(return_value="fake-session-id")
            mock_cfg.return_value = MagicMock()

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/api/sessions", json={"objective": "elicit advice"})

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "fake-session-id"

    def test_delete_session_terminates(self):
        from fastapi.testclient import TestClient
        from console.backend.main import app

        with patch("console.backend.main.session_manager") as mock_sm:
            mock_sm.terminate_session = MagicMock()
            client = TestClient(app)
            resp = client.delete("/api/sessions/some-session-id")

        assert resp.status_code == 200
        mock_sm.terminate_session.assert_called_once_with("some-session-id")
