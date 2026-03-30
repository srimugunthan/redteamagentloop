"""FastAPI SSE backend for the RedTeamAgentLoop Live Attack Console.

Endpoints
---------
POST   /api/sessions              — start a new red-team session
GET    /api/stream/{session_id}   — SSE stream of AttackEvents
DELETE /api/sessions/{session_id} — cancel a running session
GET    /api/health                — liveness probe

The built React frontend is served as static files from
console/frontend/dist/ when present (production).  In development, run
`npm run dev` in console/frontend/ and let Vite proxy /api to this server.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from console.backend.models import SessionRequest, SessionResponse
from console.backend.session_manager import session_manager
from redteamagentloop.config import check_api_keys, load_config

load_dotenv()

app = FastAPI(title="RedTeamAgentLoop Console", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(body: SessionRequest) -> SessionResponse:
    app_config = load_config()
    try:
        check_api_keys(app_config)
    except SystemExit as exc:
        raise HTTPException(status_code=500, detail="Required API keys are not configured.") from exc

    session_id = await session_manager.create_session(
        objective=body.objective,
        system_prompt=body.system_prompt,
        app_config=app_config,
        target_tag=body.target_tag,
    )
    return SessionResponse(session_id=session_id)


@app.get("/api/stream/{session_id}")
async def stream_session(session_id: str) -> EventSourceResponse:
    """Server-Sent Events endpoint.  Each event is a JSON-serialised AttackEvent."""
    async def generator():
        async for event in session_manager.get_stream(session_id):
            yield {"data": event.model_dump_json()}

    return EventSourceResponse(generator())


@app.delete("/api/sessions/{session_id}")
async def terminate_session(session_id: str) -> dict:
    session_manager.terminate_session(session_id)
    return {"status": "terminated", "session_id": session_id}


# ---------------------------------------------------------------------------
# Serve built React frontend (production)
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
