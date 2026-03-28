"""Target caller node — thin relay to the target LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from redteamagentloop.agent.state import RedTeamState


async def target_caller_node(state: "RedTeamState", config: RunnableConfig) -> dict:
    if state.get("error") is not None:
        return {}  # pass through upstream error (e.g. max_iterations_reached)

    if not state["current_prompt"]:
        return {"current_response": "", "error": "Empty prompt — nothing to send to target."}

    cfg = config.get("configurable", {})
    app_config = cfg.get("app_config")

    target_llm = cfg.get("target_llm")
    if target_llm is None:
        # Use the first target from config by default.
        # The caller can override via config["configurable"]["target_llm"].
        tc = app_config.targets[0]
        target_llm = ChatOpenAI(
            model=tc.model,
            base_url=tc.base_url,
            api_key=tc.api_key,
            timeout=tc.timeout_seconds,
            temperature=0.0,  # deterministic — reproducibility matters
        )

    messages = [
        SystemMessage(content=state["target_system_prompt"]),
        HumanMessage(content=state["current_prompt"]),
    ]

    try:
        response = await target_llm.ainvoke(messages)
        return {"current_response": response.content.strip(), "error": None}
    except httpx.TimeoutException as exc:
        return {"current_response": "", "error": f"Target LLM timeout: {exc}"}
    except Exception as exc:
        return {"current_response": "", "error": f"Target LLM error: {exc}"}
