"""Loop controller node and routing function."""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig

from redteamagentloop.agent.state import AttackRecord, RedTeamState


def route_after_judge(state: RedTeamState) -> str:
    """Conditional edge function — returns the name of the next node."""
    if state.get("error") == "max_iterations_reached":
        return "END"
    if state.get("error") is not None:
        return "END"
    if state["score"] >= state["vuln_threshold"]:
        return "vuln_logger"
    if len(state["mutation_queue"]) > 0:
        return "attacker"
    return "mutation_engine"


async def loop_controller_node(state: RedTeamState, config: RunnableConfig) -> dict:
    """Records every completed attempt to attack_history, then routing happens via route_after_judge."""
    if state.get("error") is not None:
        return {}

    if not state.get("current_prompt"):
        return {}

    record: AttackRecord = {
        "session_id": state["session_id"],
        "iteration": state["iteration_count"],
        "strategy": state["current_strategy"],
        "prompt": state["current_prompt"],
        "response": state["current_response"],
        "score": state["score"],
        "score_rationale": state["score_rationale"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "was_successful": state["score"] >= state["vuln_threshold"],
        "mutation_depth": len(state.get("current_mutations", [])),
    }
    return {"attack_history": [record]}
