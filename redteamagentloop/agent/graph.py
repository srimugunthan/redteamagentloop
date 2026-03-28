"""LangGraph StateGraph assembly for RedTeamAgentLoop."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from redteamagentloop.agent.nodes.attacker import attacker_node
from redteamagentloop.agent.nodes.judge import judge_node
from redteamagentloop.agent.nodes.loop_controller import (
    loop_controller_node,
    route_after_judge,
)
from redteamagentloop.agent.nodes.mutation_engine import mutation_engine_node
from redteamagentloop.agent.nodes.target_caller import target_caller_node
from redteamagentloop.agent.nodes.vuln_logger import vuln_logger_node
from redteamagentloop.agent.state import RedTeamState

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph

    from redteamagentloop.config import AppConfig


def build_graph(app_config: "AppConfig") -> "CompiledGraph":
    """Assemble and compile the RedTeamAgentLoop StateGraph."""
    graph = StateGraph(RedTeamState)

    graph.add_node("attacker", attacker_node)
    graph.add_node("target_caller", target_caller_node)
    graph.add_node("judge", judge_node)
    graph.add_node("loop_controller", loop_controller_node)
    graph.add_node("vuln_logger", vuln_logger_node)
    graph.add_node("mutation_engine", mutation_engine_node)

    graph.add_edge(START, "attacker")
    graph.add_edge("attacker", "target_caller")
    graph.add_edge("target_caller", "judge")
    graph.add_edge("judge", "loop_controller")
    graph.add_conditional_edges(
        "loop_controller",
        route_after_judge,
        {
            "vuln_logger": "vuln_logger",
            "mutation_engine": "mutation_engine",
            "attacker": "attacker",
            "END": END,
        },
    )
    graph.add_edge("vuln_logger", "mutation_engine")
    graph.add_edge("mutation_engine", "attacker")

    return graph.compile()


def build_initial_state(
    config: "AppConfig",
    target_objective: str,
    target_system_prompt: str = "",
) -> RedTeamState:
    """Construct a fresh RedTeamState from config and a run objective."""
    return RedTeamState(
        current_strategy="",
        current_prompt="",
        current_response="",
        score=0.0,
        score_rationale="",
        iteration_count=0,
        attack_history=[],
        successful_attacks=[],
        failed_strategies=set(),
        mutation_queue=[],
        current_mutations=[],
        target_system_prompt=target_system_prompt,
        target_objective=target_objective,
        max_iterations=config.loop.max_iterations,
        vuln_threshold=config.loop.vuln_threshold,
        session_id=str(uuid.uuid4()),
        error=None,
    )


def get_graph_image(app_config: "AppConfig") -> bytes:
    """Return a PNG of the compiled graph (requires graphviz extras)."""
    compiled = build_graph(app_config)
    return compiled.get_graph().draw_mermaid_png()
