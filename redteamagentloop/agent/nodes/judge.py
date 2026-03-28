"""Judge node — scores target responses for policy violations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Template
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from redteamagentloop.agent.state import RedTeamState

# Load template once at module level.
_TEMPLATE_PATH = Path(__file__).parent.parent.parent.parent / "prompts" / "judge_template.j2"
_JUDGE_TEMPLATE: Template = Template(_TEMPLATE_PATH.read_text())


class JudgeOutput(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    violated_policy: str
    reasoning: str
    is_refusal: bool


async def judge_node(state: "RedTeamState", config: RunnableConfig) -> dict:
    if state.get("error") is not None:
        return {}  # pass through upstream error (e.g. max_iterations_reached)

    if not state["current_response"]:
        return {
            "score": 0.0,
            "score_rationale": "Target produced no response.",
            "error": None,
        }

    cfg = config.get("configurable", {})
    app_config = cfg.get("app_config")

    judge_llm = cfg.get("judge_llm")
    if judge_llm is None:
        jc = app_config.judge
        judge_llm = ChatAnthropic(
            model=jc.model,
            temperature=jc.temperature,
            max_tokens=jc.max_tokens,
        )

    prompt_text = _JUDGE_TEMPLATE.render(
        target_objective=state["target_objective"],
        adversarial_prompt=state["current_prompt"],
        target_response=state["current_response"],
    )

    structured_llm = judge_llm.with_structured_output(JudgeOutput, method="json_mode")

    for attempt in range(2):
        try:
            result: JudgeOutput = await structured_llm.ainvoke(
                [HumanMessage(content=prompt_text)]
            )
            score = max(0.0, min(10.0, result.score))
            return {
                "score": score,
                "score_rationale": result.reasoning,
                "error": None,
            }
        except Exception:
            if attempt == 1:
                return {"score": 0.0, "score_rationale": "Judge parsing failed.", "error": None}

    return {"score": 0.0, "score_rationale": "Judge failed.", "error": None}
