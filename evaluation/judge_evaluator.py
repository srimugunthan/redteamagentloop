"""Judge quality evaluation framework.

Loads a ground-truth dataset, runs the judge on each item, and computes
accuracy metrics (MAE, RMSE, Pearson r) against human scores.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

from langchain_core.messages import HumanMessage
from jinja2 import Template


_TEMPLATE_PATH = Path(__file__).parent.parent / "redteamagentloop" / "prompts" / "judge_template.j2"
_JUDGE_TEMPLATE: Template = Template(_TEMPLATE_PATH.read_text())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalItem:
    id: int
    strategy: str
    target_objective: str
    prompt: str
    response: str
    human_score: float
    human_rationale: str


@dataclass
class EvalMetrics:
    mae: float          # Mean Absolute Error
    rmse: float         # Root Mean Squared Error
    pearson_r: float    # Pearson correlation coefficient
    n: int              # Number of items evaluated
    mean_human: float   # Mean human score
    mean_predicted: float  # Mean predicted score
    passes_mae_threshold: bool   # MAE <= 1.5
    strategy_breakdown: dict[str, dict] = field(default_factory=dict)
    # Optional RAGAS metrics (populated by ragas_eval.py)
    ragas_factual_correctness: float | None = None
    ragas_answer_relevancy: float | None = None


@dataclass
class EvalResults:
    items: list[EvalItem]
    predicted_scores: list[float]
    judge_reasonings: list[str]
    metrics: EvalMetrics


# ---------------------------------------------------------------------------
# JudgeEvaluator
# ---------------------------------------------------------------------------

class JudgeEvaluator:
    """Evaluates judge quality against a human-labelled dataset."""

    def load_dataset(self, path: str) -> list[EvalItem]:
        """Load JSONL evaluation dataset."""
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                items.append(EvalItem(
                    id=data["id"],
                    strategy=data["strategy"],
                    target_objective=data["target_objective"],
                    prompt=data["prompt"],
                    response=data["response"],
                    human_score=float(data["human_score"]),
                    human_rationale=data["human_rationale"],
                ))
        return items

    async def evaluate_all(
        self,
        judge_fn: Callable[[str, str, str], Awaitable[dict]],
        items: list[EvalItem],
        concurrency: int = 5,
    ) -> EvalResults:
        """Run judge_fn on all items and collect results.

        Args:
            judge_fn: async callable(target_objective, prompt, response) -> dict
                      with keys: score (float), reasoning (str)
            items: list of EvalItem from load_dataset()
            concurrency: max parallel judge calls

        Returns:
            EvalResults with predicted scores and computed metrics
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _call(item: EvalItem) -> tuple[float, str]:
            async with semaphore:
                try:
                    result = await judge_fn(item.target_objective, item.prompt, item.response)
                    score = max(0.0, min(10.0, float(result.get("score", 0.0))))
                    reasoning = result.get("reasoning", "")
                    return score, reasoning
                except Exception as exc:
                    return 0.0, f"[ERROR: {exc}]"

        tasks = [_call(item) for item in items]
        results = await asyncio.gather(*tasks)

        predicted_scores = [r[0] for r in results]
        reasonings = [r[1] for r in results]
        human_scores = [item.human_score for item in items]

        metrics = self.compute_metrics(predicted_scores, human_scores, items)
        return EvalResults(
            items=items,
            predicted_scores=predicted_scores,
            judge_reasonings=reasonings,
            metrics=metrics,
        )

    def compute_metrics(
        self,
        predicted: list[float],
        actual: list[float],
        items: list[EvalItem] | None = None,
    ) -> EvalMetrics:
        """Compute MAE, RMSE, Pearson r, and per-strategy breakdown."""
        n = len(predicted)
        if n == 0:
            raise ValueError("Empty prediction list")
        if len(actual) != n:
            raise ValueError(f"Length mismatch: predicted={n}, actual={len(actual)}")

        errors = [abs(p - a) for p, a in zip(predicted, actual)]
        sq_errors = [(p - a) ** 2 for p, a in zip(predicted, actual)]

        mae = sum(errors) / n
        rmse = math.sqrt(sum(sq_errors) / n)

        mean_p = sum(predicted) / n
        mean_a = sum(actual) / n

        # Pearson r
        cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual)) / n
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in predicted) / n)
        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in actual) / n)
        if std_p == 0 or std_a == 0:
            pearson_r = 0.0
        else:
            pearson_r = cov / (std_p * std_a)

        # Per-strategy breakdown
        strategy_breakdown: dict[str, dict] = {}
        if items:
            strategy_groups: dict[str, list[tuple[float, float]]] = {}
            for item, pred in zip(items, predicted):
                strategy_groups.setdefault(item.strategy, []).append((pred, item.human_score))
            for strategy, pairs in strategy_groups.items():
                preds = [p for p, _ in pairs]
                actuals = [a for _, a in pairs]
                s_errors = [abs(p - a) for p, a in zip(preds, actuals)]
                strategy_breakdown[strategy] = {
                    "n": len(pairs),
                    "mae": sum(s_errors) / len(s_errors),
                    "mean_human": sum(actuals) / len(actuals),
                    "mean_predicted": sum(preds) / len(preds),
                }

        return EvalMetrics(
            mae=mae,
            rmse=rmse,
            pearson_r=pearson_r,
            n=n,
            mean_human=mean_a,
            mean_predicted=mean_p,
            passes_mae_threshold=mae <= 1.5,
            strategy_breakdown=strategy_breakdown,
        )

    def generate_report(self, results: EvalResults, output_path: str) -> None:
        """Write a Markdown evaluation report to output_path."""
        m = results.metrics
        lines = [
            "# Judge Evaluation Report",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Items evaluated | {m.n} |",
            f"| MAE | {m.mae:.3f} |",
            f"| RMSE | {m.rmse:.3f} |",
            f"| Pearson r | {m.pearson_r:.3f} |",
            f"| Mean human score | {m.mean_human:.2f} |",
            f"| Mean predicted score | {m.mean_predicted:.2f} |",
            f"| Passes MAE ≤ 1.5 | {'✅ YES' if m.passes_mae_threshold else '❌ NO'} |",
        ]

        if m.ragas_factual_correctness is not None:
            lines.append(f"| RAGAS FactualCorrectness | {m.ragas_factual_correctness:.3f} |")
        if m.ragas_answer_relevancy is not None:
            lines.append(f"| RAGAS AnswerRelevancy | {m.ragas_answer_relevancy:.3f} |")

        lines += ["", "## Per-Strategy Breakdown", ""]
        if m.strategy_breakdown:
            lines.append("| Strategy | N | MAE | Mean Human | Mean Predicted |")
            lines.append("|----------|---|-----|-----------|----------------|")
            for strategy, stats in sorted(m.strategy_breakdown.items()):
                lines.append(
                    f"| {strategy} | {stats['n']} | {stats['mae']:.2f} | "
                    f"{stats['mean_human']:.2f} | {stats['mean_predicted']:.2f} |"
                )

        lines += ["", "## Item-Level Results", ""]
        lines.append("| ID | Strategy | Human | Predicted | Error |")
        lines.append("|----|----------|-------|-----------|-------|")
        for item, pred in zip(results.items, results.predicted_scores):
            err = abs(pred - item.human_score)
            lines.append(
                f"| {item.id} | {item.strategy} | {item.human_score:.1f} | {pred:.1f} | {err:.1f} |"
            )

        report = "\n".join(lines) + "\n"
        Path(output_path).write_text(report)


# ---------------------------------------------------------------------------
# Convenience: build judge_fn from a real or mock LLM
# ---------------------------------------------------------------------------

def make_judge_fn(judge_llm) -> Callable[[str, str, str], Awaitable[dict]]:
    """Wrap a LangChain judge LLM into the callable expected by evaluate_all."""
    from redteamagentloop.agent.nodes.judge import JudgeOutput

    async def judge_fn(target_objective: str, prompt: str, response: str) -> dict:
        prompt_text = _JUDGE_TEMPLATE.render(
            target_objective=target_objective,
            adversarial_prompt=prompt,
            target_response=response,
        )
        structured = judge_llm.with_structured_output(JudgeOutput, method="json_mode")
        try:
            result: JudgeOutput = await structured.ainvoke([HumanMessage(content=prompt_text)])
            return {
                "score": max(0.0, min(10.0, result.score)),
                "reasoning": result.reasoning,
                "violated_policy": result.violated_policy,
                "is_refusal": result.is_refusal,
            }
        except Exception as exc:
            return {"score": 0.0, "reasoning": f"[ERROR: {exc}]", "violated_policy": "error", "is_refusal": True}

    return judge_fn
