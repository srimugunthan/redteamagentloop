"""RAGAS-based evaluation wrapper for judge reasoning quality.

Uses RAGAS FactualCorrectness to measure how well the judge's reasoning
aligns with human-written rationales.

The mapping to RAGAS SingleTurnSample:
  user_input  = the adversarial prompt + target response (what the judge sees)
  response    = judge's reasoning text
  reference   = human-written rationale

FactualCorrectness checks whether the judge's reasoning contains the
same factual claims as the human rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.judge_evaluator import EvalItem, EvalMetrics, EvalResults


def build_ragas_dataset(
    items: "list[EvalItem]",
    judge_reasonings: list[str],
) -> "object":
    """Build a RAGAS EvaluationDataset from eval items and judge outputs.

    Args:
        items: ground-truth items from the eval dataset
        judge_reasonings: reasoning strings from judge_evaluator.evaluate_all

    Returns:
        ragas.EvaluationDataset ready for evaluate()
    """
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    for item, reasoning in zip(items, judge_reasonings):
        user_input = (
            f"Adversarial prompt: {item.prompt}\n\n"
            f"Target response: {item.response}"
        )
        samples.append(SingleTurnSample(
            user_input=user_input,
            response=reasoning,
            reference=item.human_rationale,
        ))
    return EvaluationDataset(samples=samples)


def run_ragas_eval(
    items: "list[EvalItem]",
    judge_reasonings: list[str],
    llm=None,
    show_progress: bool = False,
) -> dict[str, float]:
    """Run RAGAS FactualCorrectness evaluation on judge reasoning quality.

    Args:
        items: ground-truth eval items
        judge_reasonings: reasoning strings produced by the judge LLM
        llm: optional LangChain LLM to use for RAGAS metrics (uses default if None)
        show_progress: whether to show RAGAS progress bar

    Returns:
        dict with keys: factual_correctness (float 0-1)
    """
    from ragas import evaluate
    from ragas.metrics.collections import FactualCorrectness

    dataset = build_ragas_dataset(items, judge_reasonings)
    metric = FactualCorrectness()
    if llm is not None:
        from ragas.llms import LangchainLLMWrapper
        metric.llm = LangchainLLMWrapper(llm)

    result = evaluate(
        dataset=dataset,
        metrics=[metric],
        show_progress=show_progress,
    )

    scores = result.to_pandas()["factual_correctness"].dropna().tolist()
    mean_fc = sum(scores) / len(scores) if scores else 0.0

    return {"factual_correctness": mean_fc}


def attach_ragas_metrics(
    eval_results: "EvalResults",
    ragas_scores: dict[str, float],
) -> None:
    """Attach RAGAS scores to an existing EvalMetrics object in-place."""
    eval_results.metrics.ragas_factual_correctness = ragas_scores.get("factual_correctness")
    eval_results.metrics.ragas_answer_relevancy = ragas_scores.get("answer_relevancy")
