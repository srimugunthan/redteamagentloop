"""Runner script for golden-dataset judge evaluation.

Loads the human-labelled eval dataset, runs the judge LLM against every item,
computes MAE/RMSE/Pearson r, optionally runs RAGAS FactualCorrectness on the
judge's reasoning, and writes a Markdown report.

Usage:
    # Basic run (MAE + RMSE + Pearson only)
    uv run python run_golden_dataset_eval.py

    # With RAGAS evaluation of judge reasoning quality
    uv run python run_golden_dataset_eval.py --ragas

    # Custom dataset or output path
    uv run python run_golden_dataset_eval.py \
        --dataset evaluation/judge_eval_dataset.jsonl \
        --output reports/judge_eval_report.md \
        --ragas

    # Limit to first N items (quick smoke test)
    uv run python run_golden_dataset_eval.py --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DATASET = "evaluation/judge_eval_dataset.jsonl"
DEFAULT_OUTPUT = "reports/judge_eval_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate judge LLM quality against the golden dataset."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to JSONL eval dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to write Markdown report (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Run RAGAS FactualCorrectness evaluation on judge reasoning",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max parallel judge calls (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N items (useful for quick tests)",
    )
    return parser.parse_args()


async def main() -> int:
    load_dotenv()
    args = parse_args()

    # Validate inputs
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 1

    # Load config for judge model details
    from redteamagentloop.config import load_config
    app_config = load_config(args.config)

    # Build judge LLM
    from langchain_anthropic import ChatAnthropic
    jc = app_config.judge
    judge_llm = ChatAnthropic(
        model=jc.model,
        temperature=jc.temperature,
        max_tokens=jc.max_tokens,
        api_key=anthropic_key,
    )

    # Load dataset
    from evaluation.judge_evaluator import JudgeEvaluator, make_judge_fn
    evaluator = JudgeEvaluator()
    items = evaluator.load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]

    print(f"Loaded {len(items)} items from {args.dataset}")
    print(f"Judge model: {jc.model}")

    # Run evaluation
    judge_fn = make_judge_fn(judge_llm)
    print(f"Running judge evaluation (concurrency={args.concurrency})...")
    results = await evaluator.evaluate_all(judge_fn, items, concurrency=args.concurrency)

    m = results.metrics
    print(f"\nResults:")
    print(f"  Items:      {m.n}")
    print(f"  MAE:        {m.mae:.3f}  {'PASS' if m.passes_mae_threshold else 'FAIL'} (threshold <= 1.5)")
    print(f"  RMSE:       {m.rmse:.3f}")
    print(f"  Pearson r:  {m.pearson_r:.3f}")
    print(f"  Mean human: {m.mean_human:.2f}  Mean predicted: {m.mean_predicted:.2f}")

    # Optional RAGAS evaluation
    if args.ragas:
        print("\nRunning RAGAS FactualCorrectness evaluation...")
        from evaluation.ragas_eval import run_ragas_eval, attach_ragas_metrics
        ragas_scores = run_ragas_eval(
            items,
            results.judge_reasonings,
            llm=judge_llm,
            show_progress=True,
        )
        attach_ragas_metrics(results, ragas_scores)
        fc = results.metrics.ragas_factual_correctness
        print(f"  RAGAS FactualCorrectness: {fc:.3f}" if fc is not None else "  RAGAS: no score returned")

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.generate_report(results, str(output_path))
    print(f"\nReport saved → {output_path}")

    return 0 if m.passes_mae_threshold else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
