"""CLI entry point for RedTeamAgentLoop."""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from redteamagentloop.config import check_authorization, load_config

console = Console()


async def _run_target(
    graph,
    initial_state,
    app_config,
    target,
) -> None:
    """Run the graph against a single target and print a summary."""
    console.print(Panel(
        f"[bold cyan]Target:[/bold cyan] {target.model} ({target.output_tag})\n"
        f"[bold cyan]Objective:[/bold cyan] {initial_state['target_objective']}",
        title="RedTeamAgentLoop — starting run",
    ))

    final_state = await graph.ainvoke(
        initial_state,
        config={"configurable": {"app_config": app_config}},
    )

    successes = final_state.get("successful_attacks", [])
    history = final_state.get("attack_history", [])
    iterations = final_state.get("iteration_count", 0)
    error = final_state.get("error")

    # All attempts table
    all_table = Table(title=f"All attempts — {target.output_tag}", show_lines=False)
    all_table.add_column("#", style="dim", width=4)
    all_table.add_column("Strategy", width=22)
    all_table.add_column("Score", justify="right", width=6)
    all_table.add_column("Prompt (truncated)")
    all_table.add_column("Response (truncated)")

    for rec in history:
        score = rec["score"]
        score_str = f"[bold red]{score:.1f}[/bold red]" if score >= final_state.get("vuln_threshold", 7.0) else f"{score:.1f}"
        all_table.add_row(
            str(rec["iteration"]),
            rec["strategy"],
            score_str,
            rec["prompt"][:60],
            rec["response"][:80],
        )

    console.print(all_table)

    # Vulnerabilities-only table
    vuln_table = Table(title=f"[bold red]Confirmed vulnerabilities — {target.output_tag}[/bold red]")
    vuln_table.add_column("Iteration", style="dim")
    vuln_table.add_column("Strategy")
    vuln_table.add_column("Score", justify="right")
    vuln_table.add_column("Prompt (truncated)")

    for rec in successes:
        vuln_table.add_row(
            str(rec["iteration"]),
            rec["strategy"],
            f"[bold red]{rec['score']:.1f}[/bold red]",
            rec["prompt"][:80],
        )

    console.print(vuln_table)

    status = f"[yellow]{error}[/yellow]" if error else "[green]completed[/green]"
    console.print(
        f"[bold]Status:[/bold] {status}  "
        f"[bold]Iterations:[/bold] {iterations}  "
        f"[bold]Vulnerabilities found:[/bold] {len(successes)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RedTeamAgentLoop — LLM red teaming agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--objective", required=True, help="Red team objective (what the target should NOT do)")
    parser.add_argument("--system-prompt", default="", help="System prompt to use for the target LLM")
    parser.add_argument("--target", default=None, help="output_tag of the target to test (default: all targets)")
    parser.add_argument("--output-dir", default="outputs/", help="Directory for run outputs")
    parser.add_argument("--auth", default="authorization.txt", help="Path to authorization.txt")
    args = parser.parse_args()

    check_authorization(args.auth)
    app_config = load_config(args.config)

    from redteamagentloop.agent.graph import build_graph, build_initial_state

    targets = [
        t for t in app_config.targets
        if args.target is None or args.target == t.output_tag
    ]
    if not targets:
        console.print(f"[red]No targets matched '{args.target}'.[/red]")
        return

    graph = build_graph(app_config)

    async def run_all() -> None:
        for target in targets:
            initial_state = build_initial_state(
                config=app_config,
                target_objective=args.objective,
                target_system_prompt=args.system_prompt or target.system_prompt,
            )
            await _run_target(graph, initial_state, app_config, target)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
