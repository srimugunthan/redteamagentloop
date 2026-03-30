"""CLI entry point for RedTeamAgentLoop."""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv
from rich.console import Console

from redteamagentloop.config import check_api_keys, check_authorization, load_config

console = Console()


async def _run_target(
    graph,
    initial_state,
    app_config,
    target,
    output_dir: str,
) -> None:
    """Run the graph against a single target, persist results, and save report."""
    from rich.panel import Panel
    from redteamagentloop.ratelimit import RateLimiter
    from redteamagentloop.storage.manager import StorageManager
    from reports.report_generator import ReportGenerator
    from reports.terminal_dashboard import TerminalDashboard

    console.print(Panel(
        f"[bold cyan]Target:[/bold cyan] {target.model} ({target.output_tag})\n"
        f"[bold cyan]Objective:[/bold cyan] {initial_state['target_objective']}",
        title="RedTeamAgentLoop — starting run",
    ))

    storage_cfg = app_config.storage
    storage = StorageManager(
        jsonl_path=storage_cfg.jsonl_path.replace("{target_tag}", target.output_tag),
        sqlite_path=storage_cfg.sqlite_path,
        chromadb_path=storage_cfg.chromadb_path.replace("{target_tag}", target.output_tag),
        embedding_model=storage_cfg.embedding_model,
        dedup_threshold=storage_cfg.dedup_threshold,
    )

    dashboard = TerminalDashboard(
        objective=initial_state["target_objective"],
        target=target.model,
        vuln_threshold=app_config.loop.vuln_threshold,
    )

    # Build per-run rate limiters (0 rpm = disabled).
    attacker_rate_limiter = RateLimiter(app_config.attacker.rpm)
    target_rate_limiter = RateLimiter(target.rpm)
    judge_rate_limiter = RateLimiter(app_config.judge.rpm)

    run_config = {"configurable": {
        "app_config": app_config,
        "attacker_rate_limiter": attacker_rate_limiter,
        "target_rate_limiter": target_rate_limiter,
        "judge_rate_limiter": judge_rate_limiter,
    }}

    # Stream the graph so the dashboard updates live after each iteration.
    # astream(stream_mode="values") yields the full merged state after every node;
    # the Live context must be active for dashboard.update() to render anything.
    final_state = {}
    displayed = 0
    with dashboard.live_context():
        async for state in graph.astream(initial_state, config=run_config, stream_mode="values"):
            final_state = state
            history_so_far = state.get("attack_history", [])
            while displayed < len(history_so_far):
                dashboard.update(history_so_far[displayed])
                displayed += 1

    successes = final_state.get("successful_attacks", [])
    history = final_state.get("attack_history", [])
    iterations = final_state.get("iteration_count", 0)

    # Persist successful attacks to storage
    for rec in successes:
        await storage.log_attack(rec)

    dashboard.print_final_summary(final_state)

    # Generate HTML report
    generator = ReportGenerator()
    report = generator.load_session_data(
        session_id=initial_state["session_id"],
        attack_history=history,
        successful_attacks=successes,
        target_model=target.model,
        objective=initial_state["target_objective"],
        vuln_threshold=app_config.loop.vuln_threshold,
        total_iterations=iterations,
    )
    report_path = generator.save(report, output_dir)
    console.print(f"[dim]Report saved → {report_path}[/dim]")


def main() -> None:
    # Load .env before parsing config so GROQ_API_KEY / ANTHROPIC_API_KEY are set.
    load_dotenv()

    parser = argparse.ArgumentParser(description="RedTeamAgentLoop — LLM red teaming agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--objective", required=True, help="Red team objective (what the target should NOT do)")
    parser.add_argument("--system-prompt", default="", help="System prompt to use for the target LLM")
    parser.add_argument("--target", default=None, help="output_tag of the target to test (default: all targets)")
    parser.add_argument("--output-dir", default="reports/output", help="Directory for HTML reports")
    parser.add_argument("--auth", default="authorization.txt", help="Path to authorization.txt")
    args = parser.parse_args()

    check_authorization(args.auth)
    app_config = load_config(args.config)

    # Fail fast if required API keys are missing — before any LLM node runs.
    check_api_keys(app_config)

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
            await _run_target(graph, initial_state, app_config, target, args.output_dir)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
