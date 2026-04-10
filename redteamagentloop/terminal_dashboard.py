"""Rich terminal dashboard for live session monitoring and final summary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from redteamagentloop.agent.state import AttackRecord
    from redteamagentloop.report_generator import SessionReport

console = Console()


# ---------------------------------------------------------------------------
# Live dashboard (shown while the graph is running)
# ---------------------------------------------------------------------------

class TerminalDashboard:
    """Rich Live dashboard that updates as the agent produces attack records.

    Usage:
        dashboard = TerminalDashboard(objective="...", target="tinyllama")
        with dashboard.live_context():
            # call dashboard.update(record) after each scored iteration
            ...
        dashboard.print_final_summary(final_state)
    """

    def __init__(self, objective: str, target: str, vuln_threshold: float = 7.0) -> None:
        self._objective = objective
        self._target = target
        self._vuln_threshold = vuln_threshold
        self._history: list["AttackRecord"] = []
        self._live = Live(self._render(), refresh_per_second=4, console=console)

    # ------------------------------------------------------------------
    # Live context
    # ------------------------------------------------------------------

    def live_context(self) -> Live:
        """Return the Rich Live context manager."""
        return self._live

    def update(self, record: "AttackRecord") -> None:
        """Call after each scored iteration to refresh the display."""
        self._history.append(record)
        self._live.update(self._render())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> Panel:
        iteration = len(self._history)
        successes = [r for r in self._history if r["score"] >= self._vuln_threshold]

        last_strategy = self._history[-1]["strategy"] if self._history else "—"
        last_score = self._history[-1]["score"] if self._history else 0.0

        score_color = "red" if last_score >= self._vuln_threshold else "yellow" if last_score >= 4.0 else "green"

        header = (
            f"[bold]Target:[/bold]      {self._target}\n"
            f"[bold]Objective:[/bold]   {self._objective[:60]}\n"
            f"[bold]Iteration:[/bold]   {iteration}\n"
            f"[bold]Strategy:[/bold]    {last_strategy}\n"
            f"[bold]Last score:[/bold]  [{score_color}]{last_score:.1f}[/{score_color}] / 10\n"
            f"[bold]Vulns found:[/bold] [{'red' if successes else 'green'}]{len(successes)}[/{'red' if successes else 'green'}]"
        )

        table = Table(show_header=True, show_lines=False, padding=(0, 1), expand=True)
        table.add_column("Iter", style="dim", width=5)
        table.add_column("Strategy", width=22)
        table.add_column("Score", justify="right", width=6)
        table.add_column("Prompt snippet")

        for rec in self._history[-8:]:  # show last 8 rows
            s = rec["score"]
            sc = "bold red" if s >= self._vuln_threshold else "yellow" if s >= 4.0 else "green"
            table.add_row(
                str(rec["iteration"]),
                rec["strategy"],
                f"[{sc}]{s:.1f}[/{sc}]",
                rec["prompt"][:55] + ("…" if len(rec["prompt"]) > 55 else ""),
            )

        from rich.columns import Columns
        from rich.console import Group
        content = Group(header, "", table)
        return Panel(content, title="[bold cyan]RedTeamAgentLoop — Live[/bold cyan]", border_style="cyan")

    # ------------------------------------------------------------------
    # Final summary (printed after graph terminates)
    # ------------------------------------------------------------------

    def print_final_summary(self, final_state: dict) -> None:
        """Print a final Rich summary panel after the graph has finished."""
        history = final_state.get("attack_history", [])
        successes = final_state.get("successful_attacks", [])
        iterations = final_state.get("iteration_count", 0)
        error = final_state.get("error")

        max_score = max((r["score"] for r in history), default=0.0)
        avg_score = (sum(r["score"] for r in history) / len(history)) if history else 0.0

        status_text = f"[yellow]{error}[/yellow]" if error else "[bold green]completed[/bold green]"
        vuln_color = "red" if successes else "green"

        summary = (
            f"[bold]Status:[/bold]         {status_text}\n"
            f"[bold]Total iterations:[/bold] {iterations}\n"
            f"[bold]Vulnerabilities:[/bold]  [{vuln_color}]{len(successes)}[/{vuln_color}]\n"
            f"[bold]Max score:[/bold]        [{'red' if max_score >= self._vuln_threshold else 'green'}]{max_score:.1f}[/{'red' if max_score >= self._vuln_threshold else 'green'}] / 10\n"
            f"[bold]Avg score:[/bold]        {avg_score:.2f} / 10"
        )

        console.print(Panel(
            summary,
            title="[bold]RedTeamAgentLoop — Session complete[/bold]",
            border_style="green" if not successes else "red",
        ))

        if successes:
            vuln_table = Table(title="[bold red]Confirmed vulnerabilities[/bold red]", show_lines=True)
            vuln_table.add_column("Iter", style="dim", width=5)
            vuln_table.add_column("Strategy", width=22)
            vuln_table.add_column("Score", justify="right", width=6)
            vuln_table.add_column("Prompt")
            for rec in successes:
                vuln_table.add_row(
                    str(rec["iteration"]),
                    rec["strategy"],
                    f"[bold red]{rec['score']:.1f}[/bold red]",
                    rec["prompt"][:80] + ("…" if len(rec["prompt"]) > 80 else ""),
                )
            console.print(vuln_table)


# ---------------------------------------------------------------------------
# Standalone final summary (no live context needed)
# ---------------------------------------------------------------------------

def print_run_summary(final_state: dict, target_label: str, objective: str, vuln_threshold: float = 7.0) -> None:
    """Print a final summary panel from a completed final_state dict.

    Equivalent to TerminalDashboard.print_final_summary but usable
    without constructing a TerminalDashboard (e.g. in the CLI).
    """
    dashboard = TerminalDashboard(objective=objective, target=target_label, vuln_threshold=vuln_threshold)
    dashboard.print_final_summary(final_state)
