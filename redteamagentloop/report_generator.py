"""HTML report generator for RedTeamAgentLoop sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Template

if TYPE_CHECKING:
    from redteamagentloop.agent.state import AttackRecord
    from evaluation.judge_evaluator import EvalMetrics

_DEFAULT_TEMPLATE = Path(__file__).parent / "templates" / "report.html.j2"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SessionReport:
    session_id: str
    target_model: str
    objective: str
    vuln_threshold: float
    total_iterations: int
    attack_history: list["AttackRecord"]
    successful_attacks: list["AttackRecord"]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    eval_metrics: "EvalMetrics | None" = None

    @property
    def vulnerabilities_found(self) -> int:
        return len(self.successful_attacks)

    @property
    def strategies_tested(self) -> int:
        return len({r["strategy"] for r in self.attack_history})

    @property
    def max_score(self) -> float:
        return max((r["score"] for r in self.attack_history), default=0.0)

    @property
    def avg_score(self) -> float:
        if not self.attack_history:
            return 0.0
        return sum(r["score"] for r in self.attack_history) / len(self.attack_history)


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generates HTML and Markdown reports from a SessionReport."""

    def load_session_data(
        self,
        session_id: str,
        attack_history: list["AttackRecord"],
        successful_attacks: list["AttackRecord"],
        target_model: str,
        objective: str,
        vuln_threshold: float,
        total_iterations: int,
        eval_metrics: "EvalMetrics | None" = None,
    ) -> SessionReport:
        """Build a SessionReport from raw data.

        In Phase 7 the caller passes the final LangGraph state fields directly.
        A future version can pull from StorageManager instead.
        """
        return SessionReport(
            session_id=session_id,
            target_model=target_model,
            objective=objective,
            vuln_threshold=vuln_threshold,
            total_iterations=total_iterations,
            attack_history=attack_history,
            successful_attacks=successful_attacks,
            eval_metrics=eval_metrics,
        )

    def render_html(
        self,
        report: SessionReport,
        template_path: str = str(_DEFAULT_TEMPLATE),
    ) -> str:
        """Render the session report to an HTML string."""
        template_text = Path(template_path).read_text()
        template = Template(template_text)

        # Build strategy chart data from attack history
        strategy_scores: dict[str, list[float]] = {}
        for rec in report.attack_history:
            strategy_scores.setdefault(rec["strategy"], []).append(rec["score"])
        chart_labels = sorted(strategy_scores.keys())
        chart_scores = [
            round(sum(strategy_scores[s]) / len(strategy_scores[s]), 2)
            for s in chart_labels
        ]

        return template.render(
            session=report,
            strategy_chart_data={
                "labels": chart_labels,
                "scores": chart_scores,
            },
        )

    def save(
        self,
        report: SessionReport,
        output_dir: str,
        template_path: str = str(_DEFAULT_TEMPLATE),
    ) -> str:
        """Render and save the HTML report.

        Returns the path to the saved file.
        """
        html = self.render_html(report, template_path)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{report.session_id[:8]}_{ts}.html"
        path = out_dir / filename
        path.write_text(html, encoding="utf-8")
        return str(path)
