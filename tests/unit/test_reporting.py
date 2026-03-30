"""Phase 7 tests — report generator and terminal dashboard."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reports.report_generator import ReportGenerator, SessionReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEMPLATE_PATH = Path(__file__).parent.parent.parent / "reports" / "templates" / "report.html.j2"


def make_attack_record(
    iteration: int = 1,
    strategy: str = "DirectJailbreak",
    score: float = 8.5,
    prompt: str = "Test adversarial prompt",
    response: str = "Test target response",
    was_successful: bool = True,
) -> dict:
    return {
        "session_id": "test-session-001",
        "iteration": iteration,
        "strategy": strategy,
        "prompt": prompt,
        "response": response,
        "score": score,
        "score_rationale": "Test rationale",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "was_successful": was_successful,
        "mutation_depth": 0,
    }


def make_session_report(
    history: list | None = None,
    successes: list | None = None,
) -> SessionReport:
    if history is None:
        history = [
            make_attack_record(1, "DirectJailbreak", 8.5, was_successful=True),
            make_attack_record(2, "PersonaHijack", 3.0, was_successful=False),
            make_attack_record(3, "ObfuscatedRequest", 1.0, was_successful=False),
        ]
    if successes is None:
        successes = [r for r in history if r["was_successful"]]

    return SessionReport(
        session_id="test-session-abc123",
        target_model="tinyllama",
        objective="elicit unlicensed investment advice",
        vuln_threshold=7.0,
        total_iterations=len(history),
        attack_history=history,
        successful_attacks=successes,
    )


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_load_session_data_returns_session_report(self):
        generator = ReportGenerator()
        report = generator.load_session_data(
            session_id="abc123",
            attack_history=[make_attack_record()],
            successful_attacks=[make_attack_record()],
            target_model="tinyllama",
            objective="test objective",
            vuln_threshold=7.0,
            total_iterations=5,
        )
        assert isinstance(report, SessionReport)
        assert report.session_id == "abc123"
        assert report.target_model == "tinyllama"

    def test_render_html_returns_string(self):
        generator = ReportGenerator()
        report = make_session_report()
        html = generator.render_html(report, str(TEMPLATE_PATH))
        assert isinstance(html, str)
        assert len(html) > 100

    def test_render_html_contains_session_id(self):
        generator = ReportGenerator()
        report = make_session_report()
        html = generator.render_html(report, str(TEMPLATE_PATH))
        assert report.session_id in html

    def test_render_html_contains_target_model(self):
        generator = ReportGenerator()
        report = make_session_report()
        html = generator.render_html(report, str(TEMPLATE_PATH))
        assert "tinyllama" in html

    def test_render_html_contains_strategy_name(self):
        generator = ReportGenerator()
        report = make_session_report()
        html = generator.render_html(report, str(TEMPLATE_PATH))
        assert "DirectJailbreak" in html

    def test_render_html_contains_correct_iteration_count(self):
        generator = ReportGenerator()
        history = [make_attack_record(i, score=float(i)) for i in range(1, 6)]
        report = make_session_report(history=history, successes=[])
        html = generator.render_html(report, str(TEMPLATE_PATH))
        assert "5" in html

    def test_render_html_is_valid_html(self):
        generator = ReportGenerator()
        report = make_session_report()
        html = generator.render_html(report, str(TEMPLATE_PATH))
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_render_html_includes_chart_js(self):
        generator = ReportGenerator()
        report = make_session_report()
        html = generator.render_html(report, str(TEMPLATE_PATH))
        assert "chart.js" in html.lower() or "Chart" in html

    def test_render_html_empty_vulnerabilities_shows_empty_state(self):
        generator = ReportGenerator()
        report = make_session_report(
            history=[make_attack_record(1, score=2.0, was_successful=False)],
            successes=[],
        )
        html = generator.render_html(report, str(TEMPLATE_PATH))
        assert "No confirmed vulnerabilities" in html

    def test_render_html_with_eval_metrics(self):
        from evaluation.judge_evaluator import EvalMetrics
        generator = ReportGenerator()
        report = make_session_report()
        report.eval_metrics = EvalMetrics(
            mae=0.75,
            rmse=1.1,
            pearson_r=0.96,
            n=10,
            mean_human=5.0,
            mean_predicted=4.8,
            passes_mae_threshold=True,
        )
        html = generator.render_html(report, str(TEMPLATE_PATH))
        assert "0.750" in html or "0.75" in html
        assert "Pearson" in html

    def test_save_creates_file_in_output_dir(self):
        generator = ReportGenerator()
        report = make_session_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generator.save(report, tmpdir, str(TEMPLATE_PATH))
            assert Path(path).exists()
            assert path.endswith(".html")

    def test_save_filename_includes_session_id_prefix(self):
        generator = ReportGenerator()
        report = make_session_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generator.save(report, tmpdir, str(TEMPLATE_PATH))
            filename = Path(path).name
            assert filename.startswith(report.session_id[:8])

    def test_save_creates_output_dir_if_missing(self):
        generator = ReportGenerator()
        report = make_session_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "new_subdir", "reports")
            path = generator.save(report, nested, str(TEMPLATE_PATH))
            assert Path(path).exists()


# ---------------------------------------------------------------------------
# SessionReport properties
# ---------------------------------------------------------------------------

class TestSessionReport:
    def test_vulnerabilities_found_counts_successful(self):
        history = [make_attack_record(i, score=float(i * 2), was_successful=(i > 3)) for i in range(1, 6)]
        successes = [r for r in history if r["was_successful"]]
        report = make_session_report(history=history, successes=successes)
        assert report.vulnerabilities_found == len(successes)

    def test_strategies_tested_counts_unique(self):
        history = [
            make_attack_record(1, strategy="DirectJailbreak"),
            make_attack_record(2, strategy="DirectJailbreak"),
            make_attack_record(3, strategy="PersonaHijack"),
        ]
        report = make_session_report(history=history, successes=[])
        assert report.strategies_tested == 2

    def test_max_score_returns_highest(self):
        history = [make_attack_record(i, score=float(i * 2)) for i in range(1, 6)]
        report = make_session_report(history=history, successes=[])
        assert report.max_score == 10.0

    def test_avg_score_computed_correctly(self):
        history = [make_attack_record(i, score=float(i * 2)) for i in range(1, 4)]  # 2, 4, 6
        report = make_session_report(history=history, successes=[])
        assert abs(report.avg_score - 4.0) < 1e-9

    def test_empty_history_safe(self):
        report = make_session_report(history=[], successes=[])
        assert report.max_score == 0.0
        assert report.avg_score == 0.0
        assert report.strategies_tested == 0


# ---------------------------------------------------------------------------
# TerminalDashboard
# ---------------------------------------------------------------------------

class TestTerminalDashboard:
    def test_dashboard_instantiates(self):
        from reports.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard(objective="test", target="tinyllama")
        assert dashboard is not None

    def test_update_appends_record(self):
        from reports.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard(objective="test", target="tinyllama")
        record = make_attack_record()
        dashboard.update(record)
        assert len(dashboard._history) == 1

    def test_print_final_summary_does_not_raise(self):
        from reports.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard(objective="test", target="tinyllama")
        final_state = {
            "attack_history": [make_attack_record(1, score=8.5)],
            "successful_attacks": [make_attack_record(1, score=8.5)],
            "iteration_count": 3,
            "error": None,
        }
        # Should not raise
        with patch.object(dashboard, "_live"):
            dashboard.print_final_summary(final_state)

    def test_print_final_summary_no_vulns_does_not_raise(self):
        from reports.terminal_dashboard import TerminalDashboard
        dashboard = TerminalDashboard(objective="test", target="tinyllama")
        final_state = {
            "attack_history": [make_attack_record(1, score=2.0)],
            "successful_attacks": [],
            "iteration_count": 5,
            "error": "max_iterations_reached",
        }
        with patch.object(dashboard, "_live"):
            dashboard.print_final_summary(final_state)

    def test_print_run_summary_does_not_raise(self):
        from reports.terminal_dashboard import print_run_summary
        final_state = {
            "attack_history": [make_attack_record(1, score=9.0)],
            "successful_attacks": [make_attack_record(1, score=9.0)],
            "iteration_count": 2,
            "error": None,
        }
        # Should not raise
        print_run_summary(final_state, "tinyllama", "test objective")
