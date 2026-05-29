"""Regression tests for GUI navigation and sample task defaults."""

from __future__ import annotations

from pathlib import Path

from ascends.gui_correlation_routes import _current_task_from_manifest


def test_home_tab_is_rendered_before_workflow_steps() -> None:
    template = Path("templates/base.html").read_text(encoding="utf-8")

    home_idx = template.index('class="tab tab-home')
    workflow_idx = template.index("{% for step in workflow_steps %}")
    assert home_idx < workflow_idx
    assert 'href="/"' in template
    assert "path == '/'" in template


def test_correlation_task_default_prefers_manifest_classification() -> None:
    assert _current_task_from_manifest({"task": "c", "train_params": {"task": "c"}}) == "c"


def test_correlation_task_default_falls_back_to_regression() -> None:
    assert _current_task_from_manifest({}) == "r"


def test_correlation_template_uses_current_task_for_radio_defaults() -> None:
    template = Path("templates/correlation.html").read_text(encoding="utf-8")

    assert "current_task != 'c'" in template
    assert "current_task == 'c'" in template
