"""Regression tests for GUI navigation and sample task defaults."""

from __future__ import annotations

from pathlib import Path


def test_home_tab_is_rendered_before_workflow_steps() -> None:
    template = Path("templates/base.html").read_text(encoding="utf-8")

    home_idx = template.index('class="tab tab-home')
    workflow_idx = template.index("{% for step in workflow_steps %}")
    assert home_idx < workflow_idx
    assert 'href="/"' in template
    assert "path == '/'" in template
