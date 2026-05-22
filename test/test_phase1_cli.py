"""Phase 1 CLI contract tests."""

from typer.testing import CliRunner

from ascends.cli import app


def test_train_help_does_not_expose_dead_tuning_options() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "--tune" not in result.output
    assert "--tune-trials" not in result.output
