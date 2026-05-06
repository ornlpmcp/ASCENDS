"""CLI entrypoint that registers ASCENDS command modules."""

from ascends.cli_app import app

# Import command modules for their Typer registration side effects.
from ascends import cli_correlation as _cli_correlation  # noqa: F401
from ascends import cli_gui as _cli_gui  # noqa: F401
from ascends import cli_parity as _cli_parity  # noqa: F401
from ascends import cli_predict as _cli_predict  # noqa: F401
from ascends import cli_shap as _cli_shap  # noqa: F401
from ascends import cli_train as _cli_train  # noqa: F401


if __name__ == "__main__":
    app()
