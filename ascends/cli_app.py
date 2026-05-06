"""Shared Typer application wiring for ASCENDS CLI."""

import typer
import typer.main as _typer_main


class _CmdOrder(_typer_main.TyperGroup):
    """Keep the most common commands first in help output."""

    DESIRED = ["gui", "correlation", "train", "parity-plot", "shap", "predict"]

    def list_commands(self, ctx):
        names = list(self.commands.keys())
        ordered = [name for name in self.DESIRED if name in names]
        remainder = [name for name in names if name not in self.DESIRED]
        return ordered + remainder


app = typer.Typer(cls=_CmdOrder, no_args_is_help=True, add_completion=False)
