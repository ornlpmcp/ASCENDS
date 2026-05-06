"""GUI launch command for the ASCENDS CLI."""

import threading
import time
import urllib.request
import webbrowser

import typer
import uvicorn

from ascends.cli_app import app


@app.command()
def gui(
    port: int = typer.Option(7777, help="Port number to run the ASCENDS GUI"),
    reload: bool = typer.Option(True, help="Auto-reload on code changes (dev mode)"),
    open_browser: bool = typer.Option(False, help="Open the GUI URL in the default browser"),
):
    """Launch the ASCENDS GUI (FastAPI-based ascends_server).

    Example:
      uv run ascends gui
    """
    url = f"http://127.0.0.1:{port}"
    if open_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    typer.echo(f"🚀 Launching ASCENDS GUI on {url} (reload={reload})")
    uvicorn.run("ascends_server:app", host="127.0.0.1", port=port, reload=reload)


def _open_browser_when_ready(url: str, timeout: float = 30.0) -> None:
    """Open the local GUI after the server is accepting requests."""
    deadline = time.time() + timeout
    health_url = f"{url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0):
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(0.5)
    webbrowser.open(url)
