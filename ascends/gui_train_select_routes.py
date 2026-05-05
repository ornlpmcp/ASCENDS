"""Train-tab column selection route for the ASCENDS FastAPI GUI."""

from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


def create_train_select_router(
    *,
    templates: Jinja2Templates,
    load_manifest: Callable[[str], dict[str, Any]],
    save_manifest: Callable[[str, dict[str, Any]], None],
    list_saved_runs: Callable[[], list[dict[str, Any]]],
) -> APIRouter:
    """Create the Train tab route that manages selected inputs and target."""
    router = APIRouter()

    @router.post("/train/select", response_class=HTMLResponse)
    async def train_select(
        request: Request,
        ws_id: str = Form(...),
        action: str = Form(...),
        columns: Optional[list[str]] = Form(None),
        rm_inputs: Optional[list[str]] = Form(None),
        target_choice: Optional[str] = Form(None),
    ) -> HTMLResponse:
        """Handle Train tab selection state (columns/inputs/target)."""
        mf = load_manifest(ws_id)
        if not mf:
            return templates.TemplateResponse(
                "train.html",
                {
                    "request": request,
                    "ws_id": ws_id,
                    "train_error": "Invalid session. Please upload/select data from Correlation tab first.",
                    "saved_runs": list_saved_runs(),
                },
            )

        all_columns: list[str] = list(mf.get("columns", []))
        selected = set(mf.get("selected", []))
        inputs = set(mf.get("inputs", []))
        target = mf.get("target")

        chosen_cols = columns or []
        to_remove = rm_inputs or []

        if action == "select_all":
            selected = set(all_columns)
        elif action == "select_none":
            selected = set()
        elif action == "to_inputs":
            selected = set(chosen_cols)
            for column in chosen_cols:
                if column in all_columns:
                    inputs.add(column)
            if target in inputs:
                inputs.discard(target)
        elif action == "remove_inputs":
            for column in to_remove:
                inputs.discard(column)
        elif action == "set_target":
            selected = set(chosen_cols)
            if target_choice and target_choice in all_columns:
                target = target_choice
                if target in inputs:
                    inputs.discard(target)

        ordered_inputs = sorted(inputs, key=lambda column: all_columns.index(column)) if all_columns else list(inputs)
        ordered_selected = sorted(selected, key=lambda column: all_columns.index(column)) if all_columns else list(selected)

        mf["inputs"] = ordered_inputs
        mf["target"] = target
        mf["selected"] = ordered_selected
        save_manifest(ws_id, mf)

        ctx: dict[str, Any] = {
            "request": request,
            "ws_id": ws_id,
            "csv_path": mf.get("csv_path"),
            "all_columns": all_columns,
            "inputs": ordered_inputs,
            "target": target,
            "selected": ordered_selected,
            "saved_runs": list_saved_runs(),
        }
        return templates.TemplateResponse("train.html", ctx)

    return router
