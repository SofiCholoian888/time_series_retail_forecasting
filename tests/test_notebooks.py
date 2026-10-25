"""Validate notebook format and Python syntax without claiming full execution."""

import ast
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path", sorted((ROOT / "notebooks").rglob("*.ipynb")), ids=lambda p: p.name
)
def test_notebook_schema_and_cell_syntax(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs
