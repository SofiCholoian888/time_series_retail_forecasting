"""Create a source ZIP using a strict allowlist, excluding environments and data."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT_FILES = {
    "README.md",
    "START_HERE_RU.md",
    "AUDIT_RU.md",
    "model_card.md",
    "Model_BPMN.md",
    "Model_retrain_BPMN.md",
    "pyproject.toml",
    "poetry.lock",
    "poetry.toml",
    ".gitignore",
    ".pre-commit-config.yaml",
}
DIRECTORIES = {"src", "tests", "scripts", "configs", "docs", "notebooks", ".github"}
SUFFIXES = {".py", ".md", ".json", ".ipynb", ".yml", ".yaml", ".toml"}


def export(root: Path, target: Path) -> int:
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if any(
            part in {"__pycache__", ".ipynb_checkpoints", ".venv", ".git"} for part in rel.parts
        ):
            continue
        if (
            rel.as_posix() in ROOT_FILES
            or (rel.parts[0] in DIRECTORIES and path.suffix in SUFFIXES)
            or (rel.parts[:2] == ("reports", "figures") and path.suffix == ".png")
        ):
            paths.append(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, Path("time_series_retail_forecasting") / path.relative_to(root))
    return len(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/project_source.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    count = export(root, args.output.resolve())
    print(f"Saved {count} source files: {args.output.resolve()}")
