"""Project directory paths (independent of the CWD)."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "docs"
DATASETS_DIR = ROOT_DIR / "datasets"
EVALS_DIR = ROOT_DIR / "evals"
