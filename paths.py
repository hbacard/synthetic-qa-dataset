"""Chemins des répertoires du projet (indépendants du CWD)."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT_DIR / "docs"
DATASETS_DIR = ROOT_DIR / "datasets"
EVALS_DIR = ROOT_DIR / "evals"
