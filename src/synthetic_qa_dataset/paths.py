"""Project directory paths (independent of the CWD)."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT_DIR / "corpus"
DATASETS_DIR = ROOT_DIR / "datasets"
EVALS_DIR = ROOT_DIR / "evals"
