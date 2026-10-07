"""Unit tests for :mod:`synthetic_qa_dataset.paths`."""

from pathlib import Path

from synthetic_qa_dataset import paths


def test_root_dir_is_project_root():
    """ROOT_DIR must be the repository root (two levels above paths.py)."""
    expected = Path(__file__).resolve().parents[1]
    assert paths.ROOT_DIR == expected


def test_derived_dirs_are_below_root_dir():
    """DOCS_DIR, DATASETS_DIR and EVALS_DIR must sit below ROOT_DIR."""
    assert paths.DOCS_DIR == paths.ROOT_DIR / "docs"
    assert paths.DATASETS_DIR == paths.ROOT_DIR / "datasets"
    assert paths.EVALS_DIR == paths.ROOT_DIR / "evals"


def test_derived_dirs_are_absolute():
    """All paths must be absolute so they do not depend on the CWD."""
    for value in (paths.ROOT_DIR, paths.DOCS_DIR, paths.DATASETS_DIR, paths.EVALS_DIR):
        assert value.is_absolute()
