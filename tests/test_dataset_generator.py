"""Unit tests for :mod:`synthetic_qa_dataset.dataset_generator`."""

import json
from datetime import UTC, datetime

import pytest
from deepeval.dataset.golden import Golden

from synthetic_qa_dataset.dataset_generator import (
    list_document_paths,
    record_to_dict,
    save_dataset,
)


def test_list_document_paths_sorted(tmp_path):
    """``list_document_paths`` must return files sorted by name."""
    for name in ("b.md", "a.md", "c.txt"):
        (tmp_path / name).write_text("content", encoding="utf-8")
    paths = list_document_paths(tmp_path)
    assert [p.name for p in paths] == ["a.md", "b.md", "c.txt"]


def test_list_document_paths_missing_dir(tmp_path):
    """A missing directory must raise ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        list_document_paths(tmp_path / "does-not-exist")


def test_list_document_paths_empty_dir(tmp_path):
    """An empty directory must raise ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        list_document_paths(tmp_path)


def test_record_to_dict(tmp_path):
    """``record_to_dict`` must flatten a ``Golden`` into a plain dict."""
    golden = Golden(
        input="Question?",
        expected_output="Expected answer.",
        context=["ctx-a", "ctx-b"],
    )
    record = record_to_dict(golden)
    assert record["input"] == "Question?"
    assert record["expected_output"] == "Expected answer."
    assert record["context"] == ["ctx-a", "ctx-b"]
    assert record["actual_output"] is None


def test_save_dataset_writes_timestamped_file(tmp_path):
    """``save_dataset`` must write a timestamped JSON list of records."""
    goldens = [Golden(input="Q1", expected_output="A1")]
    output = save_dataset(goldens, tmp_path)

    assert output.parent == tmp_path
    assert output.name.startswith("synthetic_qa_")
    assert output.name.endswith(".json")

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(saved, list)
    assert len(saved) == 1
    assert saved[0]["input"] == "Q1"
    assert saved[0]["expected_output"] == "A1"


def test_save_dataset_creates_missing_dir(tmp_path):
    """``save_dataset`` must create the output directory if absent."""
    output_dir = tmp_path / "nested" / "datasets"
    output = save_dataset([Golden(input="Q")], output_dir)
    assert output.is_file()
    assert output.name[13:28] == datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
