"""Unit tests for :mod:`synthetic_qa_dataset.eval_synthetic_dataset`."""

import json

import pytest

from synthetic_qa_dataset.eval_synthetic_dataset import (
    build_test_case,
    latest_dataset,
    load_data,
    save_results,
)


def _write_dataset(tmp_path, name, data) -> None:
    """Write ``data`` as JSON to ``tmp_path/name``."""
    (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")


def test_latest_dataset_returns_last_sorted(tmp_path):
    """``latest_dataset`` must return the most recent file by name order."""
    for name in (
        "synthetic_qa_20260101_000000.json",
        "synthetic_qa_20260102_000000.json",
    ):
        _write_dataset(tmp_path, name, [])
    assert latest_dataset(tmp_path).name == "synthetic_qa_20260102_000000.json"


def test_latest_dataset_ignores_other_files(tmp_path):
    """``latest_dataset`` must only consider ``synthetic_qa_*.json`` files."""
    _write_dataset(tmp_path, "notes.txt", "x")
    with pytest.raises(FileNotFoundError):
        latest_dataset(tmp_path)


def test_load_data_returns_list(tmp_path):
    """``load_data`` must return the JSON list content."""
    _write_dataset(tmp_path, "data.json", [{"input": "Q"}])
    assert load_data(tmp_path / "data.json") == [{"input": "Q"}]


def test_load_data_missing_file(tmp_path):
    """A missing file must raise ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.json")


def test_load_data_not_a_list(tmp_path):
    """A non-list document must raise ``TypeError``."""
    _write_dataset(tmp_path, "data.json", {"input": "Q"})
    with pytest.raises(TypeError):
        load_data(tmp_path / "data.json")


def test_build_test_case_maps_expected_output():
    """``build_test_case`` must set actual_output to expected_output."""
    case = build_test_case({"input": "Q", "expected_output": "A", "context": ["ctx"]})
    assert case.input == "Q"
    assert case.actual_output == "A"
    assert case.context == ["ctx"]


def test_build_test_case_handles_missing_fields():
    """``build_test_case`` must tolerate records with missing fields."""
    case = build_test_case({})
    assert case.input == ""
    assert case.actual_output == ""
    assert case.context == []


def test_save_results_writes_file(tmp_path):
    """``save_results`` must write ``eval_data_<dataset_name>.json``."""
    results = [{"hallucination_metric": {"score": 0.1, "reason": "ok"}}]
    output = save_results(results, "ds.json", tmp_path)
    assert output.name == "eval_data_ds.json.json"
    assert json.loads(output.read_text(encoding="utf-8")) == results
