from __future__ import annotations

import zipfile
from pathlib import Path

from examples.opendeepthink_fixed_runner import (
    OFFICIAL_AC_PATH,
    compact_eval_result,
    evaluation_status,
    extract_official_ac_source,
    summarize_eval_population,
    summarize_evaluation,
)


def test_extract_official_ac_source(tmp_path: Path) -> None:
    package = tmp_path / "polygon_package.zip"
    with zipfile.ZipFile(package, "w") as zf:
        zf.writestr(OFFICIAL_AC_PATH, "int main() { return 0; }\n")
    destination = tmp_path / "out" / "official.cpp"

    result = extract_official_ac_source(package, destination)

    assert result["path"] == str(destination)
    assert result["path_in_package"] == OFFICIAL_AC_PATH
    assert destination.read_text() == "int main() { return 0; }\n"


def test_extract_official_ac_source_missing_package(tmp_path: Path) -> None:
    result = extract_official_ac_source(tmp_path / "missing.zip", tmp_path / "official.cpp")

    assert result["path_in_package"] == OFFICIAL_AC_PATH
    assert "Polygon package not found" in result["error"]


def test_extract_official_ac_source_missing_ac_file(tmp_path: Path) -> None:
    package = tmp_path / "polygon_package.zip"
    with zipfile.ZipFile(package, "w") as zf:
        zf.writestr("solutions/wa.cpp", "int main() { return 1; }\n")

    result = extract_official_ac_source(package, tmp_path / "official.cpp")

    assert result["path_in_package"] == OFFICIAL_AC_PATH
    assert "not found" in result["error"]


def test_evaluation_status() -> None:
    assert evaluation_status({"status": "passed", "returncode": 1}) == "passed"
    assert evaluation_status({"error": "missing"}) == "error"
    assert evaluation_status({"returncode": 2}) == "error"
    assert evaluation_status({"returncode": 0}) == "unknown"


def test_compact_eval_result() -> None:
    result = compact_eval_result({
        "status": "failed",
        "tests_passed": 3,
        "tests_total": 7,
        "returncode": 0,
        "path_in_package": OFFICIAL_AC_PATH,
        "failures": [{"large": "omitted"}],
    })

    assert result == {
        "status": "failed",
        "tests_passed": 3,
        "tests_total": 7,
        "returncode": 0,
        "path_in_package": OFFICIAL_AC_PATH,
    }


def test_summarize_eval_population() -> None:
    summary = summarize_eval_population([
        {"status": "passed"},
        {"status": "failed"},
        {"returncode": 1},
        {"error": "private judge dir not configured"},
        {"returncode": 0},
    ])

    assert summary["total"] == 5
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["statuses"] == {
        "error": 2,
        "failed": 1,
        "passed": 1,
        "unknown": 1,
    }


def test_summarize_evaluation() -> None:
    summary = summarize_evaluation({
        "selected": {"status": "failed", "tests_passed": 0, "tests_total": 24, "returncode": 0},
        "gen0": [{"status": "passed"}, {"status": "failed"}, {"returncode": 1}],
        "official_ac": {
            "status": "passed",
            "tests_passed": 24,
            "tests_total": 24,
            "returncode": 0,
            "path_in_package": OFFICIAL_AC_PATH,
        },
    })

    assert summary["selected"]["status"] == "failed"
    assert summary["selected"]["tests_total"] == 24
    assert summary["gen0"]["statuses"] == {"error": 1, "failed": 1, "passed": 1}
    assert summary["official_ac_sanity"] == {
        "status": "passed",
        "tests_passed": 24,
        "tests_total": 24,
        "returncode": 0,
        "path_in_package": OFFICIAL_AC_PATH,
    }
