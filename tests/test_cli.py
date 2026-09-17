"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is installed.
Mirrors the test_cli.py pattern already used across the fleet (e.g.
rng-leak-audit, causality-audit) for a repo that had none.

The branch-coverage tests below mock ``core.diagnose`` so every CLI
message path (torch-unavailable, the bug/no-bug branches including the
nested bug_signature_confirmed line, and a guard-mismatch "fail" line)
is exercised deterministically, regardless of whether this host's
installed torch build happens to reproduce the underlying bug. Real
fleet-wide gap found by pytest-cov inspection: cli.py sat at 82%
coverage with the negative/nested branches of each status line (lines
43-49, 69, 81) and the ``__main__`` guard (line 113) never hit by any
existing test."""
from __future__ import annotations

import json
import runpy
import sys

import pytest

from torch_fp16_layernorm_tail_guard import core
from torch_fp16_layernorm_tail_guard.cli import main


def _fake_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "constant_cases": [],
        "nonconstant_cases": [],
        "any_bug_present": False,
        "bug_signature_confirmed": False,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-fp16-layernorm-tail-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert "guard_fully_correct" in report
    assert code in (0, 1)


def test_json_exit_code_matches_guard_fully_correct(capsys):
    pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert code == (0 if report["guard_fully_correct"] else 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_text_output_reports_constant_and_nonconstant_cases(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "constant-row cases" in out
    assert "non-constant accuracy cases" in out


def test_torch_unavailable_json_mode_reports_error_and_exit_2(monkeypatch, capsys):
    """cli.py lines 44-45: TorchUnavailableError + --json emits a JSON
    error object and exits 2, regardless of torch install state."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis"}
    assert code == 2


def test_torch_unavailable_text_mode_reports_fail_headline_and_exit_2(monkeypatch, capsys):
    """cli.py lines 46-49: TorchUnavailableError in text mode prints a
    'fail' status headline (not the JSON branch) and exits 2."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch unavailable: torch is required for diagnosis" in out
    assert "[X]" in out
    assert code == 2


def test_no_bug_present_prints_info_line(monkeypatch, capsys):
    """cli.py line 69: the 'info' (not 'warn') branch when the host's
    torch build does NOT reproduce the tail-corruption bug -- and the
    nested bug_signature_confirmed line (lines 60-67) is skipped
    entirely since it lives inside the any_bug_present branch."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report())
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "bug NOT reproduced on this host's installed torch build" in out
    assert "bug reproduced on this host's installed torch build" not in out
    assert "confirmed signature" not in out


def test_bug_present_without_signature_confirmed_prints_warn_only(monkeypatch, capsys):
    """cli.py line 59 true / line 60 false: bug reproduced but the
    nested bug_signature_confirmed line must NOT print."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(any_bug_present=True))
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "bug reproduced on this host's installed torch build" in out
    assert "confirmed signature" not in out


def test_bug_present_with_signature_confirmed_prints_nested_info_line(monkeypatch, capsys):
    """cli.py lines 59-67: both any_bug_present AND
    bug_signature_confirmed true prints the nested 'info' line."""
    monkeypatch.setattr(
        core, "diagnose", lambda: _fake_report(any_bug_present=True, bug_signature_confirmed=True)
    )
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "bug reproduced on this host's installed torch build" in out
    assert "confirmed signature: nonzero count == length % 8 on the trigger case" in out


def test_guard_mismatch_prints_fail_line_and_exit_1(monkeypatch, capsys):
    """cli.py line 81 + 109: guard_fully_correct=False prints the
    'fail' headline (not the 'ok' one) and the process exits 1."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "guard did NOT produce a correct result on at least one case" in out
    assert "is exact-0 on every constant-row case" not in out
    assert code == 1


def test_module_entry_point_runs_main_and_exits_with_its_code(monkeypatch):
    """cli.py line 113 (``if __name__ == "__main__": sys.exit(main())``):
    running the module as a script must invoke main() and propagate its
    return code via SystemExit, not just be dead code."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    monkeypatch.setattr(sys, "argv", ["torch-fp16-layernorm-tail-guard", "--no-color"])
    monkeypatch.delitem(sys.modules, "torch_fp16_layernorm_tail_guard.cli", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("torch_fp16_layernorm_tail_guard.cli", run_name="__main__")
    assert exc_info.value.code == 1
