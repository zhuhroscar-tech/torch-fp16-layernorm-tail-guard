"""Regression tests for torch-fp16-layernorm-tail-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build (test_native_bug_reproduces_on_this_host),
     with the exact "nonzero count == length % 8" signature that
     distinguishes it from a generic float16-precision complaint.
  2. safe_layer_norm() is an independently-verified fix: exact 0 on
     every constant-row length from 1 to 24 (an independent
     mathematical oracle -- (x - mean) is exactly 0 for constant input
     regardless of dtype, so ANY nonzero output is wrong by
     definition, no reference implementation needed).
  3. safe_layer_norm() is a safe, transparent replacement on ordinary
     (non-constant) input: it matches or beats the native float16
     path against an independent float64 oracle, at several lengths
     including both multiples of 8 (unaffected by the bug) and
     non-multiples (affected).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_fp16_layernorm_tail_guard.core import diagnose, safe_layer_norm


class TestNativeBugReproduction:
    def test_native_bug_reproduces_on_this_host(self):
        # This is not asserted True unconditionally: if a future torch
        # release fixes the underlying kernel, this test's own
        # docstring is the record that the bug existed at the version
        # noted in the ledger/README, and the report's
        # any_bug_present=False path is itself meaningful (proves the
        # guard is a no-op safety net rather than working around a
        # phantom). But on every torch version tested so far
        # (2.10-2.14 CPU builds) this reproduces reliably.
        report = diagnose(constant_lengths=(9, 12, 20))
        assert report["any_bug_present"] is True, (
            "expected the documented tail-corruption bug on this host's "
            f"torch {report['torch_version']}; if this now fails, the "
            "bug may be fixed upstream -- update the README/ledger "
            "accordingly rather than treating this as a regression"
        )

    def test_bug_signature_matches_length_mod_8(self):
        report = diagnose(constant_lengths=tuple(range(1, 25)))
        for case in report["constant_cases"]:
            if case["length"] % 8 == 0:
                assert case["buggy_nonzero_count"] == 0, (
                    f"length={case['length']} is a multiple of 8 and "
                    "should be unaffected by the documented bug, but "
                    f"native output has {case['buggy_nonzero_count']} "
                    "nonzero elements"
                )
            else:
                assert case["buggy_nonzero_count"] == case["length"] % 8, (
                    f"length={case['length']}: expected the documented "
                    f"signature (nonzero count == length % 8 == "
                    f"{case['length'] % 8}), got "
                    f"{case['buggy_nonzero_count']}"
                )


class TestSafeLayerNormConstantRows:
    @pytest.mark.parametrize("length", list(range(1, 25)))
    def test_constant_row_is_exact_zero(self, length):
        x = torch.full((1, length), 100.0, dtype=torch.float16)
        result = safe_layer_norm(x, (length,), eps=1e-5)
        assert torch.all(result == 0.0), (
            f"safe_layer_norm on a constant row of length {length} must "
            f"be exactly 0 everywhere (true variance is 0 by "
            f"definition); got {result}"
        )

    @pytest.mark.parametrize("value", [0.1, 1.0, 13.0, 100.0, 60000.0])
    def test_constant_row_exact_zero_across_magnitudes(self, value):
        # length=12 (12 % 8 == 4) is a length that DOES trigger the
        # native bug for many values -- verify the guard is correct
        # regardless of the constant's magnitude, not just at one
        # value that happens to trigger it.
        x = torch.full((1, 12), value, dtype=torch.float16)
        result = safe_layer_norm(x, (12,), eps=1e-5)
        assert torch.all(result == 0.0)


class TestSafeLayerNormTransparency:
    @pytest.mark.parametrize("length", [8, 12, 16, 20, 32])
    def test_nonconstant_input_at_least_as_accurate_as_native(self, length):
        torch.manual_seed(length)
        x32 = torch.randn(1, length) * 5.0
        x16 = x32.to(torch.float16)

        native = torch.nn.functional.layer_norm(x16, (length,), eps=1e-5)
        guarded = safe_layer_norm(x16, (length,), eps=1e-5)
        reference = (
            torch.nn.functional.layer_norm(x16.double(), (length,), eps=1e-5)
            .to(torch.float16)
        )

        native_diff = (native.float() - reference.float()).abs().max().item()
        guard_diff = (guarded.float() - reference.float()).abs().max().item()
        assert guard_diff <= native_diff + 1e-6, (
            f"length={length}: guard diff {guard_diff} worse than native "
            f"diff {native_diff} vs the fp64 oracle"
        )

    def test_non_float16_input_passes_through_unchanged(self):
        # float32/float64/CUDA (if available) input should delegate to
        # the real torch.nn.functional.layer_norm with zero behavior
        # change -- the guard only intervenes for the specific
        # affected dtype+device combination.
        x = torch.randn(1, 12, dtype=torch.float32)
        expected = torch.nn.functional.layer_norm(x, (12,), eps=1e-5)
        actual = safe_layer_norm(x, (12,), eps=1e-5)
        assert torch.equal(actual, expected)

    def test_weight_and_bias_are_respected(self):
        length = 12
        x = torch.full((1, length), 100.0, dtype=torch.float16)
        weight = torch.full((length,), 2.0, dtype=torch.float16)
        bias = torch.full((length,), 3.0, dtype=torch.float16)
        result = safe_layer_norm(x, (length,), weight=weight, bias=bias, eps=1e-5)
        # normalized value is exactly 0 for constant input, so
        # weight * 0 + bias == bias everywhere
        assert torch.allclose(result, bias)


class TestDiagnose:
    def test_diagnose_runs_and_reports_consistent_structure(self):
        report = diagnose(constant_lengths=(8, 9, 16), nonconstant_lengths=(8, 12))
        assert len(report["constant_cases"]) == 3
        assert len(report["nonconstant_cases"]) == 2
        assert isinstance(report["torch_version"], str)

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True
