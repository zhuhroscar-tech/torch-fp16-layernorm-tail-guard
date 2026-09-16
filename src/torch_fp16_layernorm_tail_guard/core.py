"""torch-fp16-layernorm-tail-guard core: detect and safely fix a real
torch CPU float16 ``torch.nn.functional.layer_norm`` bug where an
exact-constant row (true variance == 0) whose normalized length is NOT
a multiple of 8 gets nonzero, wrong values in the last
``length % 8`` output elements, instead of the mathematically
guaranteed exact 0 for every element.

Reproduced from scratch on this host (see README for the exact
commands and the live-source verification trail):

  torch.nn.functional.layer_norm(
      torch.full((1, 12), 100.0, dtype=torch.float16), (12,), eps=1e-5,
  )
  # macOS (Accelerate-linked build): [0,0,0,0,0,0,0,0, 0.0977,0.0977,0.0977,0.0977]
  # Linux (2.14.0+cu130, CPU path):  [0.000977]*12 (every element wrong,
  #                                   uniformly, by a small constant amount)
  # every true value must be exactly 0.0 (constant input has zero
  # variance by definition -- (x - mean) is exactly 0 everywhere).

The failure shape is CI-confirmed to differ by build/platform, which
matters for how this is described: on this macOS development host
(Accelerate-linked CPU build), it is confined to the input length's
remainder modulo 8 (the AVX/SIMD lane width torch's CPU float16
layer_norm kernel vectorizes over) -- every length from 1-39 shows
exactly ``length % 8`` wrong elements, with exact multiples of 8
always correct. On GitHub Actions' ubuntu-latest runner (torch
2.14.0+cu130 wheel, still running the CPU kernel since no GPU is
present), ALL elements come back wrong instead, uniformly, by a small
fixed amount (~0.001) independent of length. Both are real, both are
wrong by the same unambiguous standard (exact-zero variance means the
true output is exactly 0.0, not "close to 0"), and both are fixed by
the same float32-upcast guard -- but this package does NOT claim a
single universal root cause or a single wrong-element-count signature
across all platforms; the tests and diagnose() output treat the
per-platform signature as informational, and only assert the
underlying invariant (any constant row should be all-zero) and the
guard's correctness, which hold regardless of which shape the native
bug takes on a given host. float32 and float64 LayerNorm on the
identical input are correct at every length tested on every platform
tested, and float16 LayerNorm on NON-constant input (genuine nonzero
variance) matches a float32-upcast reference to machine precision at
every length tested -- the bug is specific to the exact-zero-variance
codepath.

This package does not wait for or depend on an upstream fix (as of
this writing, no existing PyTorch issue was found describing this
specific tail-corruption pattern for the exact-zero-variance case --
see README for the distinct, previously-filed float16 LayerNorm
issues this is NOT a duplicate of). It works around the bug entirely
at the call site with a simple, independently-testable strategy that
mirrors the standard real-world mitigation already used by production
LayerNorm/RMSNorm implementations (e.g. HF Transformers casting the
reduction to float32 before the sqrt): compute the normalization at
float32 precision internally and cast the result back to float16,
which both avoids the buggy kernel path entirely and is numerically at
least as accurate as the native float16 path for every input, not just
the constant-input trigger case.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported. Kept as a distinct type so
    callers can distinguish "torch isn't installed" from an actual
    diagnostic failure."""


def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


def _is_affected_dtype_and_device(torch_module, input_tensor) -> bool:
    return (
        input_tensor.dtype == torch_module.float16
        and input_tensor.device.type == "cpu"
    )


def safe_layer_norm(
    input,
    normalized_shape,
    weight=None,
    bias=None,
    eps: float = 1e-5,
):
    """Drop-in replacement for ``torch.nn.functional.layer_norm`` that
    avoids the CPU float16 exact-constant-row tail-corruption bug.

    For any input that is float16 on CPU, the normalization is computed
    internally at float32 precision (upcasting input/weight/bias, then
    casting the result back to float16), entirely sidestepping the
    buggy kernel path. This is a safe, transparent replacement: on
    finite, non-triggering input it matches the native float16 path to
    within ordinary float16 rounding, and it is *more* accurate on the
    documented trigger case (constant rows with length % 8 != 0),
    where the native path is simply wrong.

    All other dtype/device combinations delegate to the real
    ``torch.nn.functional.layer_norm`` unchanged.
    """
    torch_module = _import_torch()
    import torch.nn.functional as F

    if not _is_affected_dtype_and_device(torch_module, input):
        return F.layer_norm(input, normalized_shape, weight, bias, eps)

    input_f32 = input.float()
    weight_f32 = weight.float() if weight is not None else None
    bias_f32 = bias.float() if bias is not None else None
    result_f32 = F.layer_norm(input_f32, normalized_shape, weight_f32, bias_f32, eps)
    return result_f32.to(input.dtype)


@dataclasses.dataclass
class ConstantRowCase:
    length: int
    buggy_nonzero_count: int
    expected_nonzero_count: int
    buggy_matches_tail_signature: bool
    buggy_matches_uniform_signature: bool
    guard_nonzero_count: int
    guard_is_correct: bool


def _run_constant_row_case(torch_module, length: int, value: float = 100.0) -> ConstantRowCase:
    import torch.nn.functional as F

    x = torch_module.full((1, length), value, dtype=torch_module.float16)
    buggy = F.layer_norm(x, (length,), eps=1e-5)
    buggy_nonzero = int((buggy != 0).sum().item())

    guarded = safe_layer_norm(x, (length,), eps=1e-5)
    guard_nonzero = int((guarded != 0).sum().item())

    return ConstantRowCase(
        length=length,
        buggy_nonzero_count=buggy_nonzero,
        expected_nonzero_count=0,
        # Two distinct failure shapes have been observed in CI across
        # platforms/builds (see core.py module docstring): a
        # tail-only pattern (macOS/Accelerate: nonzero count ==
        # length % 8) and a uniform whole-row pattern (Linux
        # cu130 wheel CPU path: every element wrong). Both fields are
        # recorded so diagnose()/the CLI can report which shape this
        # host exhibits without either being asserted as the only
        # valid one.
        buggy_matches_tail_signature=(buggy_nonzero == length % 8),
        buggy_matches_uniform_signature=(buggy_nonzero == length),
        guard_nonzero_count=guard_nonzero,
        guard_is_correct=(guard_nonzero == 0),
    )


@dataclasses.dataclass
class NonConstantAccuracyCase:
    length: int
    max_abs_diff_native_vs_reference: float
    max_abs_diff_guard_vs_reference: float
    guard_at_least_as_accurate: bool


def _run_nonconstant_accuracy_case(torch_module, length: int, seed: int) -> NonConstantAccuracyCase:
    import torch.nn.functional as F

    gen = torch_module.Generator().manual_seed(seed)
    x32 = torch_module.randn(1, length, generator=gen, dtype=torch_module.float32) * 5.0
    x16 = x32.to(torch_module.float16)

    native = F.layer_norm(x16, (length,), eps=1e-5)
    guarded = safe_layer_norm(x16, (length,), eps=1e-5)
    # Independent oracle: float64 computation from the exact float16
    # input values (no further precision loss beyond the initial
    # fp16 cast), used as ground truth for both comparisons.
    x64 = x16.double()
    reference = F.layer_norm(x64, (length,), eps=1e-5).to(torch_module.float16)

    native_diff = float((native.float() - reference.float()).abs().max().item())
    guard_diff = float((guarded.float() - reference.float()).abs().max().item())

    return NonConstantAccuracyCase(
        length=length,
        max_abs_diff_native_vs_reference=native_diff,
        max_abs_diff_guard_vs_reference=guard_diff,
        guard_at_least_as_accurate=(guard_diff <= native_diff + 1e-6),
    )


def diagnose(
    constant_lengths: Sequence[int] = tuple(range(1, 25)),
    nonconstant_lengths: Sequence[int] = (8, 12, 16, 20, 32),
) -> Dict[str, Any]:
    """Reproduce the bug from scratch against the currently installed
    torch build, at several row lengths, and verify the guard's
    correctness both on the trigger case (constant rows) and on
    ordinary non-constant input (transparency / no regression). Never
    trusts a cached/prior result -- every call re-runs the actual
    repro."""
    torch_module = _import_torch()

    constant_cases: List[ConstantRowCase] = [
        _run_constant_row_case(torch_module, n) for n in constant_lengths
    ]
    nonconstant_cases: List[NonConstantAccuracyCase] = [
        _run_nonconstant_accuracy_case(torch_module, n, seed=n)
        for n in nonconstant_lengths
    ]

    any_bug_present = any(c.buggy_nonzero_count != 0 for c in constant_cases)
    bug_signature_confirmed = any(
        c.buggy_nonzero_count != 0
        and (c.buggy_matches_tail_signature or c.buggy_matches_uniform_signature)
        for c in constant_cases
    )
    guard_fully_correct = all(c.guard_is_correct for c in constant_cases) and all(
        c.guard_at_least_as_accurate for c in nonconstant_cases
    )

    return {
        "torch_version": torch_module.__version__,
        "constant_cases": [dataclasses.asdict(c) for c in constant_cases],
        "nonconstant_cases": [dataclasses.asdict(c) for c in nonconstant_cases],
        "any_bug_present": any_bug_present,
        "bug_signature_confirmed": bug_signature_confirmed,
        "guard_fully_correct": guard_fully_correct,
    }
