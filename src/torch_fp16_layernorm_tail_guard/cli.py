"""Command-line interface: run the from-scratch diagnosis of the CPU
float16 LayerNorm constant-row tail-corruption bug against the
currently installed torch build, using the shared semantic-color
design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-fp16-layernorm-tail-guard",
        description=(
            "Diagnose whether the currently installed torch build's CPU "
            "float16 nn.LayerNorm produces nonzero garbage in the tail "
            "of an exact-constant row whose length is not a multiple of "
            "8 (instead of the mathematically guaranteed exact 0), and "
            "verify safe_layer_norm() avoids it -- never trusts a cached "
            "or previously-reported result, always re-runs the repro on "
            "THIS host's actual installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-fp16-layernorm-tail-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_bug_present"]:
        print(status_headline(style, "warn", "bug reproduced on this host's installed torch build"))
        if report["bug_signature_confirmed"]:
            print(
                status_headline(
                    style,
                    "info",
                    "confirmed signature: nonzero count == length % 8 on the trigger case",
                )
            )
    else:
        print(
            status_headline(
                style,
                "info",
                "bug NOT reproduced on this host's installed torch build (fixed upstream, or this "
                "host's kernel dispatch path differs)",
            )
        )

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_layer_norm() is exact-0 on every constant-row case and >= as accurate elsewhere"))
    else:
        print(status_headline(style, "fail", "guard did NOT produce a correct result on at least one case"))

    section("constant-row cases (length -> nonzero output count; expected 0)")
    for c in report["constant_cases"]:
        flag = "BUG" if c["buggy_nonzero_count"] != 0 else "ok"
        guard_flag = "guard-ok" if c["guard_is_correct"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"length={c['length']:3d}",
                    f"native_nonzero={c['buggy_nonzero_count']}  {flag:4s}  {guard_flag}",
                )
            ]
        )

    section("non-constant accuracy cases (native vs guard max abs diff from fp64 oracle)")
    for c in report["nonconstant_cases"]:
        flag = "ok" if c["guard_at_least_as_accurate"] else "REGRESSION"
        print_fields(
            [
                (
                    f"length={c['length']:3d}",
                    f"native_diff={c['max_abs_diff_native_vs_reference']:.6f}  "
                    f"guard_diff={c['max_abs_diff_guard_vs_reference']:.6f}  {flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
