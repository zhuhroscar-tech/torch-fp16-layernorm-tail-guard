"""Minimal terminal styling: color only when it helps, never when it hurts.

Design goals -- spare and high-clarity rather than busy: color is always
semantic (green = healthy/yes, red = blocking/no, dim = neutral/unknown),
never decorative. Color is disabled automatically for non-interactive or
piped output so downstream tools/scripts never see stray ANSI escapes, and
this respects the NO_COLOR convention (https://no-color.org) plus an
explicit --no-color flag.

(Canonical copy, synced from reboot-safety-check's style.py -- see that
repo's ledger history for the shared design-system rationale.)
"""
from __future__ import annotations

import os
import sys


class Style:
    """A tiny ANSI wrapper. When disabled, every method is the identity
    function, so callers never need an `if style.enabled` branch."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def bold_green(self, text: str) -> str:
        return self._wrap("1;32", text)

    def bold_yellow(self, text: str) -> str:
        return self._wrap("1;33", text)

    def bold_red(self, text: str) -> str:
        return self._wrap("1;31", text)


def resolve_style(no_color_flag: bool = False, stream=None) -> Style:
    """Decide whether to emit ANSI color, in priority order:

    1. --no-color flag (explicit user request always wins).
    2. NO_COLOR env var set to anything (https://no-color.org convention).
    3. FORCE_COLOR env var set to anything (explicit opt-in -- e.g. a CI log
       a human will read later through a color-capable viewer). Checked
       before the TTY test so forcing color works even when not attached
       to a terminal.
    4. Otherwise: color only if stdout is a real terminal. Any pipe,
       redirect, or captured output gets plain text, so JSON output and
       scripted consumption are never polluted with escape codes.
    """
    stream = stream if stream is not None else sys.stdout
    if no_color_flag:
        return Style(False)
    if os.environ.get("NO_COLOR"):
        return Style(False)
    if os.environ.get("FORCE_COLOR"):
        return Style(True)
    try:
        is_tty = stream.isatty()
    except (AttributeError, ValueError):
        is_tty = False
    return Style(is_tty)


def bool_badge(style: Style, value) -> str:
    """Render True/False/None as a consistently styled yes/no/unknown."""
    if value is True:
        return style.green("yes")
    if value is False:
        return style.red("no")
    return style.dim("unknown")


def print_fields(rows, indent: str = "  ") -> None:
    """Print (label, value) pairs as an aligned two-column block."""
    if not rows:
        return
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{indent}{label.ljust(label_width)}   {value}")


# Status glyphs: a single filled circle rather than a checkmark/x mix, so
# every state reads as one consistent visual family at a glance -- only
# the color and label change. Falls back to plain ASCII when color/UTF-8
# fanciness isn't appropriate (same enabled flag as Style, since a
# non-interactive/piped stream shouldn't get Unicode decoration either).
_GLYPH_UNICODE = "\u25cf"  # ●
_GLYPH_ASCII = {"ok": "[OK]", "warn": "[!]", "fail": "[X]", "info": "[i]"}

_LEVEL_COLOR = {
    "ok": "bold_green",
    "warn": "bold_yellow",
    "fail": "bold_red",
    "info": "dim",
}


def status_headline(style: Style, level: str, text: str) -> str:
    """A single-line, color-coded headline: a filled dot + bold text.

    `level` is one of "ok" (green -- healthy/passing), "warn" (yellow --
    a limitation to be aware of but not necessarily fixable/wrong),
    "fail" (red -- an actionable problem), or "info" (dim -- neutral,
    no action implied). Kept to one glyph family (a dot, not a mix of
    checkmarks/crosses/warning-triangles) so the whole product line
    reads as one consistent visual system.
    """
    color_fn = getattr(style, _LEVEL_COLOR[level])
    if style.enabled:
        return f"{color_fn(_GLYPH_UNICODE)}  {style.bold(text)}"
    return f"{_GLYPH_ASCII[level]} {text}"


def section(title: str) -> None:
    """Print a dim, understated section label -- never a heavy ASCII-art
    banner. Whitespace and restraint carry the hierarchy, not decoration."""
    print()
    print(title)
