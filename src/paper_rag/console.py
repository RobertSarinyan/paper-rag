"""Small command-line output helpers."""

from __future__ import annotations

import sys


def configure_console_output() -> None:
    """Use UTF-8 for redirected output and native Unicode for a Windows terminal."""
    if not sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
