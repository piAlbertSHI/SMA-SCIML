"""Audited baseline runners for the SciML work packages."""

from typing import Any


def run_benchmarks(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily import the WP2 runner, keeping ``python -m`` warning-free."""
    from .scalar_benchmarks import run_benchmarks as _run_benchmarks

    return _run_benchmarks(*args, **kwargs)


__all__ = ["run_benchmarks"]
