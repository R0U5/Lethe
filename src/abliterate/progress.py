"""Tiny progress-reporting helper shared by the pipeline and the UI.

A progress callback has the signature ``(phase: str, fraction: float | None)``
where ``fraction`` is the completion of the *current stage* in ``[0, 1]``, or
``None`` when the stage has no measurable progress (the UI shows a spinner
instead of a frozen bar).

Reporting is best-effort: a misbehaving callback must never break a run, so
``emit`` swallows exceptions.
"""

from __future__ import annotations

from typing import Callable, Optional

ProgressCallback = Callable[[str, Optional[float]], None]


def emit(
    on_progress: Optional[ProgressCallback],
    phase: str,
    fraction: Optional[float] = None,
) -> None:
    """Invoke ``on_progress(phase, fraction)`` if one was supplied."""
    if on_progress is None:
        return
    try:
        on_progress(phase, fraction)
    except Exception:  # pragma: no cover - reporting must never break the run
        pass
