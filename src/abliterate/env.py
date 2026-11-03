"""Put the CLI on the interpreter that belongs to it.

lethe is a library plus a CLI, and both import torch and transformers. Those
two conflict across projects -- different tools pin different versions -- so
lethe is meant to live in its own ``.venv`` (``scripts/setup-env``). But a pip
console script embeds the interpreter it was built with: an ``abliterate.exe``
installed into a global site-packages runs lethe forever on whatever versions
that interpreter happens to hold, no matter how healthy a ``.venv`` sits next to
the source. The result is confusing by construction -- e.g. a UI that reports
``transformers=4.57.6`` under a project venv holding a 5.x release that knows
the checkpoint's architecture.

``bootstrap()`` closes that gap. It runs first in ``abliterate.cli.main()`` and:

1. when already running on the project interpreter, checks the venv is
   complete. A cancelled torch install commonly leaves transformers in place
   and torch half-extracted (its wheel is ~2.6 GB), which otherwise only
   surfaces as an unhelpful ``ModuleNotFoundError`` deep under the UI;
2. when an explicitly activated environment is in play (``VIRTUAL_ENV`` set),
   respects it and steps aside -- the user drives that one;
3. otherwise, when the source checkout has a ``.venv``, relaunches
   ``python -m abliterate.cli`` on the venv interpreter with the arguments and
   the exit code forwarded intact. Windows has no ``exec(2)``; ``os.execv``
   there is a spawn plus an emulated exit, so the hand-off is a synchronous
   spawn instead, and an ``ABLitERATE_BOOTSTRAPPED`` guard makes a loop
   impossible;
4. when there is no ``.venv``, prints a single hint and keeps running -- a
   global interpreter may be exactly what the user intends.

Opt out entirely with ``ABLitERATE_NO_BOOTSTRAP=1``. The module imports only
``os`` / ``subprocess`` / ``sys`` / ``pathlib`` -- never torch -- so it can run
before anything else, on an interpreter whose site-packages are wrong.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_OPT_OUT_VAR = "ABLitERATE_NO_BOOTSTRAP"
_HANDOVER_GUARD = "ABLitERATE_BOOTSTRAPPED"

# Everything lethe does routes through these. torch is additionally the
# package a cancelled ``pip`` run most often leaves half-installed, being a
# one-time ~2.6 GB download during ``setup-env``.
_REQUIRED = ("torch", "transformers")


def project_root() -> Optional[Path]:
    """The lethe checkout this copy was installed editable from, or None.

    An editable install keeps its source inside the checkout (the ``src``
    layout), so the repo root is two levels up. A regular wheel install has no
    relationship to any checkout, and no ``.venv`` to hand over to, hence a
    None instead of a guess.
    """
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        return root
    return None


def venv_python(root: Path) -> Path:
    """The checkout's ``.venv`` interpreter, regardless of whether it exists."""
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def site_packages(venv: Path) -> Optional[Path]:
    """Where third-party packages land inside ``venv``, or None if it has none.

    Windows venvs put packages in ``Lib\\site-packages``; POSIX ones under
    ``lib/pythonX.Y/site-packages`` with a version component that has to be
    globbed rather than guessed.
    """
    if os.name == "nt":
        win = venv / "Lib" / "site-packages"
        return win if win.is_dir() else None
    for candidate in sorted((venv / "lib").glob("python*/site-packages")):
        if candidate.is_dir():
            return candidate
    return None


def missing_packages(venv: Path) -> list[str]:
    """Packages a venv directory lacks, from an earlier interrupted install.

    The ``*.dist-info`` record is written only once every file of a wheel is
    in place, so its presence means "the install finished" while its absence
    means "the package may be half-extracted". A ``version.py`` marker would
    report packages that are still unusable -- a cancelled install leaves
    those behind too.
    """
    packages = site_packages(venv)
    if packages is None:
        return list(_REQUIRED)
    return [name for name in _REQUIRED if not any(packages.glob(f"{name}-*.dist-info"))]


def running_in_venv(venv: Path) -> bool:
    """Whether this process's interpreter *is* ``venv``'s (not just a sibling)."""
    if os.name == "nt":
        python = venv / "Scripts" / "python.exe"
    else:
        python = venv / "bin" / "python"
    if not python.is_file():
        return False
    here = os.path.normcase(str(Path(sys.executable).resolve()))
    there = os.path.normcase(str(python.resolve()))
    return here == there


def is_global_interpreter() -> bool:
    """Running without a venv at all (the situation pip console scripts create)."""
    return sys.prefix == sys.base_prefix


def incomplete_venv_error(root: Path) -> Optional[str]:
    """What is wrong with the checkout's ``.venv``, or None when it is whole."""
    missing = missing_packages(root / ".venv")
    if not missing:
        return None
    name = "setup-env.ps1" if os.name == "nt" else "setup-env.sh"
    return (
        "the project .venv is incomplete: missing "
        + (", ".join(missing))
        + f".\nA cancelled or crashed install commonly lands here -- torch's "
          f"wheel alone is ~2.6 GB.\nRe-running the setup finishes it "
          f"(idempotent; the wheels are already in pip's cache):\n"
        f"    {(root / 'scripts' / name)}\n"
    )


def _hand_off(argv: list[str], venv_python_path: Path) -> None:
    """Re-run this CLI on the venv interpreter, then exit with its status.

    Arguments are forwarded verbatim, stdout/stderr are inherited, and the
    parent exits with the child's status, so the caller sees one continuous
    program. The guard environment variable keeps the child from handing off
    again should the venv somehow not own what the check expects.
    """
    sys.stderr.write(
        f"abliterate: relaunching on the project .venv interpreter "
        f"({venv_python_path}).\n"
        f"abliterate: this shell's Python is {sys.executable}; set "
        f"{_OPT_OUT_VAR}=1 to always run there instead.\n"
    )
    sys.stderr.flush()
    run_env = dict(os.environ)
    run_env[_HANDOVER_GUARD] = "1"
    code = subprocess.call(
        [str(venv_python_path), "-m", "abliterate.cli", *argv], env=run_env
    )
    raise SystemExit(code)


def bootstrap(argv: Optional[list[str]] = None) -> None:
    """Make sure the process runs on the checkout's ``.venv`` interpreter.

    ``argv`` defaults to ``sys.argv[1:]`` -- the same default ``argparse``
    uses -- and an explicitly passed list is forwarded to the relaunched CLI
    unchanged. Never raises on a healthy path; the process either continues
    here or is replaced by a child running the identical CLI entry point.
    """
    extra = list(sys.argv[1:] if argv is None else argv)

    if os.environ.get(_OPT_OUT_VAR):
        return

    root = project_root()
    if root is None:
        return  # pip-installed copy: no checkout, nothing to hand off to

    venv = root / ".venv"

    # On the project interpreter already: only make sure it is complete.
    if running_in_venv(venv):
        problem = incomplete_venv_error(root)
        if problem:
            raise SystemExit(f"abliterate: {problem}")
        return

    # A venv the user activated on purpose: not ours to override.
    if os.environ.get("VIRTUAL_ENV"):
        return

    venv_python_path = venv_python(root)
    if venv_python_path.is_file():
        missing = missing_packages(venv)
        if missing:
            sys.stderr.write(
                f"abliterate: warning: {venv} lacks {', '.join(missing)}, so "
                f"staying on this interpreter ({sys.executable}).\n"
                f"abliterate: finish the venv with scripts/setup-env; the "
                f"loader will flag impossible architectures either way.\n"
            )
            return
        if os.environ.get(_HANDOVER_GUARD):
            sys.stderr.write(
                f"abliterate: refusing to hand off to {venv_python_path} "
                f"again (loop guard).\n"
            )
            return
        _hand_off(extra, venv_python_path)

    elif is_global_interpreter():
        sys.stderr.write(
            f"abliterate: no project .venv at {venv}; running on the global "
            f"interpreter {sys.executable}.\n"
            f"abliterate: scripts/setup-env creates an isolated one (one "
            f"time; downloads torch).\n"
        )


def environment_note() -> Optional[str]:
    """One sentence of context for errors an interpreter can be responsible for.

    Raised from loaders and other version-sensitive spots via e.g.
    ``UnsupportedArchitectureError``: transformers' dispatch is per-environment,
    so an error read on interpreter A can misrepresent what a fix in
    interpreter B must reach. Returns None when nothing needs saying -- pip
    installs unrelated to any checkout, or the checkout's own ``.venv``.
    """
    root = project_root()
    if root is None:
        return None
    venv = root / ".venv"
    if running_in_venv(venv) or os.environ.get("VIRTUAL_ENV"):
        return None
    python = venv_python(root)
    if not python.is_file():
        return None
    missing = missing_packages(venv)
    if missing:
        return (
            f"note: lethe's project .venv ({python}) is missing "
            f"{', '.join(missing)}; the error above came from interpreter "
            f"{sys.executable} instead."
        )
    return (
        f"note: the CLI normally relaunches itself on lethe's .venv "
        f"({python}), so an error raised this far from {sys.executable} "
        f"means a hand-off was skipped (ABLitERATE_NO_BOOTSTRAP?) or the "
        f"upgrade must reach this interpreter."
    )
