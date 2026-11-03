"""The CLI must run on the interpreter its .venv prepared, or fail loudly.

Regression guard for the failure modes behind ``abliterate.env``:

* A pip console script embeds the interpreter it was built with. An
  ``abliterate`` installed into a global site-packages therefore kept running
  on that interpreter no matter how healthy a ``.venv`` ``scripts/setup-env``
  had built next to the source -- and loading Qwen3.5 there produced a nicely
  worded "upgrade transformers" error aimed at an environment nobody wanted
  to use.
* A cancelled torch install leaves the ``.venv`` half-built (transformers in,
  torch half-extracted), which previously surfaced only as a bare
  ``ModuleNotFoundError`` deep in a job backtrace.

These tests exercise the decision logic against fabricated venv trees:
no subprocess, no torch import, no network.
"""

import os
import sys
from pathlib import Path

import pytest

from abliterate import env


def _venv_tree(tmp_path, torch_installed, transformers_installed, with_python=False):
    """A fake checkout with a ``.venv`` holding a chosen package census.

    ``torch_installed`` / ``transformers_installed`` represent what a finished
    pip install leaves behind: the ``*.dist-info`` records, and nothing else.
    """
    root = tmp_path / "lethe"
    if os.name == "nt":
        packages_dir = root / ".venv" / "Lib" / "site-packages"
    else:
        packages_dir = root / ".venv" / "lib" / "python3.999" / "site-packages"
    packages_dir.mkdir(parents=True)
    for name, present in (
        ("torch", torch_installed),
        ("transformers", transformers_installed),
    ):
        if present:
            (packages_dir / f"{name}-2.9.1.dist-info").mkdir()
    if with_python:
        python = env.venv_python(root)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"")  # existence is all the path checks need
    return root, root / ".venv"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def test_project_root_is_the_checkout():
    root = env.project_root()
    if root is None:
        pytest.skip("not running from an editable install")
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "abliterate" / "env.py").is_file()


def test_venv_python_layout_matches_platform():
    python = env.venv_python(Path("somewhere") / "lethe")
    assert python.parent.name == ("Scripts" if os.name == "nt" else "bin")
    assert python.name == ("python.exe" if os.name == "nt" else "python")
    assert python.parent.parent.name == ".venv"


def test_site_packages_rejects_a_packageless_venv(tmp_path):
    assert env.site_packages(tmp_path / "not-a-venv") is None


# --------------------------------------------------------------------------- #
# Completeness census
# --------------------------------------------------------------------------- #
def test_fresh_venv_reports_everything_missing(tmp_path):
    _, venv = _venv_tree(tmp_path, torch_installed=False, transformers_installed=False)
    assert env.missing_packages(venv) == ["torch", "transformers"]


def test_complete_venv_reports_nothing_missing(tmp_path):
    _, venv = _venv_tree(tmp_path, torch_installed=True, transformers_installed=True)
    assert env.missing_packages(venv) == []


def test_partially_built_venv_names_the_gap(tmp_path):
    _, venv = _venv_tree(tmp_path, torch_installed=True, transformers_installed=False)
    assert env.missing_packages(venv) == ["transformers"]


def test_incomplete_venv_error_points_at_the_fix(tmp_path):
    root, _ = _venv_tree(tmp_path, torch_installed=False, transformers_installed=True)
    error = env.incomplete_venv_error(root)
    assert error is not None
    assert "torch" in error
    assert "setup-env" in error


def test_incomplete_venv_error_is_none_when_complete(tmp_path):
    root, _ = _venv_tree(tmp_path, torch_installed=True, transformers_installed=True)
    assert env.incomplete_venv_error(root) is None


# --------------------------------------------------------------------------- #
# environment_note(): the extra sentence an environment-sensitive error carries
# --------------------------------------------------------------------------- #
def test_environment_note_silent_for_pip_installed_copy(monkeypatch):
    monkeypatch.setattr(env, "project_root", lambda: None)
    assert env.environment_note() is None


def test_environment_note_silent_inside_the_project_venv(tmp_path, monkeypatch):
    root, _ = _venv_tree(tmp_path, torch_installed=True, transformers_installed=True)
    monkeypatch.setattr(env, "project_root", lambda: root)
    monkeypatch.setattr(env, "running_in_venv", lambda _venv: True)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    assert env.environment_note() is None


def test_environment_note_silent_when_another_env_is_active(tmp_path, monkeypatch):
    root, _ = _venv_tree(tmp_path, torch_installed=True, transformers_installed=True)
    monkeypatch.setattr(env, "project_root", lambda: root)
    monkeypatch.setattr(env, "running_in_venv", lambda _venv: False)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))
    assert env.environment_note() is None


def test_environment_note_silent_when_no_venv_exists(tmp_path, monkeypatch):
    root, _ = _venv_tree(tmp_path, torch_installed=True, transformers_installed=True)
    (root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )).unlink(missing_ok=True)
    monkeypatch.setattr(env, "project_root", lambda: root)
    monkeypatch.setattr(env, "running_in_venv", lambda _venv: False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    assert env.environment_note() is None


def test_environment_note_calls_out_a_complete_venv(tmp_path, monkeypatch):
    root, _ = _venv_tree(tmp_path, torch_installed=True, transformers_installed=True,
                         with_python=True)
    monkeypatch.setattr(env, "project_root", lambda: root)
    monkeypatch.setattr(env, "running_in_venv", lambda _venv: False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    note = env.environment_note()
    assert note is not None
    assert "relaunch" in note


def test_environment_note_reports_an_incomplete_venv(tmp_path, monkeypatch):
    root, _ = _venv_tree(tmp_path, torch_installed=False, transformers_installed=True,
                         with_python=True)
    monkeypatch.setattr(env, "project_root", lambda: root)
    monkeypatch.setattr(env, "running_in_venv", lambda _venv: False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    note = env.environment_note()
    assert note is not None
    assert "torch" in note


# --------------------------------------------------------------------------- #
# bootstrap(): the actual hand-off decisions
# --------------------------------------------------------------------------- #
def test_bootstrap_fails_fast_inside_an_incomplete_venv(tmp_path, monkeypatch):
    root, _ = _venv_tree(tmp_path, torch_installed=False, transformers_installed=True,
                         with_python=True)
    monkeypatch.setattr(env, "project_root", lambda: root)
    monkeypatch.delenv(env._OPT_OUT_VAR, raising=False)
    monkeypatch.delenv(env._HANDOVER_GUARD, raising=False)
    monkeypatch.setattr(sys, "executable", str(env.venv_python(root)))
    with pytest.raises(SystemExit) as excinfo:
        env.bootstrap([])
    # Names what is missing before anything loads, with the repair path.
    assert "torch" in str(excinfo.value)
    assert "setup-env" in str(excinfo.value)


def test_bootstrap_respects_an_explicitly_activated_env(tmp_path, monkeypatch):
    root, _ = _venv_tree(tmp_path, torch_installed=True, transformers_installed=True,
                         with_python=True)
    monkeypatch.setattr(env, "project_root", lambda: root)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))
    handed_off = []
    monkeypatch.setattr(env, "_hand_off", lambda argv, py: handed_off.append(True))
    env.bootstrap(["ui", "--no-browser"])
    assert handed_off == []


def test_bootstrap_hands_off_from_the_wrong_interpreter(tmp_path, monkeypatch):
    root, _ = _venv_tree(tmp_path, torch_installed=True, transformers_installed=True,
                         with_python=True)
    monkeypatch.setattr(env, "project_root", lambda: root)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv(env._HANDOVER_GUARD, raising=False)
    handed_off = []
    monkeypatch.setattr(
        env, "_hand_off", lambda argv, py: handed_off.append((argv, py))
    )
    env.bootstrap(["run", "--model", "test-model"])
    assert handed_off == [(["run", "--model", "test-model"], env.venv_python(root))]


def test_bootstrap_opt_out_env_var(tmp_path, monkeypatch):
    root, _ = _venv_tree(tmp_path, torch_installed=True, transformers_installed=True,
                         with_python=True)
    monkeypatch.setattr(env, "project_root", lambda: root)
    monkeypatch.setenv(env._OPT_OUT_VAR, "1")
    handed_off = []
    monkeypatch.setattr(env, "_hand_off", lambda argv, py: handed_off.append(True))
    env.bootstrap(["ui"])
    assert handed_off == []
