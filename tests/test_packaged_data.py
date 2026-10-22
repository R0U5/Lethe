"""Default prompt files must work outside a source checkout."""

import shutil
import subprocess
import sys
from pathlib import Path


def test_default_prompts_in_installed_package_layout(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src" / "abliterate"
    site = tmp_path / "site-packages"
    shutil.copytree(
        source, site / "abliterate", ignore=shutil.ignore_patterns("__pycache__")
    )

    script = """
import sys
sys.path.insert(0, sys.argv[1])
from abliterate.data import load_prompts
for filename in ("harmful.txt", "harmless.txt"):
    prompts = load_prompts(None, default_file=filename)
    assert prompts, filename
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(site)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
