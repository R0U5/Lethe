"""Default prompt files must work outside a source checkout."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def test_default_prompts_in_installed_package_layout(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src" / "abliterate"
    site = tmp_path / "site-packages"
    shutil.copytree(
        source, site / "abliterate", ignore=shutil.ignore_patterns("__pycache__")
    )

    script = """
import sys
import importlib.util
sys.path.insert(0, sys.argv[1])
# Import just the data submodule so this packaging test does not need to load
# the package's heavyweight model dependencies in its isolated subprocess.
spec = importlib.util.spec_from_file_location(
    "abliterate", sys.argv[1] + "/abliterate/__init__.py"
)
package = importlib.util.module_from_spec(spec)
sys.modules["abliterate"] = package
from abliterate.data import load_prompts
for filename in ("harmful.txt", "harmless.txt"):
    prompts = load_prompts(None, default_file=filename)
    assert prompts, filename
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(site)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_default_prompts_in_zip_import(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src" / "abliterate"
    archive = tmp_path / "abliterate.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("abliterate/__init__.py", "")
        package.write(source / "data.py", "abliterate/data.py")
        for filename in ("harmful.txt", "harmless.txt"):
            package.write(
                source / "sample_data" / filename,
                f"abliterate/sample_data/{filename}",
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
        [sys.executable, "-I", "-c", script, str(archive)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
