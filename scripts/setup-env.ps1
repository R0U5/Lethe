<#
.SYNOPSIS
    Create an isolated .venv for lethe and install the project into it.

.DESCRIPTION
    lethe is a library plus a CLI, and both import torch and transformers.
    Install it with a bare `pip install -e .` and those land in whichever
    interpreter happens to be first on PATH -- normally a global one shared with
    everything else on the machine. Any other tool that then pins a different
    torch or transformers silently breaks lethe, and lethe silently breaks it.

    This script gives lethe its own environment, so its dependencies cannot be
    shadowed by a global install and cannot leak into one.

.PARAMETER Python
    Interpreter used to build the venv. Defaults to `python` on PATH.

.PARAMETER Transformers
    transformers spec to install. The default is the newest release that both
    ships the current architectures (Qwen3.5 and friends need >= 5.2) and still
    supports Python 3.10; 5.21+ requires Python >= 3.11.

.PARAMETER Extras
    lethe extras to install (dev, optimize, quantize, all). Default `dev`.

.PARAMETER Cpu
    Install CPU-only torch instead of the CUDA build (~200 MB vs ~2.6 GB).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup-env.ps1

.EXAMPLE
    # CPU-only, no CUDA download
    powershell -ExecutionPolicy Bypass -File scripts\setup-env.ps1 -Cpu

.EXAMPLE
    # a specifically located interpreter
    powershell -ExecutionPolicy Bypass -File scripts\setup-env.ps1 -Python "C:\Program Files\Python310\python.exe"
#>
[CmdletBinding()]
param(
    [string] $Python = "python",
    [string] $Transformers = ">=5.20,<5.21",
    [string] $Extras = "dev",
    [string] $TorchVersion = "2.9.1",
    [string] $TorchIndex = "https://download.pytorch.org/whl/cu126",
    [switch] $Cpu
)

$ErrorActionPreference = "Stop"

$root    = Split-Path -Parent $PSScriptRoot
$venv    = Join-Path $root ".venv"
$venvPy  = Join-Path $venv "Scripts\python.exe"

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Interpreter '$Python' not found on PATH. Pass -Python <path-to-python.exe>."
}

# lethe's floor. transformers 5.x needs torch >= 2.6 anyway.
$version = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$version -lt [version]"3.10") {
    throw "lethe requires Python >= 3.10; '$Python' is $version."
}

Write-Host "==> interpreter : $((& $Python -c 'import sys; print(sys.executable)'))  ($version)"

if (-not (Test-Path $venvPy)) {
    Write-Host "==> creating   : $venv"
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    Write-Host "==> reusing    : $venv"
}

Write-Host "==> upgrading  : pip / setuptools / wheel"
& $venvPy -m pip install --quiet --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }

if ($Cpu) {
    Write-Host "==> installing : torch (CPU-only)"
    & $venvPy -m pip install "torch==$TorchVersion"
} else {
    Write-Host "==> installing : torch $TorchVersion from $TorchIndex"
    # torch comes from its own index; everything else stays on PyPI, hence two
    # installs rather than one with a global --index-url.
    & $venvPy -m pip install "torch==$TorchVersion" --index-url $TorchIndex
}
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "torch install failed. If the CUDA channel above does not match your" -ForegroundColor Yellow
    Write-Host "driver, retry with -Cpu, or pass a channel that does (cu121/cu124/cu126/cu128)." -ForegroundColor Yellow
    throw "torch install failed"
}

Write-Host "==> installing : lethe[$Extras] + transformers$Transformers (editable)"
& $venvPy -m pip install -e ".[$Extras]" "transformers$Transformers"
if ($LASTEXITCODE -ne 0) { throw "project install failed" }

Write-Host ""
Write-Host "==> done. Environment is isolated from your global interpreter:" -ForegroundColor Green
& $venvPy -c @"
import sys, torch, transformers
print('    python       ', sys.version.split()[0])
print('    torch        ', torch.__version__, '(cuda available:', torch.cuda.is_available(), ')')
print('    transformers ', transformers.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C
print('    qwen3_5 known:', 'qwen3_5' in C)
"@

Write-Host ""
Write-Host "Activate it:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1      # PowerShell"
Write-Host "    .\.venv\Scripts\activate.bat      # cmd"
Write-Host "Or call it directly without activating:"
Write-Host "    .\.venv\Scripts\python.exe -m abliterate.cli --help"
Write-Host "    .\.venv\Scripts\abliterate.exe --help"
