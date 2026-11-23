<#
.SYNOPSIS
    Clean up the Lethe environment created by the installer.

.DESCRIPTION
    Called by the Inno Setup uninstaller (and usable standalone) to remove
    the project .venv next to this script so an uninstall leaves nothing behind.
#>
[CmdletBinding()]
param(
    [switch] $NoRemoveVenv
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $scriptDir ".venv"

Write-Host "==> Lethe uninstall cleanup"

if (-not $NoRemoveVenv -and (Test-Path $venv)) {
    Write-Host "==> removing   : $venv"
    Remove-Item -Recurse -Force $venv -ErrorAction SilentlyContinue
    if (Test-Path $venv) {
        Write-Warning "Could not fully remove $venv (files may be in use). Remove it manually."
    } else {
        Write-Host "==> removed."
    }
} else {
    Write-Host "==> skipping   : .venv (NoRemoveVenv or not present)"
}

Write-Host "==> done."