#!/usr/bin/env pwsh
# =========================================================================
# Lethe Installer - self-contained Windows GUI installer (no compiler needed)
#
# Right-click → "Run with PowerShell", or open a PowerShell window and:
#   .\scripts\setup-installer.ps1
#
# What it does (smart):
#   * locates Python >= 3.10 on the machine (PATH, common install dirs)
#   * detects an NVIDIA GPU (nvidia-smi) and pre-selects CUDA torch (~2.6 GB);
#     falls back to CPU-only torch (~200 MB) when no GPU is found
#   * lets the user choose the install folder
#   * copies the project, creates an isolated .venv, installs torch + lethe +
#     transformers (reuses scripts/setup-env.ps1 so behavior stays identical)
#   * creates Start Menu + optional desktop shortcuts to launch the web UI
#   * registers itself in Add/Remove Programs and uninstalls cleanly
# =========================================================================
[CmdletBinding()]
param(
    [switch] $NoLaunch,
    [switch] $Quiet
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

# -------------------------------------------------------------------------
# Load WinForms (available on every Windows with PowerShell)
# -------------------------------------------------------------------------
Add-Type -AssemblyName System.Windows.Forms, System.Drawing, System.IO.FileSystem

[System.Windows.Forms.Application]::EnableVisualStyles()

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
function Find-Python {
    <# Locate a Python >= 3.10 interpreter. #>
    $candidates = New-Object System.Collections.Generic.List[string]

    # On PATH
    foreach ($name in @('python', 'python3', 'py')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $candidates.Add($cmd.Source) }
    }

    # Common installation locations (newest first)
    foreach ($ver in @('3.13','3.12','3.11','3.10')) {
        $candidates.Add("C:\Python$ver\python.exe")
        $candidates.Add("C:\Program Files\Python$ver\python.exe")
        $candidates.Add("C:\Program Files (x86)\Python$ver\python.exe")
        $candidates.Add("$env:LOCALAPPDATA\Programs\Python\Python$ver\python.exe")
    }

    foreach ($c in $candidates) {
        if (-not $c -or -not (Test-Path $c)) { continue }
        $ver = & $c -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
        if ($ver -and [version]::TryParse($ver, [ref]$null) -and ([version]$ver -ge [version]"3.10")) {
            return $c
        }
    }
    return $null
}

function Test-Cuda {
    <# True when an NVIDIA driver/GPU is present. #>
    $null = nvidia-smi 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Show-Balloon($Icon, $Title, $Text) {
    if ($Quiet) { return }
    $ni = New-Object System.Windows.Forms.NotifyIcon
    $ni.Icon = [System.Drawing.SystemIcons]::$Icon
    $ni.BalloonTipTitle = $Title
    $ni.BalloonTipText = $Text
    $ni.Visible = $true
    $ni.ShowBalloonTip(10000)
    Start-Sleep -Milliseconds 300
    $ni.Dispose()
}

function New-LetheShortcut($LnkPath, $IconIndex = 0) {
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($LnkPath)
    $sc.TargetPath = $script:VenvPython
    $sc.Arguments = '-m abliterate.cli ui'
    $sc.WorkingDirectory = $script:InstallDir
    $sc.Description = "Lethe web UI"
    $sc.IconLocation = "$($script:VenvPython),$IconIndex"
    $sc.Save()
}

function Register-Uninstall {
    <# Add/Remove Programs entry so the installed app uninstalls cleanly. #>
    $key = "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Lethe"
    if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }
    Set-ItemProperty -Path $key -Name "DisplayName"      -Value "Lethe $script:AppVersion" -Force
    Set-ItemProperty -Path $key -Name "DisplayVersion"   -Value $script:AppVersion -Force
    Set-ItemProperty -Path $key -Name "Publisher"        -Value "Lethe" -Force
    Set-ItemProperty -Path $key -Name "InstallLocation"  -Value $script:InstallDir -Force
    Set-ItemProperty -Path $key -Name "UninstallString"  -Value ("powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$($script:InstallDir)\scripts\uninstall.ps1`"") -Force
    Set-ItemProperty -Path $key -Name "DisplayIcon"      -Value "$($script:VenvPython),0" -Force
    Set-ItemProperty -Path $key -Name "NoModify"         -Value 1 -Force
    Set-ItemProperty -Path $key -Name "NoRepair"         -Value 1 -Force
}

function Remove-UninstallRegistration {
    $key = "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Lethe"
    if (Test-Path $key) { Remove-Item -Path $key -Force -ErrorAction SilentlyContinue }
}

# -------------------------------------------------------------------------
# State shared by the wizard pages
# -------------------------------------------------------------------------
$script:AppVersion   = "0.1.0"
$script:Root         = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:Py           = $null
$script:UseCpu       = $false
$script:InstallDir   = Join-Path $env:LOCALAPPDATA "Lethe"
$script:CreateDesktopIcon = $true
$script:LaunchAfter = $true
$script:InstallDirChosen = $false
$script:VenvPython   = Join-Path $script:InstallDir ".venv\Scripts\python.exe"

# -------------------------------------------------------------------------
# Build the wizard form
# -------------------------------------------------------------------------
$form = New-Object System.Windows.Forms.Form
$form.Text = "Lethe Setup"
$form.Size = New-Object System.Drawing.Size(640, 500)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.MinimizeBox = $false
$form.Icon = [System.Drawing.SystemIcons]::Information

$panel = New-Object System.Windows.Forms.Panel
$panel.Dock = "Fill"
$form.Controls.Add($panel)

# Header band (title + subtitle) reused on every page
$header = New-Object System.Windows.Forms.Panel
$header.Size = New-Object System.Drawing.Size(620, 72)
$header.Location = New-Object System.Drawing.Point(0, 0)
$header.BackColor = [System.Drawing.Color]::White

$lblTitle = New-Object System.Windows.Forms.Label
$lblTitle.Location = New-Object System.Drawing.Point(18, 14)
$lblTitle.Size = New-Object System.Drawing.Size(580, 22)
$lblTitle.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
$lblTitle.ForeColor = [System.Drawing.Color]::FromArgb(0x1A, 0x1C, 0x1F)

$lblSub = New-Object System.Windows.Forms.Label
$lblSub.Location = New-Object System.Drawing.Point(18, 36)
$lblSub.Size = New-Object System.Drawing.Size(590, 30)
$lblSub.Font = New-Object System.Drawing.Font("Segoe UI", 9)
$lblSub.ForeColor = [System.Drawing.Color]::FromArgb(0x5F, 0x66, 0x72)
$lblSub.AutoSize = $false

$header.Controls.AddRange(@($lblTitle, $lblSub))
$panel.Controls.Add($header)

# Navigation buttons (Back / Next / Cancel / Install / Finish)
$btnBack = New-Object System.Windows.Forms.Button
$btnBack.Text = "< &Back"
$btnBack.Size = New-Object System.Drawing.Size(88, 28)
$btnBack.Location = New-Object System.Drawing.Point(422, 440)
$btnBack.Enabled = $false
$btnBack.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Right

$btnNext = New-Object System.Windows.Forms.Button
$btnNext.Text = "&Next >"
$btnNext.Size = New-Object System.Drawing.Size(88, 28)
$btnNext.Location = New-Object System.Drawing.Point(522, 440)
$btnNext.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Right

$btnInstall = New-Object System.Windows.Forms.Button
$btnInstall.Text = "&Install"
$btnInstall.Size = New-Object System.Drawing.Size(100, 28)
$btnInstall.Location = New-Object System.Drawing.Point(522, 440)
$btnInstall.Visible = $false
$btnInstall.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Right

$btnCancel = New-Object System.Windows.Forms.Button
$btnCancel.Text = "&Cancel"
$btnCancel.Size = New-Object System.Drawing.Size(88, 28)
$btnCancel.Location = New-Object System.Drawing.Point(522, 440)
$btnCancel.DialogResult = "Cancel"
$btnCancel.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Right

$btnFinish = New-Object System.Windows.Forms.Button
$btnFinish.Text = "&Finish"
$btnFinish.Size = New-Object System.Drawing.Size(88, 28)
$btnFinish.Location = New-Object System.Drawing.Point(522, 440)
$btnFinish.Visible = $false
$btnFinish.DialogResult = "OK"
$btnFinish.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Right

$panel.Controls.AddRange(@($btnBack, $btnNext, $btnInstall, $btnCancel, $btnFinish))
$form.AcceptButton = $btnNext
$form.CancelButton = $btnCancel

# Separator line above the buttons
$sep = New-Object System.Windows.Forms.Panel
$sep.Size = New-Object System.Drawing.Size(620, 1)
$sep.Location = New-Object System.Drawing.Point(0, 424)
$sep.BackColor = [System.Drawing.Color]::FromArgb(0xE5, 0xE8, 0xEC)
$sep.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right
$panel.Controls.Add($sep)

# Content area (pages live here)
$content = New-Object System.Windows.Forms.Panel
$content.Location = New-Object System.Drawing.Point(0, 72)
$content.Size = New-Object System.Drawing.Size(620, 350)
$content.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right
$panel.Controls.Add($content)

# -------------------------------------------------------------------------
# Page: Welcome
# -------------------------------------------------------------------------
$pgWelcome = New-Object System.Windows.Forms.Panel
$pgWelcome.Size = $content.Size

$picWelcome = New-Object System.Windows.Forms.PictureBox
$picWelcome.Size = New-Object System.Drawing.Size(64, 64)
$picWelcome.Location = New-Object System.Drawing.Point(24, 24)
$picWelcome.Image = [System.Drawing.SystemIcons]::Information
$picWelcome.SizeMode = "CenterImage"

$lblWTitle = New-Object System.Windows.Forms.Label
$lblWTitle.Location = New-Object System.Drawing.Point(100, 20)
$lblWTitle.Size = New-Object System.Drawing.Size(480, 24)
$lblWTitle.Font = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Bold)
$lblWTitle.Text = "Welcome to the Lethe Setup Wizard"

$lblWSub = New-Object System.Windows.Forms.Label
$lblWSub.Location = New-Object System.Drawing.Point(100, 46)
$lblWSub.Size = New-Object System.Drawing.Size(480, 60)
$lblWSub.AutoSize = $false
$lblWSub.Text = "This wizard installs Lethe - a local tool that removes refusals from a " +
                  "language model - into its own isolated environment on this " +
                  "computer. It will download and install torch (~200 MB CPU or " +
                  "~2.6 GB CUDA), transformers, and Lethe itself."
$lblWSub.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$pgWelcome.Controls.AddRange(@($picWelcome, $lblWTitle, $lblWSub))

# -------------------------------------------------------------------------
# Page: Prerequisites (Python detection)
# -------------------------------------------------------------------------
$pgPre = New-Object System.Windows.Forms.Panel
$pgPre.Size = $content.Size

$picPre = New-Object System.Windows.Forms.PictureBox
$picPre.Size = New-Object System.Drawing.Size(48, 48)
$picPre.Location = New-Object System.Drawing.Point(24, 24)
$picPre.Image = [System.Drawing.SystemIcons]::Information
$picPre.SizeMode = "CenterImage"

$lblPTitle = New-Object System.Windows.Forms.Label
$lblPTitle.Location = New-Object System.Drawing.Point(86, 20)
$lblPTitle.Size = New-Object System.Drawing.Size(500, 24)
$lblPTitle.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
$lblPTitle.Text = "Checking prerequisites"

$lblPSub = New-Object System.Windows.Forms.Label
$lblPSub.Location = New-Object System.Drawing.Point(86, 46)
$lblPSub.Size = New-Object System.Drawing.Size(500, 30)
$lblPSub.Text = "Lethe requires Python 3.10 or newer."
$lblPSub.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$lblPy = New-Object System.Windows.Forms.Label
$lblPy.Location = New-Object System.Drawing.Point(86, 80)
$lblPy.Size = New-Object System.Drawing.Size(500, 30)
$lblPy.Text = "Locating Python…"
$lblPy.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$lblCuda = New-Object System.Windows.Forms.Label
$lblCuda.Location = New-Object System.Drawing.Point(86, 110)
$lblCuda.Size = New-Object System.Drawing.Size(500, 30)
$lblCuda.Text = "Checking for an NVIDIA GPU…"
$lblCuda.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$pgPre.Controls.AddRange(@($picPre, $lblPTitle, $lblPSub, $lblPy, $lblCuda))

# -------------------------------------------------------------------------
# Page: Install options (folder + CPU/CUDA)
# -------------------------------------------------------------------------
$pgOpt = New-Object System.Windows.Forms.Panel
$pgOpt.Size = $content.Size

$picOpt = New-Object System.Windows.Forms.PictureBox
$picOpt.Size = New-Object System.Drawing.Size(48, 48)
$picOpt.Location = New-Object System.Drawing.Point(24, 24)
$picOpt.Image = [System.Drawing.SystemIcons]::Information
$picOpt.SizeMode = "CenterImage"

$lblOTitle = New-Object System.Windows.Forms.Label
$lblOTitle.Location = New-Object System.Drawing.Point(86, 20)
$lblOTitle.Size = New-Object System.Drawing.Size(500, 24)
$lblOTitle.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
$lblOTitle.Text = "Choose install location"

$lblOSub = New-Object System.Windows.Forms.Label
$lblOSub.Location = New-Object System.Drawing.Point(86, 46)
$lblOSub.Size = New-Object System.Drawing.Size(500, 30)
$lblOSub.Text = "Where should Lethe be installed?"
$lblOSub.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$txtDir = New-Object System.Windows.Forms.TextBox
$txtDir.Location = New-Object System.Drawing.Point(86, 80)
$txtDir.Size = New-Object System.Drawing.Size(370, 24)
$txtDir.Text = $script:InstallDir
$txtDir.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right

$btnBrowse = New-Object System.Windows.Forms.Button
$btnBrowse.Text = "Browse…"
$btnBrowse.Size = New-Object System.Drawing.Size(80, 28)
$btnBrowse.Location = New-Object System.Drawing.Point(466, 76)
$btnBrowse.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Right

$grpTorch = New-Object System.Windows.Forms.GroupBox
$grpTorch.Text = "Torch build"
$grpTorch.Location = New-Object System.Drawing.Point(86, 118)
$grpTorch.Size = New-Object System.Drawing.Size(480, 60)
$grpTorch.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right

$radCuda = New-Object System.Windows.Forms.RadioButton
$radCuda.Text = "CUDA (NVIDIA GPU) - ~2.6 GB download"
$radCuda.Location = New-Object System.Drawing.Point(14, 20)
$radCuda.Size = New-Object System.Drawing.Size(300, 20)

$radCpu = New-Object System.Windows.Forms.RadioButton
$radCpu.Text = "CPU only - ~200 MB download"
$radCpu.Location = New-Object System.Drawing.Point(14, 40)
$radCpu.Size = New-Object System.Drawing.Size(240, 20)

$grpTorch.Controls.AddRange(@($radCuda, $radCpu))

$chkDesktop = New-Object System.Windows.Forms.CheckBox
$chkDesktop.Text = "Create a desktop icon"
$chkDesktop.Location = New-Object System.Drawing.Point(86, 190)
$chkDesktop.Size = New-Object System.Drawing.Size(220, 22)
$chkDesktop.Checked = $true

$pgOpt.Controls.AddRange(@($picOpt, $lblOTitle, $lblOSub, $txtDir, $btnBrowse, $grpTorch, $chkDesktop))

# -------------------------------------------------------------------------
# Page: Installing (progress + log)
# -------------------------------------------------------------------------
$pgInst = New-Object System.Windows.Forms.Panel
$pgInst.Size = $content.Size

$picInst = New-Object System.Windows.Forms.PictureBox
$picInst.Size = New-Object System.Drawing.Size(48, 48)
$picInst.Location = New-Object System.Drawing.Point(24, 24)
$picInst.Image = [System.Drawing.SystemIcons]::Information
$picInst.SizeMode = "CenterImage"

$lblITitle = New-Object System.Windows.Forms.Label
$lblITitle.Location = New-Object System.Drawing.Point(86, 20)
$lblITitle.Size = New-Object System.Drawing.Size(500, 24)
$lblITitle.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
$lblITitle.Text = "Installing Lethe"

$lblIStatus = New-Object System.Windows.Forms.Label
$lblIStatus.Location = New-Object System.Drawing.Point(86, 48)
$lblIStatus.Size = New-Object System.Drawing.Size(500, 24)
$lblIStatus.Text = "Preparing…"
$lblIStatus.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$prog = New-Object System.Windows.Forms.ProgressBar
$prog.Location = New-Object System.Drawing.Point(86, 78)
$prog.Size = New-Object System.Drawing.Size(460, 22)
$prog.Style = "Continuous"
$prog.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right

$lblLog = New-Object System.Windows.Forms.Label
$lblLog.Location = New-Object System.Drawing.Point(86, 110)
$lblLog.Size = New-Object System.Drawing.Size(460, 200)
$lblLog.AutoSize = $false
$lblLog.BorderStyle = "FixedSingle"
$lblLog.BackColor = [System.Drawing.Color]::FromArgb(0xF6, 0xF7, 0xF9)
$lblLog.Font = New-Object System.Drawing.Font("Consolas", 8.5)
$lblLog.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right

$pgInst.Controls.AddRange(@($picInst, $lblITitle, $lblIStatus, $prog, $lblLog))

# -------------------------------------------------------------------------
# Page: Complete
# -------------------------------------------------------------------------
$pgDone = New-Object System.Windows.Forms.Panel
$pgDone.Size = $content.Size

$picDone = New-Object System.Windows.Forms.PictureBox
$picDone.Size = New-Object System.Drawing.Size(48, 48)
$picDone.Location = New-Object System.Drawing.Point(24, 24)
$picDone.Image = [System.Drawing.SystemIcons]::Information
$picDone.SizeMode = "CenterImage"

$lblDTitle = New-Object System.Windows.Forms.Label
$lblDTitle.Location = New-Object System.Drawing.Point(86, 20)
$lblDTitle.Size = New-Object System.Drawing.Size(500, 24)
$lblDTitle.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
$lblDTitle.Text = "Installation complete"

$lblDSub = New-Object System.Windows.Forms.Label
$lblDSub.Location = New-Object System.Drawing.Point(86, 46)
$lblDSub.Size = New-Object System.Drawing.Size(500, 60)
$lblDSub.AutoSize = $false
$lblDSub.Text = "Lethe is installed and ready to use. You can launch it from the Start " +
                  "Menu (or the desktop icon, if you created one)."
$lblDSub.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$chkLaunch = New-Object System.Windows.Forms.CheckBox
$chkLaunch.Text = "&Launch Lethe web UI now"
$chkLaunch.Location = New-Object System.Drawing.Point(86, 116)
$chkLaunch.Size = New-Object System.Drawing.Size(240, 22)
$chkLaunch.Checked = $true

$pgDone.Controls.AddRange(@($picDone, $lblDTitle, $lblDSub, $chkLaunch))

# -------------------------------------------------------------------------
# Page navigation
# -------------------------------------------------------------------------
$pages = @($pgWelcome, $pgPre, $pgOpt, $pgInst, $pgDone)
$pageIndex = 0

function Show-Page($idx) {
    foreach ($p in $pages) { $p.Visible = $false }
    $pages[$idx].Visible = $true
    $content.Controls.Clear()
    $content.Controls.Add($pages[$idx])

    $btnBack.Enabled = ($idx -gt 0)
    $btnNext.Visible = ($idx -lt 3)
    $btnInstall.Visible = ($idx -eq 3)
    $btnFinish.Visible = ($idx -eq 4)
    $btnCancel.Visible = ($idx -ne 4)

    if ($idx -eq 4) { $form.AcceptButton = $btnFinish }
    else { $form.AcceptButton = $btnNext }

    $form.Text = "Lethe Setup - $($pages[$idx].Tag)"
}

# -------------------------------------------------------------------------
# Prereq check runs as soon as the prerequisites page is shown
# -------------------------------------------------------------------------
$pgPre.add_Enter({
    $lblPy.Text = "Locating Python…"
    $lblCuda.Text = "Checking for an NVIDIA GPU…"
    $form.Refresh()

    $py = Find-Python
    $script:Py = $py
    if (-not $py) {
        $lblPy.ForeColor = [System.Drawing.Color]::Red
        $lblPy.Text = "Python 3.10+ was not found. Install Python from https://python.org, then re-run this installer."
        $btnNext.Enabled = $false
        Show-Balloon Error "Python not found" "Install Python 3.10 or newer, then re-run the Lethe installer."
    } else {
        $ver = & $py -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
        $lblPy.ForeColor = [System.Drawing.Color]::FromArgb(0x1A,0x9D,0x63)
        $lblPy.Text = "Python $ver found: $py"
        $btnNext.Enabled = $true
    }

    $cuda = Test-Cuda
    $script:UseCpu = -not $cuda
    $radCuda.Checked = -not $cuda
    $radCpu.Checked  = $cuda
    if ($cuda) {
        $lblCuda.ForeColor = [System.Drawing.Color]::FromArgb(0x1A,0x9D,0x63)
        $lblCuda.Text = "NVIDIA GPU detected - CUDA torch will be used."
    } else {
        $lblCuda.ForeColor = [System.Drawing.Color]::FromArgb(0xD1,0x48,0x3B)
        $lblCuda.Text = "No NVIDIA GPU detected - CPU-only torch will be used (~200 MB)."
    }
})

# Browse folder
$btnBrowse.add_Click({
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = "Select the Lethe install folder"
    $dlg.SelectedPath = $txtDir.Text
    $dlg.ShowNewFolderButton = $true
    if ($dlg.ShowDialog() -eq "OK") {
        $txtDir.Text = $dlg.SelectedPath
        $script:InstallDirChosen = $true
    }
})

# Keep defaults in sync
$txtDir.add_TextChanged({ $script:InstallDir = $txtDir.Text })
$radCuda.add_CheckStateChanged({ if ($radCuda.Checked) { $script:UseCpu = $false } })
$radCpu.add_CheckStateChanged({  if ($radCpu.Checked)  { $script:UseCpu = $true } })
$chkDesktop.add_CheckStateChanged({ $script:CreateDesktopIcon = $chkDesktop.Checked })
$chkLaunch.add_CheckStateChanged({ $script:LaunchAfter = $chkLaunch.Checked })

# -------------------------------------------------------------------------
# Install page worker (BackgroundWorker keeps the UI responsive)
# -------------------------------------------------------------------------
$worker = New-Object System.ComponentModel.BackgroundWorker
$worker.WorkerReportsProgress = $true

$worker.add_DoWork({
    param($srcRoot, $installDir, $py, $useCpu, $sender, $eventArgs)

    function Log($msg) {
        $sender.ReportProgress(-1, $msg)
    }

    function Step($pct, $msg) {
        $sender.ReportProgress($pct, $msg)
    }

    Log("Install root: $installDir")
    Step(5, "Copying project files…")

    # Copy the project tree into the install dir (exclude caches/git/venv).
    $excludeDirs = @('.git', '.venv', '__pycache__', '.pytest_cache', 'installer')
    $excludeFiles = @('*.pyc')

    $dirsToCopy = @('src', 'configs', 'data', 'docs', 'scripts')
    $filesToCopy = @('pyproject.toml', 'README.md', 'LICENSE', 'requirements.txt')

    $total = ($dirsToCopy.Count + $filesToCopy.Count)
    $done = 0

    foreach ($d in $dirsToCopy) {
        $src = Join-Path $srcRoot $d
        $dst = Join-Path $installDir $d
        if (-not (Test-Path $src)) { continue }
        # Robocopy is fast and supports directory exclusions cleanly.
        $args = "/E /NFL /NDL /NJH /NJS /nc /ns /np"
        foreach ($ex in $excludeDirs) { $args += " /XD `"$ex`"" }
        foreach ($ex in $excludeFiles) { $args += " /XF `"$ex`"" }
        $robocopy = & robocopy "$src" "$dst" $args 2>&1
        # Robocopy exit codes 0-7 indicate success; >= 8 is an error.
        $rc = $LASTEXITCODE
        if ($rc -ge 8) {
            throw "robocopy failed copying $d (exit code $rc)"
        }
        $done++
        Step([int]($done / $total * 35), "Copying $d…")
    }

    foreach ($f in $filesToCopy) {
        $src = Join-Path $srcRoot $f
        $dst = Join-Path $installDir $f
        if (Test-Path $src) {
            Copy-Item -Path $src -Destination $dst -Force
        }
        $done++
        Step([int]($done / $total * 35), "Copying $f…")
    }

    # Clear any leftover caches inside the copied tree.
    Get-ChildItem -Path $installDir -Filter __pycache__ -Recurse -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    Step(40, "Creating isolated Python environment & installing dependencies…")

    # Run the project's setup-env.ps1 inside the install dir. It creates
    # $installDir\.venv and installs torch + lethe + transformers there.
    $setupPs1 = Join-Path $installDir "scripts\setup-env.ps1"
    $params = "-NoProfile -ExecutionPolicy Bypass -File `"$setupPs1`" -Python `"$py`" -Extras all"
    if ($useCpu) { $params += " -Cpu" }

    $installProc = Start-Process -FilePath powershell.exe -ArgumentList $params -WorkingDirectory $installDir -Wait -PassThru -NoNewWindow
    if ($installProc.ExitCode -ne 0) {
        throw "Environment setup exited with code $($installProc.ExitCode). See the installer console output."
    }

    Step(80, "Creating shortcuts…")

    $script:VenvPython = Join-Path $installDir ".venv\Scripts\python.exe"
    if (-not (Test-Path $script:VenvPython)) {
        throw "Expected venv interpreter not found: $($script:VenvPython)"
    }

    # Start Menu shortcut
    $startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Lethe"
    if (-not (Test-Path $startMenu)) { New-Item -Path $startMenu -ItemType Directory -Force | Out-Null }
    New-LetheShortcut -LnkPath (Join-Path $startMenu "Lethe Web UI.lnk") -IconIndex 0

    # Desktop shortcut (optional)
    if ($script:CreateDesktopIcon) {
        $desktop = [Environment]::GetFolderPath("Desktop")
        New-LetheShortcut -LnkPath (Join-Path $desktop "Lethe Web UI.lnk") -IconIndex 0
    }

    # Register in Add/Remove Programs
    Register-Uninstall

    Step(95, "Finalizing…")
    Step(100, "Done.")
})

$worker.add_ProgressChanged({
    param($sender, $e)
    if ($e.ProgressPercentage -ge 0) {
        $prog.Value = $e.ProgressPercentage
    }
    if ($e.UserState -is [string]) {
        $lblIStatus.Text = $e.UserState
        # Append to the log box
        $cur = $lblLog.Text
        $new = ($cur + "`n" + $e.UserState).Trim("`n")
        # Keep the log bounded
        $lines = ($new -split "`n")
        if ($lines.Count -gt 120) { $new = ($lines[-120..-1] -join "`n") }
        $lblLog.Text = $new
        $form.Refresh()
    }
})

$worker.add_RunWorkerCompleted({
    param($sender, $e)
    $btnInstall.Enabled = $true
    $btnCancel.Enabled = $true
    if ($e.Error) {
        $lblIStatus.ForeColor = [System.Drawing.Color]::Red
        $lblIStatus.Text = "Installation failed: $($e.Error.Message)"
        Show-Balloon Error "Install failed" $e.Error.Message
        return
    }
    if ($e.Cancelled) {
        $lblIStatus.Text = "Installation cancelled."
        return
    }
    $lblIStatus.ForeColor = [System.Drawing.Color]::FromArgb(0x1A,0x9D,0x63)
    $lblIStatus.Text = "Installation complete."
    # Move to the finish page automatically
    $pageIndex = 4
    Show-Page 4
})

$btnInstall.add_Click({
    $script:InstallDir = $txtDir.Text
    if ($script:UseCpu) { } # already set by radio buttons
    $btnInstall.Enabled = $false
    $btnCancel.Enabled = $false
    $prog.Value = 0
    $lblLog.Text = ""
    $worker.RunWorkerAsync(@($script:Root, $script:InstallDir, $script:Py, $script:UseCpu, $worker, $null))
})

# -------------------------------------------------------------------------
# Navigation handlers
# -------------------------------------------------------------------------
$btnBack.add_Click({
    if ($pageIndex -gt 0) { $pageIndex--; Show-Page $pageIndex }
})

$btnNext.add_Click({
    switch ($pageIndex) {
        0 { $pageIndex = 1 }                        # welcome -> prereq
        1 {
            if (-not $script:Py) { $lblPy.Focus(); return }
            $pageIndex = 2                          # prereq -> options
        }
        2 {
            $script:InstallDir = $txtDir.Text
            if ([string]::IsNullOrWhiteSpace($script:InstallDir)) {
                $txtDir.Focus(); return
            }
            $pageIndex = 3                          # options -> installing
        }
        3 { }                                       # installing (worker moves on)
    }
    Show-Page $pageIndex
})

$btnFinish.add_Click({
    if ($script:LaunchAfter -and -not $NoLaunch) {
        # Launch the UI on the installed venv interpreter, detached.
        Start-Process -FilePath $script:VenvPython -ArgumentList "-m abliterate.cli ui" -WorkingDirectory $script:InstallDir
    }
    $form.Close()
})

# Wire up form close to cleanup if cancelled mid-install
$form.add_FormClosing({
    param($sender, $e)
    if ($worker.IsBusy) {
        $e.Cancel = $true
        Show-Balloon Warning "Install in progress" "Wait for installation to finish before closing."
    }
})

# -------------------------------------------------------------------------
# Go
# -------------------------------------------------------------------------
Show-Page 0
$null = $form.ShowDialog()
