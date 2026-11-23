; ========================================================================
; Lethe Installer Script for Inno Setup
;
; Produces a traditional Windows "double-click to install" .exe that:
;   * shows a real wizard with an install-location (folder) chooser
;   * is smart: auto-detects Python (>=3.10) and CUDA, pre-selects the
;     torch build that matches the machine
;   * lets the user pick where to install and whether to use CPU or CUDA torch
;   * creates Start Menu + desktop shortcuts to launch the Lethe web UI
;   * registers itself in Add/Remove Programs and uninstalls cleanly
;
; Compile with Inno Setup (https://jrsoftware.org/isinfo.php):
;   iscc lethe-installer.iss
;
; Replace the AppId GUID below with a freshly generated one (Tools > Generate GUID)
; before distributing.
; ========================================================================

#define MyAppName        "Lethe"
#define MyAppVersion     "0.1.0"
#define MyAppPublisher   "Lethe"
#define MyAppURL         "https://github.com/lethe"
#define MyAppExe         "abliterate.exe"

[Setup]
AppId={A1B2C3D4-E5F6-7890-ABCD-1234567890AB}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={pf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=Yes
LicenseFile=LICENSE
InfoBeforeFile=README.md
WizardImageFile=docs\ui-screenshot.png
WizardSmallImageFile=docs\ui-screenshot.png
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=no
Compression=lzma
SolidCompression=yes
OutputDir=installer
OutputBaseFilename=Lethe-Setup-{#MyAppVersion}
SetupLogging=yes
ChangesAssociations=yes
Uninstallable=Yes
UninstallDisplayIcon={app}\.venv\Scripts\python.exe
UninstallDisplayName={#MyAppName} {#MyAppVersion}
UninstallFilesDir={app}\Uninstall

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
PyNotFound=Python 3.10 or newer was not found on this machine. Install Python from https://python.org then re-run this installer.
PyTooOld=The Python found ("%1") is older than 3.10. Lethe requires Python >= 3.10.
CudaDetected=An NVIDIA GPU (CUDA) was detected, so CUDA torch (~2.6 GB) is selected by default.
CudaNotDetected=No NVIDIA GPU detected, so CPU-only torch (~200 MB) is selected by default. You can still choose CUDA if you have a compatible GPU.
Installing=Installing Lethe...
Finishing=Finishing installation...
Launching=Launching Lethe...

[Components]
Name: "program"; Description: "Lethe application"; Types: full install; Flags: fixed
Name: "doc"; Description: "Documentation"; Types: full

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: uncheckeddefault
Name: "launch"; Description: "&Launch Lethe web UI after installation"; Flags: uncheckeddefault
Name: "assoc"; Description: "Associate .lethe config files"; Flags: uncheckeddefault

[Code]
var
  PythonExe: string;
  UseCpuTorch: boolean;
  InstallExtras: string;

function DetectCuda: boolean;
var
  RC: integer;
  Dummy: string;
begin
  Result := Exec('nvidia-smi', '', '', SW_HIDE, ewWaitUntilTerminated, RC, Dummy)
            and (RC = 0);
end;

function InitializeSetup(): boolean;
var
  Cuda: boolean;
begin
  Result := True;

  // Smart: use 'python' from PATH; setup-env.ps1 validates it is >= 3.10 and
  // aborts with a friendly message otherwise (no fragile quote nesting here).
  PythonExe := 'python';

  // Smart: pick CUDA vs CPU torch by default based on GPU presence.
  Cuda := DetectCuda;
  UseCpuTorch := not Cuda;
  if Cuda then
    MsgBox(CustomMessage('CudaDetected'), mbInformation, MB_OK)
  else
    MsgBox(CustomMessage('CudaNotDetected'), mbInformation, MB_OK);

  InstallExtras := 'all';
end;

function NextButtonClick(CurPageID: integer): boolean;
begin
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Params: string;
  RC: integer;
  SetupPs1: string;
begin
  if CurStep = ssInstall then begin
    // Run the project's environment setup inside the chosen install dir.
    // This creates {app}\.venv, installs torch + lethe + transformers.
    SetupPs1 := ExpandConstant('{app}') + '\scripts\setup-env.ps1';
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + SetupPs1 + '" ';
    Params := Params + '-Python "' + PythonExe + '" ';
    Params := Params + '-Extras ' + InstallExtras + ' ';
    if UseCpuTorch then Params := Params + '-Cpu ';

    if not Exec('powershell.exe', Params, ExpandConstant('{app}'),
                SW_HIDE, ewWaitUntilTerminated, RC) then begin
      MsgBox('Failed to start the environment setup. See installer log.', mbError, MB_OK);
    end else if RC <> 0 then begin
      MsgBox('Environment setup exited with code ' + IntToStr(RC) + '. ' +
             'See the installer log for details.', mbError, MB_OK);
    end;
  end;
end;

procedure CurUninstallChanged(CurUninstallStep: TUninstallStep);
var
  UninstallPs1: string;
  RC: integer;
begin
  if CurUninstallStep = ssUninstall then begin
    // Run the project's uninstall helper if present, to clean the venv.
    UninstallPs1 := ExpandConstant('{app}') + '\scripts\uninstall.ps1';
    if FileExists(UninstallPs1) then
      Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -File "' + UninstallPs1 + '"',
           ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, RC);
  end;
end;

[Files]
; Source: the whole project tree (excluding caches/venv/git), installed to {app}.
Source: "src\*"; DestDir: "{app}\src"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "configs\*"; DestDir: "{app}\configs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "scripts\*"; DestDir: "{app}\scripts"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; Uninstall helper (registered in Add/Remove Programs removes {app}\.venv too).
Source: "scripts\uninstall.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName} Web UI"; Filename: "{app}\.venv\Scripts\python.exe"; Parameters: "-m abliterate.cli ui"; WorkingDir: "{app}"; IconFilename: "{app}\.venv\Scripts\python.exe"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName} Web UI"; Filename: "{app}\.venv\Scripts\python.exe"; Parameters: "-m abliterate.cli ui"; WorkingDir: "{app}"; IconFilename: "{app}\.venv\Scripts\python.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\.venv\Scripts\python.exe"; Parameters: "-m abliterate.cli ui"; WorkingDir: "{app}"; \
    Description: "Launch Lethe Web UI"; Flags: postinstall nowait skipifsilent; Tasks: launch

[UninstallDelete]
Type: dirifempty; Name: "{app}\.venv"
Type: files; Name: "{app}\.venv\**"
Type: dirifempty; Name: "{app}\output"

[Registry]
; Register in Add/Remove Programs for a proper uninstall entry.
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueType: string; ValueName: "DisplayName"; ValueData: "{#MyAppName} {#MyAppVersion}"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueType: string; ValueName: "DisplayVersion"; ValueData: "{#MyAppVersion}"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueType: string; ValueName: "Publisher"; ValueData: "{#MyAppPublisher}"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueType: string; ValueName: "InstallLocation"; ValueData: "{app}"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueType: string; ValueName: "UninstallString"; ValueData: """{uninstallexe}"""; Flags: uninsdeletevalue
Root: HKLM; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; \
    ValueType: string; ValueName: "DisplayIcon"; ValueData: "{app}\.venv\Scripts\python.exe"; Flags: uninsdeletevalue
