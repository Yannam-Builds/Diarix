#define AppName "Diarix"
#define AppPublisher "Diarix"

#ifndef AppVersion
  #define AppVersion "0.1.0-alpha.1"
#endif
#ifndef EditionName
  #define EditionName "Full CUDA"
#endif
#ifndef PayloadDir
  #define PayloadDir "..\artifacts\payloads\full-cuda-0.1.0-alpha.1"
#endif
#ifndef OutputDir
  #define OutputDir "..\artifacts\installers"
#endif

[Setup]
AppId={{B1A3B6F9-65E5-46D0-9E91-92D7B6F777E1}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion} ({#EditionName})
AppPublisher={#AppPublisher}
MinVersion=10.0.22000
DefaultDirName={localappdata}\Diarix
DefaultGroupName=Diarix
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=Diarix {#EditionName} Setup {#AppVersion}
UninstallDisplayIcon={app}\Diarix.exe
Compression=lzma2/fast
SolidCompression=no
LZMANumBlockThreads=2
LZMADictionarySize=64
#if EditionName == "Full CUDA"
DiskSpanning=yes
DiskSliceSize=2000000000
#else
DiskSpanning=no
#endif
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern
DisableWelcomePage=no
DisableDirPage=no
DisableReadyPage=no
CloseApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Diarix"; Filename: "{app}\Diarix.exe"
Name: "{autodesktop}\Diarix"; Filename: "{app}\Diarix.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Diarix.exe"; Description: "Launch Diarix"; Flags: nowait postinstall skipifsilent

[Code]
const
  WebView2Key = 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  WebView2MachineKey = 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  VCRuntimeKey = 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  WebView2Url = 'https://developer.microsoft.com/microsoft-edge/webview2/';
  VCRuntimeUrl = 'https://aka.ms/vc14/vc_redist.x64.exe';
  NvidiaDriverUrl = 'https://www.nvidia.com/Download/index.aspx';

var
  PrerequisitePage: TOutputMsgMemoWizardPage;
  MissingWebView2: Boolean;
  MissingVCRuntime: Boolean;
  MissingNvidiaDriver: Boolean;
  DependencyLinksOpened: Boolean;

function WebView2Installed: Boolean;
var
  Version: String;
begin
  Result :=
    (RegQueryStringValue(HKLM64, WebView2MachineKey, 'pv', Version) and
      (Version <> '') and (Version <> '0.0.0.0')) or
    (RegQueryStringValue(HKCU, WebView2Key, 'pv', Version) and
      (Version <> '') and (Version <> '0.0.0.0'));
end;

function VCRuntimeInstalled: Boolean;
var
  Installed: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM64, VCRuntimeKey, 'Installed', Installed) and
    (Installed = 1);
end;

function NvidiaDriverInstalled: Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\nvcuda.dll'));
end;

procedure OpenDependencyUrl(const Url: String);
var
  ErrorCode: Integer;
begin
  if not ShellExec('open', Url, '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode) then
    MsgBox('Windows could not open this download page:' + #13#10 + Url,
      mbError, MB_OK);
end;

procedure InitializeWizard;
var
  Status: String;
begin
  MissingWebView2 := not WebView2Installed;
  MissingVCRuntime := not VCRuntimeInstalled;
  MissingNvidiaDriver :=
    (CompareText('{#EditionName}', 'Full CUDA') = 0) and
    (not NvidiaDriverInstalled);

  Status := 'Diarix includes its local server, FFmpeg, and FFprobe. Python and the CUDA Toolkit are not required.' + #13#10#13#10;

  if not (MissingWebView2 or MissingVCRuntime or MissingNvidiaDriver) then
    Status := Status + 'Ready: all external system prerequisites were detected.'
  else
  begin
    Status := Status + 'The following external prerequisite(s) were not detected:' + #13#10;
    if MissingWebView2 then
      Status := Status + #13#10 + '  - Microsoft Edge WebView2 Runtime (required for the app window)';
    if MissingVCRuntime then
      Status := Status + #13#10 + '  - Microsoft Visual C++ 2015-2022 x64 Runtime';
    if MissingNvidiaDriver then
      Status := Status + #13#10 + '  - NVIDIA display driver (required for GPU acceleration)';
    Status := Status + #13#10#13#10 +
      'Setup can continue. The Full CUDA edition keeps its compact CPU fallback available if NVIDIA acceleration is not ready.';
  end;

  PrerequisitePage := CreateOutputMsgMemoPage(
    wpSelectDir,
    'System check',
    'Diarix prerequisites',
    'Review the detected system dependencies before installation.',
    Status);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID <> PrerequisitePage.ID) or DependencyLinksOpened or
    not (MissingWebView2 or MissingVCRuntime or MissingNvidiaDriver) then
    Exit;

  DependencyLinksOpened := True;
  if MsgBox(
    'Open the official download page for each missing prerequisite now?' + #13#10#13#10 +
    'You can return to this setup after installing them.',
    mbConfirmation, MB_YESNO) = IDYES then
  begin
    if MissingWebView2 then
      OpenDependencyUrl(WebView2Url);
    if MissingVCRuntime then
      OpenDependencyUrl(VCRuntimeUrl);
    if MissingNvidiaDriver then
      OpenDependencyUrl(NvidiaDriverUrl);
  end;
end;
