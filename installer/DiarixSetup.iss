#define AppName "Diarix"
#define AppVersion "0.1.0"
#define AppPublisher "Diarix"
#define PayloadDir "Z:\\Diarix Studio\\Diarix Setup Payload 0.1.0"
#define OutputDir "Z:\\Diarix Studio\\Diarix Setup 0.1.0"

[Setup]
AppId={{B1A3B6F9-65E5-46D0-9E91-92D7B6F777E1}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Diarix
DefaultGroupName=Diarix
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=Diarix Setup
UninstallDisplayIcon={app}\Diarix.exe
Compression=lzma2/fast
SolidCompression=no
LZMANumBlockThreads=2
LZMADictionarySize=64
DiskSpanning=yes
DiskSliceSize=2000000000
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
