# Orchestrates the full Diarix 0.1.0 build with a ~70% CPU cap:
# every top-level build process runs at BelowNormal priority with a
# 14-of-20-logical-processor affinity mask (0x3FFF); children inherit both.
# Logs land in installer\build-logs\. Run installer\build-diarix-setup.ps1's
# final ISCC step separately once Inno Setup 6 is installed.

$ErrorActionPreference = 'Stop'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$Toolchains = 'Z:\Diarix Studio\Toolchains'
$CudaPython = 'Z:\#####Transcription\Python311\python.exe'
$SysPython  = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
$VenvDir    = "$RepoRoot\backend\venv"
$VenvPython = "$VenvDir\Scripts\python.exe"
$LogDir     = "$RepoRoot\installer\build-logs"
$PayloadDir = 'Z:\Diarix Studio\Diarix Setup Payload 0.1.0'

$AffinityMask = 0x3FFF   # 14 of 20 logical processors ~= 70%

New-Item -ItemType Directory -Force $LogDir | Out-Null

$env:PATH        = "$Toolchains\bun;$Toolchains\cargo\bin;$env:PATH"
$env:RUSTUP_HOME = "$Toolchains\rustup"
$env:CARGO_HOME  = "$Toolchains\cargo"

function Invoke-Throttled {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$Arguments,
        [string]$WorkDir
    )
    Write-Host "== $Name =="
    $log = "$LogDir\$Name.log"

    # Raw System.Diagnostics.Process instead of Start-Process: on this
    # machine, Start-Process -PassThru's returned object never populates
    # ExitCode (even after WaitForExit()+Refresh()), which silently treated
    # every successful step as failed ($null -ne 0 is $true in PowerShell).
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    # ProcessStartInfo.ArgumentList isn't available on this machine's .NET
    # Framework (PowerShell 5.1) - build a properly quoted argument string
    # instead. Each argument is wrapped in double quotes with embedded
    # quotes escaped, which is safe for the plain paths/flags used here.
    $psi.Arguments = ($Arguments | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join ' '
    $psi.WorkingDirectory = $WorkDir
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()

    try {
        $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
        $proc.ProcessorAffinity = [IntPtr]$AffinityMask
    } catch {
        Write-Warning "Could not throttle ${Name}: $_"
    }

    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit()
    Set-Content -Path $log -Value $stdout.Result
    Set-Content -Path "$log.err" -Value $stderr.Result

    if ($proc.ExitCode -ne 0) {
        throw "$Name failed (exit $($proc.ExitCode)) - see $log / $log.err"
    }
}

# --- 1. CPU venv (PyPI torch on Windows is the CPU build, matching CI) -----
if (-not (Test-Path $VenvPython)) {
    Invoke-Throttled 'venv-create' $SysPython @('-m', 'venv', $VenvDir) $RepoRoot
}
Invoke-Throttled 'venv-pip-upgrade' $VenvPython @('-m', 'pip', 'install', '--upgrade', 'pip') $RepoRoot
Invoke-Throttled 'venv-pip-core' $VenvPython @('-m', 'pip', 'install', 'pyinstaller', '-r', 'backend/requirements.txt') $RepoRoot
Invoke-Throttled 'venv-pip-nodeps' $VenvPython @('-m', 'pip', 'install', '--no-deps', 'chatterbox-tts', 'hume-tada') $RepoRoot
# build_binary.py's generated spec unconditionally copies torchcodec's
# package metadata (Transformers 4.57 probes it while importing Whisper,
# even though this build never calls into it) but it isn't a pinned
# requirements.txt dependency, so a from-scratch venv needs it installed
# explicitly or PyInstaller's copy_metadata() raises PackageNotFoundError.
Invoke-Throttled 'venv-pip-torchcodec' $VenvPython @('-m', 'pip', 'install', 'torchcodec') $RepoRoot

# --- 2. CPU server -----------------------------------------------------------
Invoke-Throttled 'cpu-server' $VenvPython @('build_binary.py', '--require-media-tools') "$RepoRoot\backend"

# --- 3. CUDA onedir (existing cu128 env with NeMo/advanced ASR) --------------
Invoke-Throttled 'cuda-server' $CudaPython @('build_binary.py', '--cuda', '--require-media-tools') "$RepoRoot\backend"

# Frozen-runtime gate: every registered engine must import inside the bundle.
Invoke-Throttled 'cuda-self-test' "$RepoRoot\backend\dist\diarix-server-cuda\diarix-server-cuda.exe" @('--runtime-self-test') "$RepoRoot\backend\dist\diarix-server-cuda"

# --- 4. Tauri release ---------------------------------------------------------
$triple = (& "$Toolchains\cargo\bin\rustc.exe" --print host-tuple).Trim()
New-Item -ItemType Directory -Force "$RepoRoot\tauri\src-tauri\binaries" | Out-Null
Copy-Item "$RepoRoot\backend\dist\diarix-server.exe" "$RepoRoot\tauri\src-tauri\binaries\diarix-server-$triple.exe" -Force
# MCP shim (voicebox-mcp) intentionally not built or bundled: MCP is
# unwired from the backend for now (see backend/app.py create_app()).
Invoke-Throttled 'bun-install' "$Toolchains\bun\node_modules\bun\bin\bun.exe" @('install') $RepoRoot
Invoke-Throttled 'tauri-build' "$Toolchains\bun\node_modules\bun\bin\bun.exe" @('tauri', 'build', '--no-bundle') "$RepoRoot\tauri"

# --- 5. Payload assembly (no starter models) ----------------------------------
Write-Host '== Assembling payload =='
if (Test-Path $PayloadDir) { Remove-Item -Recurse -Force $PayloadDir }
New-Item -ItemType Directory -Force "$PayloadDir\backends\cuda" | Out-Null
Copy-Item "$RepoRoot\tauri\src-tauri\target\release\diarix.exe" "$PayloadDir\Diarix.exe" -Force
Copy-Item "$RepoRoot\backend\dist\diarix-server.exe" "$PayloadDir\diarix-server-$triple.exe" -Force
Copy-Item "$RepoRoot\backend\dist\diarix-server-cuda\*" "$PayloadDir\backends\cuda" -Recurse -Force

Write-Host "Payload ready at $PayloadDir"
Write-Host 'Next: install Inno Setup 6, then compile installer\DiarixSetup.iss with ISCC.exe.'
