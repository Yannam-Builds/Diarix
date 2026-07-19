# Resumes run-throttled-build.ps1 from the cpu-server step (venv + torchcodec
# already installed). Sources the same Invoke-Throttled helper by dot-sourcing
# a trimmed copy of the parent script's function + step list.

$ErrorActionPreference = 'Stop'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$Toolchains = 'Z:\Diarix Studio\Toolchains'
$CudaPython = 'Z:\Diarix Studio\diarix-cuda-venv\Scripts\python.exe'
$VenvPython = "$RepoRoot\backend\venv\Scripts\python.exe"
$LogDir     = "$RepoRoot\installer\build-logs"
$PayloadDir = 'Z:\Diarix Studio\Diarix Setup Payload 0.1.0'
$BuildCpuPercent = 25
$BuildProcessors = [Math]::Max(1, [Math]::Floor([Environment]::ProcessorCount * ($BuildCpuPercent / 100)))
$AffinityMask = if ($BuildProcessors -ge 63) {
    [Int64]::MaxValue
} else {
    ([Int64]1 -shl $BuildProcessors) - 1
}

$env:PATH        = "$Toolchains\bun;$Toolchains\cargo\bin;$env:PATH"
$env:RUSTUP_HOME = "$Toolchains\rustup"
$env:CARGO_HOME  = "$Toolchains\cargo"

function Invoke-Throttled {
    param([string]$Name, [string]$Exe, [string[]]$Arguments, [string]$WorkDir)
    Write-Host "== $Name =="
    $log = "$LogDir\$Name.log"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
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
        $proc.ProcessorAffinity = [IntPtr]$AffinityMask
    } catch { Write-Warning "Could not throttle ${Name}: $_" }
    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit()
    Set-Content -Path $log -Value $stdout.Result
    Set-Content -Path "$log.err" -Value $stderr.Result
    if ($proc.ExitCode -ne 0) { throw "$Name failed (exit $($proc.ExitCode)) - see $log / $log.err" }
}

Invoke-Throttled 'cpu-server' $VenvPython @('build_binary.py', '--require-media-tools') "$RepoRoot\backend"
Invoke-Throttled 'cuda-server' $CudaPython @('build_binary.py', '--cuda', '--require-media-tools') "$RepoRoot\backend"
Invoke-Throttled 'cuda-self-test' "$RepoRoot\backend\dist\diarix-server-cuda\diarix-server-cuda.exe" @('--runtime-self-test') "$RepoRoot\backend\dist\diarix-server-cuda"

$triple = (& "$Toolchains\cargo\bin\rustc.exe" --print host-tuple).Trim()
New-Item -ItemType Directory -Force "$RepoRoot\tauri\src-tauri\binaries" | Out-Null
Copy-Item "$RepoRoot\backend\dist\diarix-server.exe" "$RepoRoot\tauri\src-tauri\binaries\diarix-server-$triple.exe" -Force
Invoke-Throttled 'bun-install' "$Toolchains\bun\node_modules\bun\bin\bun.exe" @('install') $RepoRoot
Invoke-Throttled 'tauri-build' "$Toolchains\bun\node_modules\bun\bin\bun.exe" @('tauri', 'build', '--no-bundle') "$RepoRoot\tauri"

Write-Host '== Assembling payload =='
if (Test-Path $PayloadDir) { Remove-Item -Recurse -Force $PayloadDir }
New-Item -ItemType Directory -Force "$PayloadDir\backends\cuda" | Out-Null
Copy-Item "$RepoRoot\tauri\src-tauri\target\release\diarix.exe" "$PayloadDir\Diarix.exe" -Force
Copy-Item "$RepoRoot\backend\dist\diarix-server.exe" "$PayloadDir\diarix-server-$triple.exe" -Force
Copy-Item "$RepoRoot\backend\dist\diarix-server-cuda\*" "$PayloadDir\backends\cuda" -Recurse -Force

Write-Host "Payload ready at $PayloadDir"
