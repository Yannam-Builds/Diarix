[CmdletBinding()]
param(
    [ValidateSet('Core', 'Whisper', 'FullCuda')]
    [string]$Edition = 'FullCuda',

    [string]$Version = '0.1.0-alpha.1',
    [string]$ArtifactsDir = '',
    [string]$ToolchainsDir = 'Z:\Diarix Studio\Toolchains',
    [string]$CpuPython = '',
    [string]$CudaPython = 'Z:\Diarix Studio\diarix-cuda-venv\Scripts\python.exe',
    [string]$StarterModelsPath = '',
    [ValidateRange(10, 100)]
    [int]$BuildCpuPercent = 25,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $ArtifactsDir) {
    $ArtifactsDir = Join-Path $RepoRoot 'artifacts'
}
if (-not $CpuPython) {
    $CpuPython = Join-Path $RepoRoot 'backend\venv\Scripts\python.exe'
}

$ArtifactsDir = [IO.Path]::GetFullPath($ArtifactsDir)
$EditionSlug = switch ($Edition) {
    'Core' { 'core' }
    'Whisper' { 'whisper' }
    'FullCuda' { 'full-cuda' }
}
$EditionName = switch ($Edition) {
    'Core' { 'Core' }
    'Whisper' { 'Whisper' }
    'FullCuda' { 'Full CUDA' }
}
$PayloadDir = Join-Path $ArtifactsDir "payloads\$EditionSlug-$Version"
$OutputDir = Join-Path $ArtifactsDir 'installers'

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments = @())

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Reset-ArtifactDirectory {
    param([string]$Path)

    $resolvedRoot = [IO.Path]::GetFullPath($ArtifactsDir).TrimEnd('\') + '\'
    $resolvedPath = [IO.Path]::GetFullPath($Path)
    if (-not $resolvedPath.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace a directory outside the artifacts root: $resolvedPath"
    }
    if (Test-Path -LiteralPath $resolvedPath) {
        Remove-Item -LiteralPath $resolvedPath -Recurse -Force
    }
    New-Item -ItemType Directory -Path $resolvedPath -Force | Out-Null
}

$bun = Join-Path $ToolchainsDir 'bun\bun.cmd'
$cargoBin = Join-Path $ToolchainsDir 'cargo\bin'
$rustupHome = Join-Path $ToolchainsDir 'rustup'
$rustBin = Join-Path $rustupHome 'toolchains\stable-x86_64-pc-windows-msvc\bin'
$rustc = Join-Path $rustBin 'rustc.exe'
foreach ($required in @($bun, $CpuPython, $rustc)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required build tool not found: $required"
    }
}
if ($Edition -eq 'FullCuda' -and -not (Test-Path -LiteralPath $CudaPython)) {
    throw "FullCuda requires the CUDA build environment: $CudaPython"
}
if ($Edition -eq 'Whisper' -and (-not $StarterModelsPath -or -not (Test-Path -LiteralPath $StarterModelsPath))) {
    throw 'Whisper edition requires -StarterModelsPath pointing to the starter model directory.'
}

$logicalProcessors = [Environment]::ProcessorCount
$buildJobs = [Math]::Max(1, [Math]::Floor($logicalProcessors * ($BuildCpuPercent / 100)))
$env:PATH = "$($ToolchainsDir)\bun;$rustBin;$cargoBin;$env:PATH"
$env:RUSTUP_HOME = $rustupHome
$env:CARGO_HOME = Join-Path $ToolchainsDir 'cargo'
$env:CARGO_BUILD_JOBS = "$buildJobs"
$env:MAX_JOBS = "$buildJobs"
$env:OMP_NUM_THREADS = "$buildJobs"
$env:MKL_NUM_THREADS = "$buildJobs"

Set-Location $RepoRoot
Invoke-Checked 'node' @('scripts/verify-alpha.mjs')

if (-not $SkipBuild) {
    Write-Host "== Building compact server with $buildJobs worker(s) =="
    Set-Location (Join-Path $RepoRoot 'backend')
    Invoke-Checked $CpuPython @('build_binary.py', '--require-media-tools')

    if ($Edition -eq 'FullCuda') {
        Write-Host "== Building CUDA server with $buildJobs worker(s) =="
        Invoke-Checked $CudaPython @('build_binary.py', '--cuda', '--require-media-tools')
        Invoke-Checked (Join-Path $RepoRoot 'backend\dist\diarix-server-cuda\diarix-server-cuda.exe') @('--runtime-self-test')
    }

    Write-Host '== Building Tauri desktop =='
    Set-Location $RepoRoot
    Invoke-Checked $bun @('install', '--frozen-lockfile')
    $triple = (& $rustc --print host-tuple).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve the Rust host tuple.' }
    $sidecarDir = Join-Path $RepoRoot 'tauri\src-tauri\binaries'
    New-Item -ItemType Directory -Path $sidecarDir -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $RepoRoot 'backend\dist\diarix-server.exe') `
        -Destination (Join-Path $sidecarDir "diarix-server-$triple.exe") -Force
    Set-Location (Join-Path $RepoRoot 'tauri')
    # The edition installers are assembled by Inno Setup below. Building the
    # desktop executable without Tauri's MSI bundler also keeps semantic alpha
    # versions such as 0.1.0-alpha.1 valid on Windows.
    Invoke-Checked $bun @('tauri', 'build', '--no-bundle')
}

$triple = (& $rustc --print host-tuple).Trim()
$desktopExe = Join-Path $RepoRoot 'tauri\src-tauri\target\release\diarix.exe'
$compactServer = Join-Path $RepoRoot 'backend\dist\diarix-server.exe'
foreach ($required in @($desktopExe, $compactServer)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required release artifact not found: $required"
    }
}

Write-Host "== Assembling $EditionName payload =="
Reset-ArtifactDirectory $PayloadDir
Copy-Item -LiteralPath $desktopExe -Destination (Join-Path $PayloadDir 'Diarix.exe') -Force
Copy-Item -LiteralPath $compactServer `
    -Destination (Join-Path $PayloadDir 'diarix-server.exe') -Force

if ($Edition -eq 'FullCuda') {
    $cudaSource = Join-Path $RepoRoot 'backend\dist\diarix-server-cuda'
    if (-not (Test-Path -LiteralPath $cudaSource)) {
        throw "CUDA server output not found: $cudaSource"
    }
    $cudaDestination = Join-Path $PayloadDir 'backends\cuda'
    New-Item -ItemType Directory -Path $cudaDestination -Force | Out-Null
    Copy-Item -Path (Join-Path $cudaSource '*') -Destination $cudaDestination -Recurse -Force
}

if ($Edition -eq 'Whisper') {
    $modelDestination = Join-Path $PayloadDir 'models'
    New-Item -ItemType Directory -Path $modelDestination -Force | Out-Null
    Copy-Item -Path (Join-Path ([IO.Path]::GetFullPath($StarterModelsPath)) '*') `
        -Destination $modelDestination -Recurse -Force
}

$iscc = @(
    "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $iscc) {
    throw 'Inno Setup 6 was not found. Install it before compiling an alpha installer.'
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Write-Host '== Compiling Inno Setup bundle =='
Invoke-Checked $iscc @(
    "/DAppVersion=$Version",
    "/DEditionName=$EditionName",
    "/DPayloadDir=$PayloadDir",
    "/DOutputDir=$OutputDir",
    (Join-Path $PSScriptRoot 'DiarixSetup.iss')
)

Write-Host "Installer complete: $OutputDir"
