# Assemble the Diarix 0.1.0 standalone installer: Diarix.exe + CPU sidecar +
# CUDA onedir backend, NO pre-downloaded starter models (unlike the 0.5.0
# CUDA release). First app launch starts with an empty model cache and the
# user downloads models from the Models page.
#
# Prerequisites (all local to this machine):
#   - Toolchains: Z:\Diarix Studio\Toolchains  (bun, cargo/rustup)
#   - Python env with backend + advanced-ASR deps: Z:\#####Transcription\Python311
#   - Inno Setup 6 (ISCC.exe) for the final installer compile
#
# Steps (each is idempotent; re-run from any point):
#   1. CPU server     -> backend/dist/diarix-server.exe
#   2. CUDA onedir    -> backend/dist/diarix-server-cuda/
#   3. Tauri release  -> tauri/src-tauri/target/release/Diarix.exe
#   4. Payload        -> Z:\Diarix Studio\Diarix Setup Payload 0.1.0
#   5. Installer      -> Z:\Diarix Studio\Diarix Setup 0.1.0\Diarix Setup.exe (+ .bin slices)
#
# MCP (agent integration) is not built or bundled here - unwired from the
# backend for now (see backend/app.py create_app()), can be reinstated later.

$ErrorActionPreference = 'Stop'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$Toolchains = 'Z:\Diarix Studio\Toolchains'
$Python     = 'Z:\#####Transcription\Python311\python.exe'
$PayloadDir = 'Z:\Diarix Studio\Diarix Setup Payload 0.1.0'
$OutputDir  = 'Z:\Diarix Studio\Diarix Setup 0.1.0'

$env:PATH        = "$Toolchains\bun;$Toolchains\cargo\bin;$env:PATH"
$env:RUSTUP_HOME = "$Toolchains\rustup"
$env:CARGO_HOME  = "$Toolchains\cargo"

Set-Location $RepoRoot

# --- 1. CPU server ----------------------------------------------------------
Write-Host '== Building CPU server (diarix-server.exe) =='
Set-Location "$RepoRoot\backend"
& $Python build_binary.py --require-media-tools
if ($LASTEXITCODE -ne 0) { throw 'CPU server build failed' }

# --- 2. CUDA onedir --------------------------------------------------------
Write-Host '== Building CUDA onedir (diarix-server-cuda) =='
& $Python build_binary.py --cuda --require-media-tools
if ($LASTEXITCODE -ne 0) { throw 'CUDA server build failed' }

# Frozen-runtime self test: all registered engines must import inside the
# bundle before we ship it (mirrors the 2026-07-15 release gate).
Write-Host '== Runtime self-test on the frozen CUDA bundle =='
& "$RepoRoot\backend\dist\diarix-server-cuda\diarix-server-cuda.exe" --runtime-self-test
if ($LASTEXITCODE -ne 0) { throw 'Frozen runtime self-test failed' }

# --- 4. Tauri release ------------------------------------------------------
Write-Host '== Building Tauri release (Diarix.exe) =='
Set-Location $RepoRoot
$triple = (& rustc --print host-tuple).Trim()
New-Item -ItemType Directory -Force "$RepoRoot\tauri\src-tauri\binaries" | Out-Null
Copy-Item "$RepoRoot\backend\dist\diarix-server.exe" "$RepoRoot\tauri\src-tauri\binaries\diarix-server-$triple.exe" -Force
bun install
if ($LASTEXITCODE -ne 0) { throw 'bun install failed' }
Set-Location "$RepoRoot\tauri"
bun tauri build
if ($LASTEXITCODE -ne 0) { throw 'tauri build failed' }

# --- 5. Payload assembly (no starter models) -------------------------------
Write-Host '== Assembling payload =='
if (Test-Path $PayloadDir) { Remove-Item -Recurse -Force $PayloadDir }
New-Item -ItemType Directory -Force "$PayloadDir\backends\cuda" | Out-Null

Copy-Item "$RepoRoot\tauri\src-tauri\target\release\Diarix.exe" $PayloadDir -Force
# CPU sidecar ships beside the app under its Tauri sidecar name so
# app.shell().sidecar("diarix-server") resolves in the installed layout.
Copy-Item "$RepoRoot\backend\dist\diarix-server.exe" "$PayloadDir\diarix-server-$triple.exe" -Force
Copy-Item "$RepoRoot\backend\dist\diarix-server-cuda\*" "$PayloadDir\backends\cuda" -Recurse -Force

# Deliberately NO models/ or starter Hugging Face cache in this payload.

# --- 6. Installer ----------------------------------------------------------
Write-Host '== Compiling installer =='
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$iscc = @(
  "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
  "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
  "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw 'ISCC.exe (Inno Setup 6) not found - install it or add its path here.' }
& $iscc "$RepoRoot\installer\DiarixSetup.iss"
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup compile failed' }

Write-Host "Done. Installer written to $OutputDir"
