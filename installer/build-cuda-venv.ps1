# Builds a fresh CUDA-capable Python venv with both the TTS stack
# (requirements.txt) and the advanced-ASR stack (requirements-advanced-asr.txt),
# mirroring justfile's windows setup-python + the release workflow's CUDA
# job. Kept separate from Z:\#####Transcription\Python311 (ASR-only) so that
# working env is never touched.
#
# C: only has ~32GB free, so pip's cache is redirected to Z: to avoid
# filling the system drive with cu128 torch / NeMo wheels.

$ErrorActionPreference = 'Stop'

$RepoRoot = 'Z:\Diarix Studio\diarix-voicebox-upstream-fork-20260713'
$VenvDir  = 'Z:\Diarix Studio\diarix-cuda-venv'
$Python   = "$VenvDir\Scripts\python.exe"
$LogDir   = "$RepoRoot\installer\build-logs"
$AffinityMask = 0x3FFF

New-Item -ItemType Directory -Force $LogDir | Out-Null
$env:PIP_CACHE_DIR = 'Z:\Diarix Studio\pip-cache'

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
        $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
        $proc.ProcessorAffinity = [IntPtr]$AffinityMask
    } catch { Write-Warning "Could not throttle ${Name}: $_" }
    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit()
    Set-Content -Path $log -Value $stdout.Result
    Set-Content -Path "$log.err" -Value $stderr.Result
    if ($proc.ExitCode -ne 0) { throw "$Name failed (exit $($proc.ExitCode)) - see $log / $log.err" }
    # Report free space after each heavy step so a slow drive-fill is caught early.
    $freeGb = [math]::Round((Get-PSDrive Z).Free / 1GB, 1)
    Write-Host "   Z: free after ${Name}: ${freeGb} GB"
    if ($freeGb -lt 5) { throw "Z: drive below 5GB free after $Name - stopping to avoid filling the disk." }
}

if (-not (Test-Path $Python)) {
    Invoke-Throttled 'cuda-venv-create' "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" @('-m', 'venv', $VenvDir) $RepoRoot
}
Invoke-Throttled 'cuda-venv-pip-upgrade' $Python @('-m', 'pip', 'install', '--upgrade', 'pip') $RepoRoot

# 1. CUDA-enabled torch first (matches justfile's Windows setup-python)
Invoke-Throttled 'cuda-venv-torch' $Python @('-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu128') $RepoRoot

# 2. Base requirements (includes qwen-tts>=0.0.5 directly from PyPI)
Invoke-Throttled 'cuda-venv-requirements' $Python @('-m', 'pip', 'install', '-r', 'backend/requirements.txt') $RepoRoot
Invoke-Throttled 'cuda-venv-chatterbox' $Python @('-m', 'pip', 'install', '--no-deps', 'chatterbox-tts') $RepoRoot
Invoke-Throttled 'cuda-venv-tada' $Python @('-m', 'pip', 'install', '--no-deps', 'hume-tada') $RepoRoot

# 3. Advanced-ASR stack (WhisperX, Faster-Whisper, NeMo, Qwen3-ASR) - this
# is the heaviest, slowest step (NeMo pulls in a large dependency tree).
Invoke-Throttled 'cuda-venv-advanced-asr' $Python @('-m', 'pip', 'install', '-r', 'backend/requirements-advanced-asr.txt') $RepoRoot

# 4. torchcodec metadata (see run-throttled-build.ps1 for why this is needed)
Invoke-Throttled 'cuda-venv-torchcodec' $Python @('-m', 'pip', 'install', 'torchcodec') $RepoRoot

# 5. PyInstaller
Invoke-Throttled 'cuda-venv-pyinstaller' $Python @('-m', 'pip', 'install', 'pyinstaller') $RepoRoot

Write-Host "CUDA build venv ready at $VenvDir"
