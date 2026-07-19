[CmdletBinding()]
param(
    [string]$SourceLogo = '',
    [string]$Bun = 'Z:\Diarix Studio\Toolchains\bun\bun.cmd'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $SourceLogo) {
    $SourceLogo = Join-Path $RepoRoot 'app\src\assets\diarix-logo.png'
}

$SourceLogo = [IO.Path]::GetFullPath($SourceLogo)
if (-not (Test-Path -LiteralPath $SourceLogo)) {
    throw "Approved Diarix logo not found: $SourceLogo"
}
if (-not (Test-Path -LiteralPath $Bun)) {
    throw "Bun runtime not found: $Bun"
}

Add-Type -AssemblyName System.Drawing

$artifactsRoot = Join-Path $RepoRoot 'artifacts\icon-generation'
$generatedRoot = Join-Path $artifactsRoot 'tauri'
$masterPath = Join-Path $artifactsRoot 'diarix-shell-icon-master.png'
$iconRoot = Join-Path $RepoRoot 'tauri\src-tauri\icons'

if (Test-Path -LiteralPath $artifactsRoot) {
    Remove-Item -LiteralPath $artifactsRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $artifactsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $generatedRoot -Force | Out-Null

$size = 1024
$bitmap = New-Object System.Drawing.Bitmap(
    $size,
    $size,
    [System.Drawing.Imaging.PixelFormat]::Format32bppArgb
)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
$graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
$graphics.Clear([System.Drawing.Color]::Transparent)

$inset = 36
$diameter = $size - ($inset * 2)
$radius = 214
$path = New-Object System.Drawing.Drawing2D.GraphicsPath
$path.AddArc($inset, $inset, $radius * 2, $radius * 2, 180, 90)
$path.AddArc($inset + $diameter - ($radius * 2), $inset, $radius * 2, $radius * 2, 270, 90)
$path.AddArc(
    $inset + $diameter - ($radius * 2),
    $inset + $diameter - ($radius * 2),
    $radius * 2,
    $radius * 2,
    0,
    90
)
$path.AddArc($inset, $inset + $diameter - ($radius * 2), $radius * 2, $radius * 2, 90, 90)
$path.CloseFigure()

$background = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 17, 16, 13))
$border = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 184, 148, 49), 28)
$border.Alignment = [System.Drawing.Drawing2D.PenAlignment]::Inset
$graphics.FillPath($background, $path)
$graphics.DrawPath($border, $path)

$logo = [System.Drawing.Image]::FromFile($SourceLogo)
$logoInset = 26
$logoSize = $size - ($logoInset * 2)
$logoRectangle = [System.Drawing.Rectangle]::new($logoInset, $logoInset, $logoSize, $logoSize)
$graphics.DrawImage($logo, $logoRectangle)

$bitmap.Save($masterPath, [System.Drawing.Imaging.ImageFormat]::Png)

$logo.Dispose()
$border.Dispose()
$background.Dispose()
$path.Dispose()
$graphics.Dispose()
$bitmap.Dispose()

Push-Location (Join-Path $RepoRoot 'tauri')
try {
    & $Bun 'tauri' 'icon' $masterPath '--output' $generatedRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Tauri icon generation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

foreach ($name in @('32x32.png', '128x128.png', '128x128@2x.png', 'icon.ico', 'icon.icns')) {
    $source = Join-Path $generatedRoot $name
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Tauri did not generate the expected icon: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $iconRoot $name) -Force
}

Copy-Item -LiteralPath $masterPath -Destination (Join-Path $iconRoot 'icon-shell-master.png') -Force
Write-Host "Shell icons generated from $SourceLogo"
