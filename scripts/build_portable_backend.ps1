param(
    [ValidateSet('gpu','cpu')][string]$Edition,
    [string]$Python,
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'release_staging')
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Python)) { throw 'Python must point to the edition packaging environment interpreter' }
$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
$stagingRoot = (Resolve-Path -LiteralPath $OutputRoot -ErrorAction SilentlyContinue)
if ($null -eq $stagingRoot) { $stagingRoot = New-Item -ItemType Directory -Force -Path $OutputRoot }
$stagingRoot = $stagingRoot.Path
$editionRoot = Join-Path $stagingRoot ("backend-" + $Edition)
$distPath = $editionRoot
$workPath = Join-Path $project ("deploy\pyinstaller-work\" + $Edition)
foreach ($target in @($editionRoot, $workPath)) {
    $resolved = [IO.Path]::GetFullPath($target)
    $base = ([IO.Path]::GetFullPath($stagingRoot)).TrimEnd('\') + '\'
    if ($target -eq $workPath) { $base = ([IO.Path]::GetFullPath((Join-Path $project 'deploy\pyinstaller-work'))).TrimEnd('\') + '\' }
    if (-not $resolved.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) { throw "Refusing to clear path outside owned build root: $resolved" }
    if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $resolved | Out-Null
}

$env:WORKPIECE_PACKAGE_EDITION = $Edition
$env:WORKPIECE_PROJECT_ROOT = $project
Push-Location -LiteralPath $project
try {
    & $Python -m PyInstaller --clean --noconfirm --distpath $distPath --workpath $workPath (Join-Path $project 'deploy\orientation_backend.spec')
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }

$backend = Join-Path $editionRoot 'orientation_backend'
$exe = Join-Path $backend 'orientation_backend.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Frozen backend missing orientation_backend.exe: $exe" }
Write-Output ("Backend bundle ready: " + $backend)

