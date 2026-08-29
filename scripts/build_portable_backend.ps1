param(
    [ValidateSet('gpu','cpu')][string]$Edition,
    [string]$Python,
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'release_staging')
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Python)) { throw 'Python must point to the edition packaging environment interpreter' }
$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
if (-not (Test-Path -LiteralPath $OutputRoot -PathType Container)) {
    if (Test-Path -LiteralPath $OutputRoot) { throw "OutputRoot is not a directory: $OutputRoot" }
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
}
# Resolve after creation and normalize through DirectoryInfo.FullName.  Some
# PowerShell providers return a DirectoryInfo without a usable `.Path`; using
# the provider path as input and FullName as output is stable for nested roots.
$stagingRoot = ([IO.DirectoryInfo]::new((Resolve-Path -LiteralPath $OutputRoot -ErrorAction Stop).ProviderPath)).FullName
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
$visibleSources = @(Get-ChildItem -LiteralPath $backend -Recurse -File -Include '*.py' | Where-Object {
    $relative = $_.FullName.Substring($backend.Length + 1).Replace('\','/').ToLowerInvariant()
    -not $relative.StartsWith('_internal/cv2/')
})
if ($visibleSources.Count -gt 0) {
    throw ("Frozen backend contains project Python source: " + (($visibleSources | ForEach-Object { $_.FullName }) -join ', '))
}
Write-Output ("Dependency loader sources retained (OpenCV only): " + ((Get-ChildItem -LiteralPath $backend -Recurse -File -Include '*.py' | Measure-Object).Count))
Write-Output ("Backend bundle ready: " + $backend)
