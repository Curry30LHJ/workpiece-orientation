param(
    [ValidateSet('gpu','cpu','all')][string]$Edition = 'all',
    [string]$Version = '1.0.0',
    [string]$GpuPython = 'E:\python\anaconda3\envs\shitu\python.exe',
    [string]$CpuPython = 'E:\python\anaconda3\envs\shitu\python.exe',
    [string]$QtBin = 'E:\QT\5.14\5.14.2\msvc2017_64\bin',
    [string]$WindeployQt = '',
    [switch]$SkipSmoke,
    [string]$SmokeScript = '',
    [string]$SmokeDatasetRoot = '',
    [int]$SmokeFrontTemplateCount = 5,
    [int]$SmokeBackTemplateCount = 10,
    [int]$SmokeSeed = 20260813,
    [string]$ModelPath = '',
    [string]$PaddleConfig = '',
    [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'release_staging')
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) { $repo } else { "$repo;$env:PYTHONPATH" }
$gitCommit = (& git -C $repo rev-parse HEAD 2>$null); if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitCommit)) { throw 'Unable to determine git commit' }
$stagingRoot = [IO.Path]::GetFullPath($OutputRoot)
$ownedRootPath = [IO.Path]::GetFullPath((Join-Path $repo 'release_staging')).TrimEnd('\')
$ownedRoot = $ownedRootPath + '\'
if ($stagingRoot -ne $ownedRootPath -and -not $stagingRoot.StartsWith($ownedRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'OutputRoot must remain under release_staging' }
$editions = if ($Edition -eq 'all') { @('gpu','cpu') } else { @($Edition) }
if ($SmokeScript) {
    throw 'SmokeScript is no longer supported; use the built-in smoke CLI (or -SkipSmoke explicitly).'
}
if (-not $SkipSmoke -and [string]::IsNullOrWhiteSpace($SmokeDatasetRoot)) {
    throw 'SmokeDatasetRoot is required unless -SkipSmoke is explicitly specified.'
}
$windeploy = if ($WindeployQt) { $WindeployQt } else { Join-Path $QtBin 'windeployqt.exe' }
if (-not (Test-Path -LiteralPath $windeploy)) { throw "windeployqt not found: $windeploy" }
$qtBuild = Join-Path $repo "qt_app\build-portable-$Version"
if (Test-Path -LiteralPath $qtBuild) { Remove-Item -LiteralPath $qtBuild -Recurse -Force }
& (Join-Path $repo 'scripts\build_qt5.ps1') -ProjectRoot $repo -QtBin $QtBin -BuildDir $qtBuild -SkipRuntimeConfig
if ($LASTEXITCODE -ne 0) { throw "Qt build failed with exit code $LASTEXITCODE" }
$qtRelease = Join-Path $qtBuild 'release'
$qtExecutable = Join-Path $qtRelease 'WorkpieceOrientation.exe'
if (-not (Test-Path -LiteralPath $qtExecutable)) { throw "Qt executable not found: $qtExecutable" }
# windeployqt resolves Qt plugins and runtime DLLs through PATH.  A machine may
# have Qt6 (or another Qt installation) earlier in PATH than the requested Qt5
# toolchain; prepend the selected Qt bin directory for this call only.
$previousPath = $env:PATH
try {
    $env:PATH = "$([IO.Path]::GetFullPath($QtBin));$previousPath"
    & $windeploy '--release' '--compiler-runtime' '--no-translations' $qtExecutable
    if ($LASTEXITCODE -ne 0) { throw "windeployqt failed with exit code $LASTEXITCODE" }
}
finally {
    $env:PATH = $previousPath
}
foreach ($ed in $editions) {
    $target = Join-Path $stagingRoot "$ed-$Version"
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    $py = if ($ed -eq 'gpu') { $GpuPython } else { $CpuPython }
    & (Join-Path $repo 'scripts\build_portable_backend.ps1') -Edition $ed -Python $py -ProjectRoot $repo -OutputRoot $stagingRoot
    if ($LASTEXITCODE -ne 0) { throw "Portable backend build failed for $ed with exit code $LASTEXITCODE" }
    $backend = Join-Path $stagingRoot "backend-$ed\orientation_backend"
    $model = if ($ModelPath) { $ModelPath } else { Join-Path $repo 'models\shitu_rec' }
    $configArg = if ($PaddleConfig) { $PaddleConfig } else { Join-Path $repo 'deploy\configs\inference_general.yaml' }
    if (-not (Test-Path -LiteralPath $configArg)) {
        $configArg = Join-Path $repo 'third_party\PaddleClas\deploy\configs\inference_general.yaml'
    }
    if (-not (Test-Path -LiteralPath $configArg)) { throw "Paddle inference config not found: $configArg" }
    & $py -m release_tools.portable_package stage --edition $ed --version $Version --qt-release-dir $qtRelease --backend-dir $backend --model-dir $model --output-root $stagingRoot --guide (Join-Path $repo 'deploy\使用说明.txt') --notices (Join-Path $repo 'deploy\THIRD_PARTY-NOTICES.txt') --git-commit $gitCommit --repository-root $repo --paddle-config $configArg
    if ($LASTEXITCODE -ne 0) { throw "Portable package staging failed for $ed with exit code $LASTEXITCODE" }
    $package = Join-Path $target "WorkpieceOrientation-$($ed.ToUpper())"
    $paddleDist = if ($ed -eq 'gpu') { 'paddlepaddle-gpu' } else { 'paddlepaddle' }
    $pythonLicense = Join-Path (Split-Path $py -Parent) 'LICENSE_PYTHON.txt'
    if (-not (Test-Path -LiteralPath $pythonLicense)) { $pythonLicense = Join-Path (Split-Path $py -Parent) 'LICENSE.txt' }
    if (-not (Test-Path -LiteralPath $pythonLicense)) { throw "Python license not found beside interpreter: $pythonLicense" }
    & $py -c "from release_tools.portable_package import collect_licenses; from pathlib import Path; collect_licenses(Path(r'$package')/'third_party_licenses', distributions=('pyinstaller','$paddleDist','paddleclas','numpy','opencv-python'), python_license=Path(r'$pythonLicense'))"
    if ($LASTEXITCODE -ne 0) { throw "License collection failed for $ed with exit code $LASTEXITCODE" }
    if (-not $SkipSmoke) {
        & $py (Join-Path $repo 'scripts\smoke_portable_package.py') --package-root $package --dataset-root $SmokeDatasetRoot --front-template-count $SmokeFrontTemplateCount --back-template-count $SmokeBackTemplateCount --seed $SmokeSeed --report (Join-Path $stagingRoot 'reports' "$ed-smoke.json")
        if ($LASTEXITCODE -ne 0) { throw "Portable smoke failed for $ed" }
    }
    $label = $ed.ToUpper()
    & $py -c "from release_tools.portable_package import audit_package,write_manifest,zip_package,write_sha256; from pathlib import Path; p=Path(r'$package'); roots=[Path(r'$repo'),Path(r'$QtBin'),Path(r'$py').parent,Path.home()]; audit_package(p,edition='$ed',version='$Version',forbidden_roots=roots,runtime_roots=roots); write_manifest(p,edition='$ed',version='$Version'); audit_package(p,edition='$ed',version='$Version',forbidden_roots=roots,runtime_roots=roots); a=zip_package(p,Path(r'$repo')/'release_artifacts'/f'WorkpieceOrientation-$label-x64-$Version.zip'); write_sha256(a)"
    if ($LASTEXITCODE -ne 0) { throw "Package audit/archive failed for $ed with exit code $LASTEXITCODE" }
}
