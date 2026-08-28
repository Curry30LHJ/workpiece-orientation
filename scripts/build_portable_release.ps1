param(
    [ValidateSet('gpu','cpu','all')][string]$Edition = 'all',
    [string]$Version = '1.0.0',
    [string]$GpuPython = 'E:\python\anaconda3\envs\shitu\python.exe',
    [string]$CpuPython = 'E:\python\anaconda3\envs\shitu\python.exe',
    [string]$QtBin = 'E:\QT\5.14\5.14.2\msvc2017_64\bin',
    [string]$WindeployQt = '',
    [string]$SmokeScript = '',
    [string]$ModelPath = '',
    [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'release_staging')
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$gitCommit = (& git -C $repo rev-parse HEAD 2>$null); if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitCommit)) { throw 'Unable to determine git commit' }
$stagingRoot = [IO.Path]::GetFullPath($OutputRoot)
$ownedRootPath = [IO.Path]::GetFullPath((Join-Path $repo 'release_staging')).TrimEnd('\')
$ownedRoot = $ownedRootPath + '\'
if ($stagingRoot -ne $ownedRootPath -and -not $stagingRoot.StartsWith($ownedRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'OutputRoot must remain under release_staging' }
$windeploy = if ($WindeployQt) { $WindeployQt } else { Join-Path $QtBin 'windeployqt.exe' }
if (-not (Test-Path -LiteralPath $windeploy)) { throw "windeployqt not found: $windeploy" }
$editions = if ($Edition -eq 'all') { @('gpu','cpu') } else { @($Edition) }
$qtBuild = Join-Path $repo "qt_app\build-portable-$Version"
if (Test-Path -LiteralPath $qtBuild) { Remove-Item -LiteralPath $qtBuild -Recurse -Force }
& (Join-Path $repo 'scripts\build_qt5.ps1') -ProjectRoot $repo -QtBin $QtBin -BuildDir $qtBuild -SkipRuntimeConfig
$qtRelease = Join-Path $qtBuild 'release'
& $windeploy '--release' '--compiler-runtime' '--no-translations' (Join-Path $qtRelease 'WorkpieceOrientation.exe')
foreach ($ed in $editions) {
    $target = Join-Path $stagingRoot "$ed-$Version"
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    $py = if ($ed -eq 'gpu') { $GpuPython } else { $CpuPython }
    & (Join-Path $repo 'scripts\build_portable_backend.ps1') -Edition $ed -Python $py -ProjectRoot $repo -OutputRoot $stagingRoot
    $backend = Join-Path $stagingRoot "backend-$ed\orientation_backend"
    $model = if ($ModelPath) { $ModelPath } else { Join-Path $repo 'models\shitu_rec' }
    & $py -m release_tools.portable_package stage --edition $ed --version $Version --qt-release-dir $qtRelease --backend-dir $backend --model-dir $model --output-root $stagingRoot --guide (Join-Path $repo 'deploy\使用说明.txt') --notices (Join-Path $repo 'deploy\THIRD_PARTY-NOTICES.txt') --git-commit $gitCommit
    $package = Join-Path $target "WorkpieceOrientation-$($ed.ToUpper())"
    $paddleDist = if ($ed -eq 'gpu') { 'paddlepaddle-gpu' } else { 'paddlepaddle' }
    & $py -c "from release_tools.portable_package import collect_licenses; from pathlib import Path; collect_licenses(Path(r'$package')/'third_party_licenses', distributions=('pyinstaller','$paddleDist','paddleclas','numpy','opencv-python'), python_license=Path(r'$py').parent/'LICENSE.txt')"
    if ($SmokeScript -and (Test-Path -LiteralPath $SmokeScript)) { & $SmokeScript -PythonExecutable $py -ProjectRoot $package }
    $label = $ed.ToUpper()
    & $py -c "from release_tools.portable_package import audit_package,write_manifest,zip_package,write_sha256; from pathlib import Path; p=Path(r'$package'); roots=[Path(r'$repo'),Path(r'$QtBin'),Path(r'$py').parent,Path.home()]; audit_package(p,edition='$ed',version='$Version',forbidden_roots=roots,runtime_roots=roots); write_manifest(p,edition='$ed',version='$Version'); audit_package(p,edition='$ed',version='$Version',forbidden_roots=roots,runtime_roots=roots); a=zip_package(p,Path(r'$repo')/'release_artifacts'/f'WorkpieceOrientation-$label-x64-$Version.zip'); write_sha256(a)"
}
