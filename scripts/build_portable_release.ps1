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
    [string]$MsvcRuntimeDir = '',
    [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'release_staging')
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) { $repo } else { "$repo;$env:PYTHONPATH" }
$gitCommit = (& git -C $repo rev-parse HEAD 2>$null); if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitCommit)) { throw 'Unable to determine git commit' }
if (-not (Test-Path -LiteralPath $OutputRoot -PathType Container)) {
    if (Test-Path -LiteralPath $OutputRoot) { throw "OutputRoot is not a directory: $OutputRoot" }
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
}
$stagingRoot = ([IO.DirectoryInfo]::new((Resolve-Path -LiteralPath $OutputRoot -ErrorAction Stop).ProviderPath)).FullName
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
    Push-Location -LiteralPath $repo
    try {
        & $windeploy '--release' '--compiler-runtime' '--no-translations' (Join-Path "qt_app\build-portable-$Version\release" 'WorkpieceOrientation.exe')
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) { throw "windeployqt failed with exit code $LASTEXITCODE" }
}
finally {
    $env:PATH = $previousPath
}
$asciiInputRoot = Join-Path ([IO.Path]::GetPathRoot($repo)) ".portable-release-inputs-$PID"
if (Test-Path -LiteralPath $asciiInputRoot) { throw "Temporary input directory already exists: $asciiInputRoot" }
New-Item -ItemType Directory -Path $asciiInputRoot -Force | Out-Null
$releaseLocationPushed = $false
try {
    # PowerShell 5.1 marshals non-ASCII native arguments through the active
    # code page. Copy these two source files to an ASCII-only path first.
    $asciiGuide = Join-Path $asciiInputRoot 'guide.txt'
    $asciiNotices = Join-Path $asciiInputRoot 'notices.txt'
    $deployDir = Join-Path $repo 'deploy'
    $noticesSource = Join-Path $deployDir 'THIRD_PARTY-NOTICES.txt'
    $guideCandidates = @(Get-ChildItem -LiteralPath $deployDir -File -Filter '*.txt' |
        Where-Object { $_.Name -ine 'THIRD_PARTY-NOTICES.txt' })
    if ($guideCandidates.Count -eq 0) { throw "Offline guide .txt file not found in $deployDir" }
    if ($guideCandidates.Count -gt 1) { throw "Expected one offline guide .txt file in $deployDir; found $($guideCandidates.Count)" }
    if (-not (Test-Path -LiteralPath $noticesSource -PathType Leaf)) { throw "Third-party notices file not found: $noticesSource" }
    Copy-Item -LiteralPath $guideCandidates[0].FullName -Destination $asciiGuide -Force
    Copy-Item -LiteralPath $noticesSource -Destination $asciiNotices -Force
    $qtReleaseArg = Join-Path "qt_app\build-portable-$Version" 'release'
    $stagingRootArg = $stagingRoot.Substring($repo.Length + 1)
    # Python 3 is launched through PowerShell's native call operator, which
    # uses CreateProcessW and preserves .NET Unicode arguments. All paths
    # owned by this repository are nevertheless passed relative to this
    # Push-Location; external user-supplied paths remain intentionally intact.
    Push-Location -LiteralPath $repo
    $releaseLocationPushed = $true
    foreach ($ed in $editions) {
        $target = Join-Path $stagingRoot "$ed-$Version"
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
        $py = if ($ed -eq 'gpu') { $GpuPython } else { $CpuPython }
        & (Join-Path $repo 'scripts\build_portable_backend.ps1') -Edition $ed -Python $py -ProjectRoot $repo -OutputRoot $stagingRoot
        if ($LASTEXITCODE -ne 0) { throw "Portable backend build failed for $ed with exit code $LASTEXITCODE" }
        $backend = Join-Path $stagingRoot "backend-$ed\orientation_backend"
        $backendArg = Join-Path $stagingRootArg "backend-$ed\orientation_backend"
        $model = if ($ModelPath) { $ModelPath } else { 'models\shitu_rec' }
        $configArg = if ($PaddleConfig) { $PaddleConfig } else { 'deploy\configs\inference_general.yaml' }
        if (-not (Test-Path -LiteralPath $configArg)) {
            $configArg = Join-Path 'third_party\PaddleClas\deploy\configs' 'inference_general.yaml'
        }
        if (-not (Test-Path -LiteralPath $configArg)) { throw "Paddle inference config not found: $configArg" }
        $stageArgs = @('stage', '--edition', $ed, '--version', $Version, '--qt-release-dir', $qtReleaseArg, '--backend-dir', $backendArg, '--model-dir', $model, '--output-root', $stagingRootArg, '--guide', $asciiGuide, '--notices', $asciiNotices, '--git-commit', $gitCommit, '--repository-root', '.', '--paddle-config', $configArg)
    if ($MsvcRuntimeDir) { $stageArgs += @('--msvc-runtime-dir', $MsvcRuntimeDir) }
    & $py -m release_tools.portable_package @stageArgs
    if ($LASTEXITCODE -ne 0) { throw "Portable package staging failed for $ed with exit code $LASTEXITCODE" }
    $package = Join-Path $target "WorkpieceOrientation-$($ed.ToUpper())"
    $paddleDist = if ($ed -eq 'gpu') { 'paddlepaddle-gpu' } else { 'paddlepaddle' }
    $pythonLicense = Join-Path (Split-Path $py -Parent) 'LICENSE_PYTHON.txt'
    if (-not (Test-Path -LiteralPath $pythonLicense)) { $pythonLicense = Join-Path (Split-Path $py -Parent) 'LICENSE.txt' }
    if (-not (Test-Path -LiteralPath $pythonLicense)) { throw "Python license not found beside interpreter: $pythonLicense" }
    $packageArg = Join-Path $stagingRootArg "$ed-$Version\WorkpieceOrientation-$($ed.ToUpper())"
    & $py -c "from release_tools.portable_package import collect_licenses; from pathlib import Path; collect_licenses(Path(r'$packageArg')/'third_party_licenses', distributions=('pyinstaller','$paddleDist','paddleclas','numpy','opencv-python'), python_license=Path(r'$pythonLicense'))"
    if ($LASTEXITCODE -ne 0) { throw "License collection failed for $ed with exit code $LASTEXITCODE" }
    if (-not $SkipSmoke) {
        $smokeReportArg = Join-Path (Join-Path $stagingRootArg 'reports') "$ed-smoke.json"
        & $py 'scripts\smoke_portable_package.py' --package-root $packageArg --dataset-root $SmokeDatasetRoot --front-template-count $SmokeFrontTemplateCount --back-template-count $SmokeBackTemplateCount --seed $SmokeSeed --report $smokeReportArg
        if ($LASTEXITCODE -ne 0) { throw "Portable smoke failed for $ed" }
    }
    $label = $ed.ToUpper()
    $archive = Join-Path $repo "release_artifacts\WorkpieceOrientation-$label-x64-$Version.zip"
    $archiveArg = Join-Path 'release_artifacts' "WorkpieceOrientation-$label-x64-$Version.zip"
    & $py -c "from release_tools.portable_package import audit_package,write_manifest,zip_package,write_sha256; from pathlib import Path; p=Path(r'$packageArg'); roots=[Path.cwd().resolve(),Path(r'$QtBin'),Path(r'$py').parent,Path.home()]; audit_package(p,edition='$ed',version='$Version',forbidden_roots=roots,runtime_roots=roots); write_manifest(p,edition='$ed',version='$Version'); audit_package(p,edition='$ed',version='$Version',forbidden_roots=roots,runtime_roots=roots); a=zip_package(p,Path(r'$archiveArg')); write_sha256(a)"
    if ($LASTEXITCODE -ne 0) { throw "Package audit/archive failed for $ed with exit code $LASTEXITCODE" }
    # Re-open the exact ZIP artifact in a fresh, disposable directory and run
    # the same audit against the extracted layout. This catches archive root,
    # path, and omission errors that a source-directory audit cannot detect.
    & $py -c "from release_tools.portable_package import audit_zip_archive; from pathlib import Path; roots=[Path.cwd().resolve(),Path(r'$QtBin'),Path(r'$py').parent,Path.home()]; audit_zip_archive(Path(r'$archiveArg'),edition='$ed',version='$Version',forbidden_roots=roots,runtime_roots=roots)"
    if ($LASTEXITCODE -ne 0) { throw "Extracted ZIP audit failed for $ed with exit code $LASTEXITCODE" }
}
}
finally {
    try {
        if ($releaseLocationPushed) { Pop-Location }
    } finally {
        if (Test-Path -LiteralPath $asciiInputRoot) { Remove-Item -LiteralPath $asciiInputRoot -Recurse -Force }
    }
}
