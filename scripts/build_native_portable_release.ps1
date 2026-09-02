[CmdletBinding()]
param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$Version = '1.1.2',
    [string]$CpuPython = 'E:\python\anaconda3\envs\workpiece-package-cpu\python.exe',
    [string]$QtBin = 'E:\QT\5.14\5.14.2\msvc2017_64\bin',
    [string]$NativeRuntimeDir = '',
    [string]$ModelPath = '',
    [string]$PaddleConfig = '',
    [string]$MsvcRuntimeDir = '',
    [switch]$SkipSmoke,
    [string]$SmokeDatasetRoot = '',
    [int]$SmokeFrontTemplateCount = 5,
    [int]$SmokeBackTemplateCount = 10,
    [int]$SmokeSeed = 20260813,
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repo 'release_staging'
}
if ([string]::IsNullOrWhiteSpace($NativeRuntimeDir)) {
    $NativeRuntimeDir = Join-Path $repo '.native-build\ppshitu-cpp-task4\Release'
}
$requiredNativeFiles = @(
    'ppshitu_rec_service.exe', 'paddle_inference.dll', 'opencv_world460.dll',
    'mkldnn.dll', 'mklml.dll', 'common.dll', 'libiomp5md.dll'
)
foreach ($name in $requiredNativeFiles) {
    $path = Join-Path $NativeRuntimeDir $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "NativeRuntimeDir is missing ${name}: $path"
    }
}
if (-not (Test-Path -LiteralPath $CpuPython -PathType Leaf)) {
    throw "CPU packaging Python does not exist: $CpuPython"
}
$windeploy = Join-Path $QtBin 'windeployqt.exe'
if (-not (Test-Path -LiteralPath $windeploy -PathType Leaf)) {
    throw "windeployqt not found: $windeploy"
}
if (-not $SkipSmoke -and [string]::IsNullOrWhiteSpace($SmokeDatasetRoot)) {
    throw 'SmokeDatasetRoot is required unless -SkipSmoke is explicitly specified.'
}
if (-not [string]::IsNullOrWhiteSpace($SmokeDatasetRoot) -and
    -not (Test-Path -LiteralPath $SmokeDatasetRoot -PathType Container)) {
    throw "SmokeDatasetRoot is not a directory: $SmokeDatasetRoot"
}

# All generated files live below the named release_staging root.  This keeps
# cleanup recoverable and prevents an accidental parameter from targeting the
# repository or a user data directory.
if (-not (Test-Path -LiteralPath $OutputRoot -PathType Container)) {
    if (Test-Path -LiteralPath $OutputRoot) { throw "OutputRoot is not a directory: $OutputRoot" }
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
}
$stagingRoot = ([IO.DirectoryInfo]::new((Resolve-Path -LiteralPath $OutputRoot).ProviderPath)).FullName
$releaseRoot = ([IO.Path]::GetFullPath((Join-Path $repo 'release_staging'))).TrimEnd('\')
$stagingFull = ([IO.Path]::GetFullPath($stagingRoot)).TrimEnd('\')
$releasePrefix = $releaseRoot + '\'
if ($stagingFull -ne $releaseRoot -and -not $stagingFull.StartsWith($releasePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputRoot must remain under release_staging'
}
$workRoot = Join-Path $stagingRoot ("native-cpp-$Version")
$backendBuildRoot = Join-Path $workRoot 'backend-build'
$nativeInputRoot = Join-Path $workRoot 'native-runtime-input'
$qtBuild = Join-Path $repo ("qt_app\build-portable-native-$Version")
$package = Join-Path $workRoot ("cpu-$Version\WorkpieceOrientation-CPU")
$archive = Join-Path $repo ("release_artifacts\WorkpieceOrientation-CPU-NativeCPP-x64-$Version.zip")

foreach ($ownedPath in @($workRoot, $qtBuild)) {
    if (Test-Path -LiteralPath $ownedPath) {
        $resolved = ([IO.Path]::GetFullPath($ownedPath)).TrimEnd('\')
        $base = if ($ownedPath -eq $qtBuild) {
            ([IO.Path]::GetFullPath((Join-Path $repo 'qt_app'))).TrimEnd('\') + '\'
        } else {
            $stagingFull.TrimEnd('\') + '\'
        }
        if (-not $resolved.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clear path outside the owned build root: $resolved"
        }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $workRoot, $nativeInputRoot | Out-Null

# Stage only the release native files into an ASCII-friendly input directory;
# this also prevents debug DLLs from being picked up accidentally.  The
# staging helper adds the coherent VC runtime and optional OpenMP/debug-helper
# DLLs when available.
foreach ($name in $requiredNativeFiles) {
    Copy-Item -LiteralPath (Join-Path $NativeRuntimeDir $name) -Destination (Join-Path $nativeInputRoot $name) -Force
}
foreach ($name in @('concrt140.dll', 'vcomp140.dll', 'dbghelp.dll')) {
    $source = Get-ChildItem -LiteralPath $NativeRuntimeDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ieq $name } | Select-Object -First 1
    if (-not $source) {
        $source = Get-ChildItem -LiteralPath (Join-Path $env:SystemRoot 'System32') -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ieq $name } | Select-Object -First 1
    }
    if ($source) {
        Copy-Item -LiteralPath $source.FullName -Destination (Join-Path $nativeInputRoot $name) -Force
    }
}

$previousPath = $env:PATH
$asciiInputRoot = Join-Path ([IO.Path]::GetPathRoot($repo)) ".portable-native-inputs-$PID"
if (Test-Path -LiteralPath $asciiInputRoot) { throw "Temporary input directory already exists: $asciiInputRoot" }
New-Item -ItemType Directory -Force -Path $asciiInputRoot | Out-Null
$locationPushed = $false
try {
    $deployDir = Join-Path $repo 'deploy'
    $guideCandidates = @(Get-ChildItem -LiteralPath $deployDir -File -Filter '*.txt' |
        Where-Object { $_.Name -ine 'THIRD_PARTY-NOTICES.txt' })
    if ($guideCandidates.Count -ne 1) {
        throw "Expected exactly one offline guide .txt file in $deployDir"
    }
    $noticesSource = Join-Path $deployDir 'THIRD_PARTY-NOTICES.txt'
    if (-not (Test-Path -LiteralPath $noticesSource -PathType Leaf)) {
        throw "Third-party notices file not found: $noticesSource"
    }
    $asciiGuide = Join-Path $asciiInputRoot 'guide.txt'
    $asciiNotices = Join-Path $asciiInputRoot 'notices.txt'
    Copy-Item -LiteralPath $guideCandidates[0].FullName -Destination $asciiGuide -Force
    Copy-Item -LiteralPath $noticesSource -Destination $asciiNotices -Force

    $env:PATH = "$([IO.Path]::GetFullPath($QtBin));$previousPath"
    & (Join-Path $repo 'scripts\build_qt5.ps1') -ProjectRoot $repo -QtBin $QtBin -BuildDir $qtBuild -SkipRuntimeConfig
    if ($LASTEXITCODE -ne 0) { throw "Qt build failed with exit code $LASTEXITCODE" }
    $qtRelease = Join-Path $qtBuild 'release'
    $qtExecutable = Join-Path $qtRelease 'WorkpieceOrientation.exe'
    if (-not (Test-Path -LiteralPath $qtExecutable -PathType Leaf)) { throw "Qt executable not found: $qtExecutable" }
    Push-Location -LiteralPath $repo
    $locationPushed = $true
    & $windeploy '--release' '--compiler-runtime' '--no-translations' $qtExecutable
    if ($LASTEXITCODE -ne 0) { throw "windeployqt failed with exit code $LASTEXITCODE" }

    & (Join-Path $repo 'scripts\build_portable_backend.ps1') -Edition cpu -Python $CpuPython -ProjectRoot $repo -OutputRoot $backendBuildRoot
    if ($LASTEXITCODE -ne 0) { throw "Frozen CPU backend build failed with exit code $LASTEXITCODE" }
    $backend = Join-Path $backendBuildRoot 'backend-cpu\orientation_backend'
    if (-not (Test-Path -LiteralPath (Join-Path $backend 'orientation_backend.exe') -PathType Leaf)) {
        throw "Frozen CPU backend executable not found: $backend"
    }

    $modelArg = if ($ModelPath) { $ModelPath } else { 'third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer' }
    $configArg = if ($PaddleConfig) { $PaddleConfig } else { 'third_party\PaddleClas\deploy\configs\inference_general.yaml' }
    if (-not (Test-Path -LiteralPath $modelArg -PathType Container)) { throw "Model directory not found: $modelArg" }
    if (-not (Test-Path -LiteralPath $configArg -PathType Leaf)) { throw "Paddle inference config not found: $configArg" }
    $gitCommit = (& git -C $repo rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitCommit)) { throw 'Unable to determine git commit' }

    $stageArgs = @(
        '-m', 'release_tools.portable_package', 'stage',
        '--edition', 'cpu', '--version', $Version,
        '--qt-release-dir', $qtRelease,
        '--backend-dir', $backend,
        '--model-dir', $modelArg,
        '--output-root', $workRoot,
        '--guide', $asciiGuide, '--notices', $asciiNotices,
        '--git-commit', $gitCommit, '--repository-root', $repo,
        '--paddle-config', $configArg,
        '--pp-backend', 'native_cpp', '--native-cpp-runtime-dir', $nativeInputRoot
    )
    if ($MsvcRuntimeDir) { $stageArgs += @('--msvc-runtime-dir', $MsvcRuntimeDir) }
    & $CpuPython @stageArgs
    if ($LASTEXITCODE -ne 0) { throw "Native package staging failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $package -PathType Container)) { throw "Native package root not found: $package" }

    $pythonLicense = Join-Path (Split-Path $CpuPython -Parent) 'LICENSE_PYTHON.txt'
    if (-not (Test-Path -LiteralPath $pythonLicense -PathType Leaf)) {
        $pythonLicense = Join-Path (Split-Path $CpuPython -Parent) 'LICENSE.txt'
    }
    if (-not (Test-Path -LiteralPath $pythonLicense -PathType Leaf)) { throw "Python license not found: $pythonLicense" }
    & $CpuPython -c "from release_tools.portable_package import collect_licenses; from pathlib import Path; collect_licenses(Path(r'$package')/'third_party_licenses', distributions=('pyinstaller','paddlepaddle','paddleclas','numpy','opencv-python'), python_license=Path(r'$pythonLicense'))"
    if ($LASTEXITCODE -ne 0) { throw 'Native package license collection failed' }

    $forbidden = @($repo, $QtBin, (Split-Path $CpuPython -Parent), $NativeRuntimeDir, $nativeInputRoot)
    $auditCode = "from release_tools.portable_package import audit_package,write_manifest,zip_package,write_sha256; from pathlib import Path; p=Path(r'$package'); roots=[Path(r'$repo'),Path(r'$QtBin'),Path(r'$(Split-Path $CpuPython -Parent)'),Path(r'$NativeRuntimeDir'),Path(r'$nativeInputRoot')]; audit_package(p,edition='cpu',version='$Version',forbidden_roots=roots,runtime_roots=roots); write_manifest(p,edition='cpu',version='$Version'); audit_package(p,edition='cpu',version='$Version',forbidden_roots=roots,runtime_roots=roots); a=zip_package(p,Path(r'$archive')); write_sha256(a)"
    & $CpuPython -c $auditCode
    if ($LASTEXITCODE -ne 0) { throw 'Native package audit/archive failed' }
    $zipAuditCode = "from release_tools.portable_package import audit_zip_archive; from pathlib import Path; roots=[Path(r'$repo'),Path(r'$QtBin'),Path(r'$(Split-Path $CpuPython -Parent)'),Path(r'$NativeRuntimeDir')]; audit_zip_archive(Path(r'$archive'),edition='cpu',version='$Version',forbidden_roots=roots,runtime_roots=roots)"
    & $CpuPython -c $zipAuditCode
    if ($LASTEXITCODE -ne 0) { throw 'Native ZIP audit failed' }

    if (-not $SkipSmoke) {
        $report = Join-Path $workRoot 'reports\cpu-native-cpp-smoke.json'
        & $CpuPython (Join-Path $repo 'scripts\smoke_portable_package.py') --package-root $package --dataset-root ([IO.Path]::GetFullPath($SmokeDatasetRoot)) --front-template-count $SmokeFrontTemplateCount --back-template-count $SmokeBackTemplateCount --seed $SmokeSeed --report $report
        if ($LASTEXITCODE -ne 0) { throw 'Native portable smoke test failed' }
    }
    Write-Output ("Native C++ portable package: " + $archive)
    Write-Output ("SHA-256 sidecar: " + ($archive + '.sha256'))
}
finally {
    if ($locationPushed) { Pop-Location }
    $env:PATH = $previousPath
    if (Test-Path -LiteralPath $asciiInputRoot) { Remove-Item -LiteralPath $asciiInputRoot -Recurse -Force }
}
