param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$QtBin = 'E:\QT\5.14\5.14.2\msvc2017_64\bin',
    [string]$Jom = 'E:\QT\5.14\Tools\QtCreator\bin\jom.exe',
    [string]$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat',
    [string]$BuildDir = (Join-Path $ProjectRoot 'qt_app\build-release'),
    [switch]$SkipRuntimeConfig
)

$qmake = Join-Path $QtBin 'qmake.exe'
$projectFile = Join-Path $ProjectRoot 'qt_app\workpiece_orientation.pro'
$runtimeConfig = Join-Path $ProjectRoot 'qt_app\app_config.json'

foreach ($requiredPath in @($qmake, $Jom, $VcVars, $projectFile)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required Qt build path does not exist: $requiredPath"
    }
}

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Set-Location -LiteralPath $buildDir
$qmakeLine = 'call "' + $VcVars + '" && "' + $qmake + '" "..\workpiece_orientation.pro" CONFIG+=release'
cmd.exe /d /s /c $qmakeLine
$buildExitCode = $LASTEXITCODE
if ($buildExitCode -ne 0) {
    exit $buildExitCode
}

$makeLine = 'call "' + $VcVars + '" && "' + $Jom + '"'
cmd.exe /d /s /c $makeLine
$buildExitCode = $LASTEXITCODE
if ($buildExitCode -ne 0) {
    Write-Warning ("jom failed with exit code $buildExitCode; retrying with Visual Studio nmake")
    $fallbackLine = 'call "' + $VcVars + '" && nmake /f Makefile.Release /NOLOGO'
    cmd.exe /d /s /c $fallbackLine
    $buildExitCode = $LASTEXITCODE
}
if ($buildExitCode -ne 0) {
    exit $buildExitCode
}
$releaseDir = Join-Path $buildDir 'release'
$releaseConfig = Join-Path $releaseDir 'app_config.json'
if (-not $SkipRuntimeConfig -and (Test-Path -LiteralPath $runtimeConfig)) {
    Copy-Item -LiteralPath $runtimeConfig -Destination $releaseConfig -Force
    Write-Output ("Runtime config copied: " + $releaseConfig)
} elseif (-not $SkipRuntimeConfig) {
    Write-Warning ("Runtime config not found; create it from qt_app\app_config.json.example: " + $runtimeConfig)
}
Write-Output ("Qt build succeeded: " + (Join-Path $releaseDir 'WorkpieceOrientation.exe'))
