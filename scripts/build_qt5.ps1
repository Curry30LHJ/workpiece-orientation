param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$QtBin = 'E:\QT\5.14\5.14.2\msvc2017_64\bin',
    [string]$Jom = 'E:\QT\5.14\Tools\QtCreator\bin\jom.exe',
    [string]$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat'
)

$qmake = Join-Path $QtBin 'qmake.exe'
$projectFile = Join-Path $ProjectRoot 'qt_app\workpiece_orientation.pro'
$buildDir = Join-Path $ProjectRoot 'qt_app\build-release'

foreach ($requiredPath in @($qmake, $Jom, $VcVars, $projectFile)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required Qt build path does not exist: $requiredPath"
    }
}

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Set-Location -LiteralPath $buildDir
$line = 'call "' + $VcVars + '" && "' + $qmake + '" "' + $projectFile + '" CONFIG+=release && "' + $Jom + '"'
cmd.exe /d /s /c $line
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Output ("Qt build succeeded: " + (Join-Path $buildDir 'release\workpiece_orientation.exe'))
