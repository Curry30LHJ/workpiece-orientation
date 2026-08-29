param(
    [string[]]$Targets = @(
        'test_appconfig', 'test_backendclient', 'test_backendprocessmanager',
        'test_annotationmanager', 'test_appfoundation', 'test_inspectionpage',
        'test_workpiecelibrarypage', 'test_geometryrulecanvas',
        'test_geometryrulespage', 'test_mainwindow',
        'test_startupsmokecontroller'
    ),
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$QtBin = 'E:\QT\5.14\5.14.2\msvc2017_64\bin',
    [string]$Jom = 'E:\QT\5.14\Tools\QtCreator\bin\jom.exe',
    [string]$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat'
)

$qmake = Join-Path $QtBin 'qmake.exe'
foreach ($required in @($qmake, $Jom, $VcVars)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required Qt test path does not exist: $required"
    }
}

foreach ($target in $Targets) {
    $project = Join-Path $ProjectRoot "qt_app\tests\$target.pro"
    $buildDir = Join-Path $ProjectRoot "qt_app\tests\build-$target"
    if (-not (Test-Path -LiteralPath $project)) {
        throw "Qt test project does not exist: $project"
    }
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    Push-Location -LiteralPath $buildDir
    try {
        $configure = ('call "' + $VcVars + '" && "' + $qmake + '" "' + $project + '" CONFIG+=release')
        cmd.exe /d /s /c $configure
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        $build = 'call "' + $VcVars + '" && "' + $Jom + '"'
        cmd.exe /d /s /c $build
        if ($LASTEXITCODE -ne 0) {
            $fallback = ('call "' + $VcVars + '" && nmake /f Makefile.Release /NOLOGO')
            cmd.exe /d /s /c $fallback
        }
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        $executable = Join-Path $buildDir "release\$target.exe"
        if (-not (Test-Path -LiteralPath $executable)) {
            throw "Qt test executable does not exist: $executable"
        }
        $previousPlatform = $env:QT_QPA_PLATFORM
        try {
            $env:QT_QPA_PLATFORM = 'offscreen'
            & $executable '-platform' 'offscreen' '-o' '-,txt'
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        } finally {
            $env:QT_QPA_PLATFORM = $previousPlatform
        }
    } finally {
        Pop-Location
    }
}
