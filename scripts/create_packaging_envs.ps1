param(
    [string]$Conda = 'conda',
    [string]$PythonVersion = '3.10.20',
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'
# Some developer shells set PIP_NO_INDEX=1 globally.  Packaging must be able
# to reach the explicitly selected indexes, so child pip/conda processes inherit
# an explicit empty value.
$env:PIP_NO_INDEX = ''
$common = Join-Path $ProjectRoot 'deploy\requirements\common.txt'
if (-not (Test-Path -LiteralPath $common)) { throw "Missing requirements file: $common" }

function Ensure-Environment([string]$Name) {
    $versionOutput = (& $Conda run -n $Name python --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) {
        $versionLine = $versionOutput -split '\r?\n' | Where-Object { $_ -match '^Python ' } | Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($versionLine) -or $versionLine -notmatch '^Python 3\.10\.') {
            throw "Packaging environment $Name has unsupported Python version: $versionOutput (Python 3.10.x required)"
        }
        return
    }
    if ($LASTEXITCODE -ne 0) {
        & $Conda create -y -n $Name ("python=" + $PythonVersion)
        if ($LASTEXITCODE -ne 0) { throw "Unable to create conda environment $Name" }
    }
}

Ensure-Environment 'workpiece-package-gpu'
Ensure-Environment 'workpiece-package-cpu'
& $Conda run -n workpiece-package-gpu python -m pip install -r $common
if ($LASTEXITCODE -ne 0) { throw 'GPU common dependency installation failed' }
& $Conda run -n workpiece-package-gpu python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
if ($LASTEXITCODE -ne 0) { throw 'GPU Paddle installation failed' }
& $Conda run -n workpiece-package-cpu python -m pip install -r $common
if ($LASTEXITCODE -ne 0) { throw 'CPU common dependency installation failed' }
& $Conda run -n workpiece-package-cpu python -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
if ($LASTEXITCODE -ne 0) { throw 'CPU Paddle installation failed' }

foreach ($edition in @('gpu', 'cpu')) {
    $env:PYTHONPATH = $ProjectRoot
    & $Conda run -n ("workpiece-package-" + $edition) python -c "import sys,struct; from release_tools.backend_bundle import edition_for,validate_installed_distributions; validate_installed_distributions(edition_for('$edition')); print('validated $edition',sys.version,struct.calcsize('P')*8)"
    if ($LASTEXITCODE -ne 0) { throw "Packaging environment validation failed: $edition" }
}
