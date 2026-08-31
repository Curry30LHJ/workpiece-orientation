[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PaddleInferenceRoot,

    [Parameter(Mandatory = $true)]
    [string]$OpenCvRoot,

    [Parameter(Mandatory = $true)]
    [string]$BuildDir,

    [ValidateSet("Release", "Debug", "RelWithDebInfo", "MinSizeRel")]
    [string]$Configuration = "Release",

    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ParameterName,

        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$ParameterName does not exist or is not a directory: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Assert-RequiredFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,

        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description is missing: $Path"
    }
}

$resolvedPaddleRoot = Resolve-RequiredDirectory `
    -ParameterName "PaddleInferenceRoot" `
    -Path $PaddleInferenceRoot
$resolvedOpenCvRoot = Resolve-RequiredDirectory `
    -ParameterName "OpenCvRoot" `
    -Path $OpenCvRoot

$paddleHeader = Join-Path $resolvedPaddleRoot "paddle\include\paddle_inference_api.h"
$paddleLibrary = Join-Path $resolvedPaddleRoot "paddle\lib\paddle_inference.lib"
$paddleRuntime = Join-Path $resolvedPaddleRoot "paddle\lib\paddle_inference.dll"
Assert-RequiredFile -Description "Paddle Inference header" -Path $paddleHeader
Assert-RequiredFile -Description "Paddle Inference import library" -Path $paddleLibrary
Assert-RequiredFile -Description "Paddle Inference runtime" -Path $paddleRuntime

$openCvConfigCandidates = @(
    (Join-Path $resolvedOpenCvRoot "OpenCVConfig.cmake"),
    (Join-Path $resolvedOpenCvRoot "build\OpenCVConfig.cmake")
)
$openCvConfig = $openCvConfigCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $openCvConfig) {
    throw "OpenCvRoot does not contain OpenCVConfig.cmake: $resolvedOpenCvRoot"
}

$sourceDir = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot "..\native\ppshitu_rec_benchmark"
)).Path
$resolvedBuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$buildRoot = [System.IO.Path]::GetPathRoot($resolvedBuildDir)
if ($resolvedBuildDir -eq $buildRoot) {
    throw "BuildDir must not be a drive root: $resolvedBuildDir"
}

if ($Clean -and (Test-Path -LiteralPath $resolvedBuildDir)) {
    $existingBuildDir = (Resolve-Path -LiteralPath $resolvedBuildDir).Path
    if ($existingBuildDir -ne $resolvedBuildDir) {
        throw "Refusing to clean a BuildDir that resolves elsewhere: $existingBuildDir"
    }
    Remove-Item -LiteralPath $existingBuildDir -Recurse -Force
}

& cmake `
    -S $sourceDir `
    -B $resolvedBuildDir `
    -G "Visual Studio 16 2019" `
    -A x64 `
    "-DPADDLE_INFERENCE_ROOT=$resolvedPaddleRoot" `
    "-DOPENCV_ROOT=$resolvedOpenCvRoot"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE"
}

& cmake --build $resolvedBuildDir --config $Configuration
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed with exit code $LASTEXITCODE"
}

$outputDir = Join-Path $resolvedBuildDir $Configuration
$executable = Join-Path $outputDir "ppshitu_rec_benchmark.exe"
Assert-RequiredFile -Description "Native benchmark executable" -Path $executable

$runtimeFiles = @($paddleRuntime)
$runtimeFiles += Get-ChildItem -LiteralPath $resolvedPaddleRoot -Recurse -File -Filter "*.dll" |
    Select-Object -ExpandProperty FullName
$openCvRuntimeCandidates = @(
    (Join-Path $resolvedOpenCvRoot "build\x64\vc15\bin"),
    (Join-Path $resolvedOpenCvRoot "x64\vc15\bin")
)
$openCvRuntimeDir = $openCvRuntimeCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
    Select-Object -First 1
if (-not $openCvRuntimeDir) {
    throw "OpenCvRoot does not contain a Visual Studio 2019 runtime directory: $resolvedOpenCvRoot"
}
$runtimeFiles += Get-ChildItem -LiteralPath $openCvRuntimeDir -File -Filter "opencv_world*.dll" |
    Select-Object -ExpandProperty FullName

$seenNames = @{}
foreach ($runtimeFile in ($runtimeFiles | Sort-Object -Unique)) {
    $name = Split-Path -Leaf $runtimeFile
    if ($seenNames.ContainsKey($name)) {
        $existingHash = (Get-FileHash -LiteralPath $seenNames[$name] -Algorithm SHA256).Hash
        $candidateHash = (Get-FileHash -LiteralPath $runtimeFile -Algorithm SHA256).Hash
        if ($existingHash -ne $candidateHash) {
            throw "Conflicting runtime DLLs share the name $name"
        }
        continue
    }
    $seenNames[$name] = $runtimeFile
    Copy-Item -LiteralPath $runtimeFile -Destination (Join-Path $outputDir $name) -Force
}

Write-Output $executable
