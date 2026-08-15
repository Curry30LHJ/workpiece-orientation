param(
    [string]$PythonExecutable = 'E:\python\anaconda3\envs\shitu\python.exe',
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$ModelDir = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer'),
    [string]$LibraryDir = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'runtime_library'),
    [int]$Port = 37651
)

$serviceArguments = @(
    '-m', 'src.orientation_tcp_service',
    '--host', '127.0.0.1',
    '--port', $Port,
    '--project-root', $ProjectRoot,
    '--model-dir', $ModelDir,
    '--library-dir', $LibraryDir
)
$serviceProcess = Start-Process -WindowStyle Hidden -PassThru -FilePath $PythonExecutable `
    -ArgumentList $serviceArguments -WorkingDirectory $ProjectRoot

$probe = @'
import json
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.monotonic() + 120
sock = None
while time.monotonic() < deadline:
    try:
        sock = socket.create_connection((host, port), timeout=2)
        break
    except OSError:
        time.sleep(0.25)
if sock is None:
    raise SystemExit("orientation service did not become ready within 120 seconds")
with sock:
    stream = sock.makefile("rwb")
    for request_id, command in (("smoke-hello", "hello"), ("smoke-list", "list_workpieces"), ("smoke-shutdown", "shutdown")):
        request = {"version": 1, "request_id": request_id, "command": command}
        stream.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        stream.flush()
        response = json.loads(stream.readline().decode("utf-8"))
        if response.get("request_id") != request_id or response.get("ok") is not True:
            raise SystemExit(f"smoke request failed: {response}")
print("orientation service hello/list/shutdown passed")
'@
$probePath = Join-Path $env:TEMP ("workpiece-orientation-smoke-" + [guid]::NewGuid().ToString() + ".py")
Set-Content -LiteralPath $probePath -Value $probe -Encoding UTF8

try {
    & $PythonExecutable $probePath '127.0.0.1' $Port
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
    if ($null -ne $serviceProcess -and -not $serviceProcess.HasExited) {
        Stop-Process -Id $serviceProcess.Id -ErrorAction SilentlyContinue
    }
}
