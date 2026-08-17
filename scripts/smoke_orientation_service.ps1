param(
    [string]$PythonExecutable = 'E:\python\anaconda3\envs\shitu\python.exe',
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$ModelDir = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer'),
    [string]$LibraryDir = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'runtime_library'),
    [int]$Port = 37651,
    [int]$StartupTimeoutSeconds = 600,
    [string]$GeometryWorkpieceId = '',
    [string]$GeometryPredictImage = ''
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
startup_timeout_seconds = int(sys.argv[3])
geometry_workpiece_id = "" if sys.argv[4] == "-" else sys.argv[4]
geometry_predict_image = "" if sys.argv[5] == "-" else sys.argv[5]
deadline = time.monotonic() + startup_timeout_seconds
sock = None
while time.monotonic() < deadline:
    try:
        sock = socket.create_connection((host, port), timeout=2)
        break
    except OSError:
        time.sleep(0.25)
if sock is None:
    raise SystemExit(f"orientation service did not become ready within {startup_timeout_seconds} seconds")
sock.settimeout(120)
with sock:
    receive_buffer = bytearray()
    def request(request_id, command, fields=None):
        payload = {"version": 1, "request_id": request_id, "command": command}
        if fields:
            payload.update(fields)
        sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        while b"\n" not in receive_buffer:
            chunk = sock.recv(65536)
            if not chunk:
                raise SystemExit("orientation service closed the probe connection")
            receive_buffer.extend(chunk)
        line, remainder = bytes(receive_buffer).split(b"\n", 1)
        receive_buffer[:] = remainder
        response = json.loads(line.decode("utf-8"))
        if response.get("request_id") != request_id:
            raise SystemExit(f"smoke request id mismatch: {response}")
        return response

    attempt = 0
    while True:
        request_id = f"smoke-hello-{attempt}"
        response = request(request_id, "hello")
        if response.get("ok") is True and response.get("ready") is True:
            print("orientation service is ready", flush=True)
            break
        if response.get("ok") is True and response.get("status") == "loading":
            if attempt == 0:
                print("orientation service is loading models...", flush=True)
            attempt += 1
            time.sleep(0.25)
            continue
        raise SystemExit(f"smoke hello failed: {response}")

    list_response = request("smoke-list", "list_workpieces")
    if list_response.get("ok") is not True:
        raise SystemExit(f"smoke request failed: {list_response}")
    print("smoke list_workpieces passed", flush=True)

    if geometry_workpiece_id:
        known_ids = {item.get("id") for item in list_response.get("workpieces", [])}
        if geometry_workpiece_id not in known_ids:
            raise SystemExit(f"geometry workpiece is not present in list_workpieces: {geometry_workpiece_id}")

        profile_response = request(
            "smoke-geometry-get",
            "get_geometry_mask_profile",
            {"workpiece_id": geometry_workpiece_id},
        )
        if profile_response.get("ok") is not True:
            raise SystemExit(f"get_geometry_mask_profile failed: {profile_response}")
        profile = profile_response["profile"]
        library_revision = int(profile["library_revision"])
        draft_revision = int(profile["draft_revision"])
        draft = profile["draft"]

        saved_response = request(
            "smoke-geometry-save",
            "save_geometry_mask_draft",
            {
                "workpiece_id": geometry_workpiece_id,
                "base_library_revision": library_revision,
                "base_draft_revision": draft_revision,
                "operation_id": "smoke-geometry-save-op",
                "draft": draft,
            },
        )
        if saved_response.get("ok") is not True:
            raise SystemExit(f"save_geometry_mask_draft failed: {saved_response}")
        draft_revision = int(saved_response["profile"]["draft_revision"])

        validation_response = request(
            "smoke-geometry-validate",
            "validate_geometry_mask_draft",
            {
                "workpiece_id": geometry_workpiece_id,
                "base_library_revision": library_revision,
                "base_draft_revision": draft_revision,
                "operation_id": "smoke-geometry-validate-op",
            },
        )
        if validation_response.get("ok") is not True:
            raise SystemExit(f"validate_geometry_mask_draft failed: {validation_response}")
        job = validation_response["job"]
        print(f"geometry validation job started: {job.get('job_id')}", flush=True)
        poll_index = 0
        while job.get("state") in {"queued", "running"}:
            time.sleep(0.25)
            poll_index += 1
            job_response = request(
                f"smoke-geometry-job-{poll_index}",
                "get_geometry_mask_validation_job",
                {"job_id": job["job_id"]},
            )
            if job_response.get("ok") is not True:
                raise SystemExit(f"get_geometry_mask_validation_job failed: {job_response}")
            job = job_response["job"]
        if job.get("state") != "completed":
            raise SystemExit(f"geometry validation did not complete: {job}")
        print("geometry validation completed", flush=True)

        publish_fields = {
            "workpiece_id": geometry_workpiece_id,
            "job_id": job["job_id"],
            "base_library_revision": library_revision,
            "base_draft_revision": draft_revision,
            "operation_id": "smoke-geometry-publish-op",
            "override_reason": "",
        }
        publish_response = request("smoke-geometry-publish", "publish_geometry_mask_profile", publish_fields)
        if not publish_response.get("ok") and publish_response.get("error", {}).get("code") == "GEOMETRY_OVERRIDE_REQUIRED":
            publish_fields["override_reason"] = "smoke validation override"
            publish_response = request("smoke-geometry-publish-override", "publish_geometry_mask_profile", publish_fields)
        if publish_response.get("ok") is not True:
            raise SystemExit(f"publish_geometry_mask_profile failed: {publish_response}")
        published_profile = publish_response["profile"]
        print(f"geometry profile published: revision {published_profile.get('active_revision')}", flush=True)

        if geometry_predict_image:
            predict_response = request(
                "smoke-geometry-predict",
                "predict",
                {"workpiece_id": geometry_workpiece_id, "image_path": geometry_predict_image},
            )
            if predict_response.get("ok") is not True:
                raise SystemExit(f"predict failed: {predict_response}")
            geometry_status = predict_response.get("geometry_mask", {}).get("status", "not_configured")
            print(f"geometry predict passed: status {geometry_status}", flush=True)

        rollback_response = request(
            "smoke-geometry-rollback",
            "rollback_geometry_mask_profile",
            {
                "workpiece_id": geometry_workpiece_id,
                "base_library_revision": int(published_profile["library_revision"]),
                "operation_id": "smoke-geometry-rollback-op",
            },
        )
        if rollback_response.get("ok") is not True:
            raise SystemExit(f"rollback_geometry_mask_profile failed: {rollback_response}")
        print("geometry profile rollback passed", flush=True)
    else:
        print("geometry lifecycle skipped; pass -GeometryWorkpieceId to exercise it", flush=True)

    shutdown_response = request("smoke-shutdown", "shutdown")
    if shutdown_response.get("ok") is not True:
        raise SystemExit(f"smoke shutdown failed: {shutdown_response}")
    print("smoke shutdown passed", flush=True)
print("orientation service hello/list/shutdown passed")
'@
$probePath = Join-Path $env:TEMP ("workpiece-orientation-smoke-" + [guid]::NewGuid().ToString() + ".py")
Set-Content -LiteralPath $probePath -Value $probe -Encoding UTF8

try {
    $geometryIdArgument = if ([string]::IsNullOrWhiteSpace($GeometryWorkpieceId)) { '-' } else { $GeometryWorkpieceId }
    $geometryPredictArgument = if ([string]::IsNullOrWhiteSpace($GeometryPredictImage)) { '-' } else { $GeometryPredictImage }
    & $PythonExecutable $probePath '127.0.0.1' $Port $StartupTimeoutSeconds $geometryIdArgument $geometryPredictArgument
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
