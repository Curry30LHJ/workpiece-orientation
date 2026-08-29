"""Reusable, offline protocol smoke checks for packaged editions."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from src.shitu_baseline import split_labels


@dataclass(frozen=True)
class SmokeOptions:
    package_root: Path | None = None
    dataset_root: Path = Path(".")
    front_template_count: int = 5
    back_template_count: int = 10
    seed: int = 20260813
    report_path: Path | None = None
    source_package_root: Path | None = None
    destination_package_root: Path | None = None
    timeout_seconds: float = 600.0


class SmokeReport(dict):
    """JSON-friendly report retaining convenient attribute access."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _persist_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _process_exit_code(process: Any) -> int | None:
    value = getattr(process, "returncode", None)
    if value is None:
        try:
            value = process.poll()
        except Exception:
            value = None
    return value


def _collect_backend_log(report: dict[str, Any], process: Any) -> None:
    """Best-effort capture of backend output after process shutdown."""
    stream = getattr(process, "stdout", None)
    if stream is None:
        return
    try:
        report["backend_log"] = stream.read()
    except Exception as exc:
        report["backend_log_error"] = str(exc)


def _persist_report_safely(path: Path | None, report: dict[str, Any]) -> Exception | None:
    """Persist a report without replacing the operation's primary exception."""
    try:
        _persist_report(path, report)
    except Exception as exc:
        report["report_persist_error"] = str(exc)
        return exc
    return None


class _JsonSocket:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.file = sock.makefile("rwb")

    def request(self, command: str, **fields: Any) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        payload = {"version": 1, "request_id": request_id, "command": command, **fields}
        self.file.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        self.file.flush()
        while True:
            line = self.file.readline()
            if not line:
                raise RuntimeError("backend closed smoke connection")
            response = json.loads(line.decode("utf-8"))
            if response.get("event") == "progress":
                continue
            if response.get("request_id") != request_id:
                raise RuntimeError("backend request id mismatch")
            return response

    def close(self) -> None:
        self.file.close()
        self.sock.close()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _ok(response: dict[str, Any], command: str) -> dict[str, Any]:
    if response.get("ok") is not True:
        raise RuntimeError(f"{command} failed: {response}")
    return response


def _package_config(root: Path) -> dict[str, Any]:
    path = root / "app_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("backend_executable", "project_root", "model_dir", "paddle_config", "data_root"):
        value = config.get(key)
        if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"package config path must be relative: {key}")
    return config


def _copy_to_long_temp(root: Path, factory: Callable[..., Path] | None) -> Path:
    if factory is not None:
        target = Path(factory())
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        base = Path(tempfile.mkdtemp(prefix="离线 smoke portable package "))
        target = base / ("x" * max(1, 190 - len(str(base))))
    # A factory is allowed to choose the location, but an existing location is
    # never safe to reuse: doing so could overwrite another smoke run or the
    # caller's package data.
    if target.exists():
        raise FileExistsError(f"temporary package target already exists: {target}")
    shutil.copytree(root, target)
    if factory is None and len(str(target.resolve())) < 180:
        raise RuntimeError("temporary package path is shorter than portability requirement")
    return target


def _start_backend(root: Path, config: dict[str, Any], port: int,
                   process_factory: Callable[..., Any]) -> Any:
    token = str(uuid.uuid4())
    args = [str(root / config["backend_executable"]), "--host", "127.0.0.1", "--port", str(port),
            "--project-root", str(root / config["project_root"]), "--model-dir", str(root / config["model_dir"]),
            "--paddle-config", str(root / config["paddle_config"]), "--data-root", str(root / config["data_root"]),
            "--compute-device", str(config["compute_device"]), "--model-sha256", str(config["model_sha256"]),
            "--edition", str(config["edition"]), "--package-version", str(config["package_version"]),
            "--instance-token", token, "--parent-pid", str(os.getpid()),
            "--local-search-mode", str(config.get("local_search_mode", "adaptive")),
            "--inference-mode", str(config.get("inference_mode", "fast_geometry"))]
    try:
        return process_factory(args, cwd=str(root), token=token)
    except TypeError:
        return process_factory(args, str(root))


def _real_process_factory(args: list[str], *, cwd: str, token: str) -> subprocess.Popen:
    kwargs: dict[str, Any] = {"cwd": cwd, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                              "text": True}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return subprocess.Popen(args, **kwargs)


def run_smoke(options: SmokeOptions, *, process_factory: Callable[..., Any] | None = None,
              socket_factory: Callable[..., Any] | None = None,
              temp_root_factory: Callable[..., Path] | None = None,
              free_port_factory: Callable[[], int] | None = None,
              port_factory: Callable[[], int] | None = None) -> dict[str, Any]:
    """Run a packaged protocol smoke test. Factories make the workflow unit-testable."""
    if options.package_root is None:
        raise ValueError("package_root is required")
    package = Path(options.package_root).resolve()
    report: SmokeReport = SmokeReport(ok=False, package_root=str(package), commands=[], results=[])
    try:
        config = _package_config(package)
        templates, held_out = split_labels(Path(options.dataset_root), options.front_template_count, options.seed)
        front = [str(p) for p in templates["0"]]
        back = [str(p) for p in templates["1"]]
        template_hashes = {hashlib.sha256(p.read_bytes()).digest() for p in (*templates["0"], *templates["1"])}
        def held_out_query(label: str) -> Path:
            for candidate in held_out[label]:
                if hashlib.sha256(candidate.read_bytes()).digest() not in template_hashes:
                    return candidate
            raise ValueError(f"no held-out query distinct from templates for label {label}")
        front_query, back_query = str(held_out_query("0")), str(held_out_query("1"))
    except Exception as exc:
        report["error"] = str(exc)
        report["process_exit"] = None
        setattr(exc, "smoke_report", report)
        _persist_report_safely(options.report_path, report)
        raise
    process_factory = process_factory or _real_process_factory
    port = (free_port_factory or port_factory or _free_port)()
    try:
        temp_package = _copy_to_long_temp(package, temp_root_factory)
        config = _package_config(temp_package)
        process = _start_backend(temp_package, config, port, process_factory)
    except Exception as exc:
        report["error"] = str(exc)
        report["process_exit"] = None
        setattr(exc, "smoke_report", report)
        _persist_report_safely(options.report_path, report)
        raise
    client = None
    report["edition"] = config.get("edition")
    deadline = time.monotonic() + options.timeout_seconds
    try:
        if socket_factory is None:
            while time.monotonic() < deadline:
                try:
                    client = _JsonSocket(socket.create_connection(("127.0.0.1", port), timeout=5))
                    break
                except OSError:
                    if process.poll() is not None:
                        raise RuntimeError(f"backend exited with code {process.returncode}")
                    time.sleep(0.2)
            if client is None:
                raise TimeoutError("backend did not accept a connection")
        else:
            try:
                client = socket_factory("127.0.0.1", port, timeout=5)
            except TypeError:
                client = socket_factory("127.0.0.1", port)

        def request(command: str, **fields: Any) -> dict[str, Any]:
            report["commands"].append(command)
            try:
                raw_response = client.request(command, **fields)
            except Exception as exc:
                report["results"].append({"command": command, "error": str(exc)})
                raise
            report["results"].append({"command": command, "response": raw_response})
            return _ok(raw_response, command)

        while True:
            hello = request("hello")
            if hello.get("ready") is True:
                break
            if hello.get("status") != "loading":
                raise RuntimeError(f"backend hello failed: {hello}")
            if time.monotonic() >= deadline:
                raise TimeoutError("backend did not become ready")
            time.sleep(0.25)
        instance_token = hello.get("instance_token")
        if not isinstance(instance_token, str) or not instance_token.strip():
            raise RuntimeError("backend hello omitted instance token")
        initial = request("list_workpieces")
        if not isinstance(initial.get("workpieces"), list):
            raise RuntimeError("initial list_workpieces response omitted workpieces list")
        if initial["workpieces"] != []:
            raise RuntimeError("packaged data is not empty before smoke")
        registered = request("register", name="portable-smoke", replace=False,
                             front_images=front, back_images=back, progress_events=True)
        counts = registered.get("template_counts") or {}
        expected_counts = {"front": len(front), "back": len(back)}
        if counts != expected_counts:
            raise RuntimeError(f"template count mismatch: {counts} != {expected_counts}")
        workpiece_id = registered.get("workpiece", {}).get("id")
        if not workpiece_id:
            raise RuntimeError("register did not return workpiece id")
        listed = request("list_workpieces")
        if not any(item.get("id") == workpiece_id for item in listed.get("workpieces", [])):
            raise RuntimeError("registered workpiece missing from list")
        if request("predict", workpiece_id=workpiece_id, image_path=front_query).get("label") != "front":
            raise RuntimeError("front prediction mismatch")
        if request("predict", workpiece_id=workpiece_id, image_path=back_query).get("label") != "back":
            raise RuntimeError("back prediction mismatch")
        request("get_geometry_mask_profile", workpiece_id=workpiece_id)
        request("recycle_workpiece", workpiece_id=workpiece_id, operation_id="portable-smoke-recycle")
        recycled = request("list_recycled_workpieces")
        if not any(item.get("id") == workpiece_id for item in recycled.get("workpieces", [])):
            raise RuntimeError("recycled workpiece missing from recycle list")
        restored = request("restore_workpiece", workpiece_id=workpiece_id, operation_id="portable-smoke-restore")
        restored_item = restored.get("workpiece", {})
        if restored_item.get("id") != workpiece_id:
            raise RuntimeError("restored workpiece missing from restore response")
        restored_list = request("list_workpieces")
        if not any(item.get("id") == workpiece_id for item in restored_list.get("workpieces", [])):
            raise RuntimeError("restored workpiece missing from list")
        request("shutdown", instance_token=instance_token)
        process.wait(timeout=10)
        if process.returncode != 0:
            raise RuntimeError(f"backend exited with code {process.returncode}")
        report.update(ok=True, workpiece_id=workpiece_id, template_counts=counts, process_exit=process.returncode,
                      temp_package=str(temp_package), front_query=front_query, back_query=back_query,
                      labels={"front": "front", "back": "back"})
    except Exception as exc:
        report["error"] = str(exc)
        setattr(exc, "smoke_report", report)
        raise
    finally:
        primary_error = "error" in report
        cleanup_error: Exception | None = None
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if getattr(process, "poll", lambda: 0)() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        _collect_backend_log(report, process)
        report["process_exit"] = _process_exit_code(process)
        if options.report_path:
            persist_error = _persist_report_safely(Path(options.report_path), report)
            if persist_error is not None and not primary_error:
                raise persist_error
        if cleanup_error is not None and not primary_error:
            raise cleanup_error
    return report


def run_portability(options: SmokeOptions, **factories: Any) -> SmokeReport:
    """Register on a source edition, transfer its data copy, and validate destination."""
    if not options.source_package_root or not options.destination_package_root:
        raise ValueError("source_package_root and destination_package_root are required")
    destination = Path(options.destination_package_root).resolve()
    report = SmokeReport(ok=False, source_package_root=str(Path(options.source_package_root).resolve()),
                         destination_package_root=str(destination), commands=[], results=[])

    def ensure_empty_data(root: Path) -> None:
        data = root / "data"
        if not data.exists():
            return
        allowed = {"workpieces", "rules", "cache", "logs", "temp", "data_layout.json"}
        for entry in data.iterdir():
            if entry.name not in allowed:
                raise RuntimeError("destination package data must be empty before portability copy")
            if entry.is_file():
                if entry.name != "data_layout.json":
                    raise RuntimeError("destination package data must be empty before portability copy")
            elif any(entry.rglob("*")):
                raise RuntimeError("destination package data must be empty before portability copy")

    try:
        ensure_empty_data(destination)
    except Exception as exc:
        report["error"] = str(exc)
        partial = getattr(exc, "smoke_report", None)
        if isinstance(partial, dict):
            report["source"] = partial
            report["commands"] = list(partial.get("commands", []))
            report["results"] = list(partial.get("results", []))
        setattr(exc, "smoke_report", report)
        _persist_report_safely(options.report_path, report)
        raise
    source_factories = dict(factories)
    try:
        source = run_smoke(
            SmokeOptions(package_root=options.source_package_root, dataset_root=options.dataset_root,
                         front_template_count=options.front_template_count, back_template_count=options.back_template_count,
                         seed=options.seed, timeout_seconds=options.timeout_seconds),
            process_factory=source_factories.get("process_factory"),
            socket_factory=source_factories.get("socket_factory"),
            temp_root_factory=source_factories.get("temp_root_factory"),
            free_port_factory=source_factories.get("free_port_factory"),
            port_factory=source_factories.get("port_factory"),
        )
    except Exception as exc:
        report["error"] = str(exc)
        partial = getattr(exc, "smoke_report", None)
        if isinstance(partial, dict):
            report["source"] = partial
            report["commands"] = list(partial.get("commands", []))
            report["results"] = list(partial.get("results", []))
        setattr(exc, "smoke_report", report)
        _persist_report_safely(options.report_path, report)
        raise
    report["source"] = source
    src_data = Path(source["temp_package"]) / "data"
    destination_factory = factories.get("destination_temp_root_factory") or factories.get("temp_root_factory")
    if destination_factory is not None:
        candidate = Path(destination_factory())
        source_temp = Path(source["temp_package"]).resolve()
        if candidate.resolve() == source_temp:
            candidate = candidate.with_name(candidate.name + "-destination")
        destination_factory = lambda candidate=candidate: candidate
    try:
        dest_copy = _copy_to_long_temp(destination, destination_factory)
        # The copied destination must also be empty; this catches a factory that
        # returned an already-populated package independently of the original root.
        ensure_empty_data(dest_copy)
        shutil.rmtree(dest_copy / "data")
        shutil.copytree(src_data, dest_copy / "data")
        cfg = _package_config(dest_copy)
        process_factory = factories.get("process_factory") or _real_process_factory
        port = (factories.get("free_port_factory") or factories.get("port_factory") or _free_port)()
        process = _start_backend(dest_copy, cfg, port, process_factory)
    except Exception as exc:
        report["error"] = str(exc)
        setattr(exc, "smoke_report", report)
        _persist_report_safely(options.report_path, report)
        raise
    client = None
    hello: dict[str, Any] = {}
    destination_instance_token: str | None = None
    destination_report: dict[str, Any] = {"commands": [], "results": []}
    primary_error: Exception | None = None

    def request(command: str, **fields: Any) -> dict[str, Any]:
        destination_report["commands"].append(command)
        try:
            raw = client.request(command, **fields)
        except Exception as exc:
            destination_report["results"].append({"command": command, "error": str(exc)})
            raise
        destination_report["results"].append({"command": command, "response": raw})
        return _ok(raw, command)

    try:
        sf = factories.get("socket_factory")
        if sf:
            try:
                client = sf("127.0.0.1", port, timeout=5)
            except TypeError:
                client = sf("127.0.0.1", port)
        else:
            client = _JsonSocket(socket.create_connection(("127.0.0.1", port), timeout=5))
        deadline = time.monotonic() + options.timeout_seconds
        while True:
            hello = request("hello")
            if hello.get("ready") is True:
                break
            if hello.get("status") != "loading" or time.monotonic() >= deadline:
                raise RuntimeError("destination did not become ready")
            time.sleep(0.25)
        destination_instance_token = hello.get("instance_token")
        if not isinstance(destination_instance_token, str) or not destination_instance_token.strip():
            raise RuntimeError("destination hello omitted instance token")
        listed = request("list_workpieces")
        workpieces = listed.get("workpieces")
        if not isinstance(workpieces, list):
            raise RuntimeError("destination list_workpieces response omitted workpieces list")
        destination_ids = [item.get("id") for item in workpieces if isinstance(item, dict)]
        if len(workpieces) != 1 or destination_ids != [source["workpiece_id"]]:
            raise RuntimeError("destination workpiece set does not match source migration set")
        item = workpieces[0]
        if item is None:
            raise RuntimeError("copied workpiece missing on destination")
        counts = item.get("template_counts")
        if not isinstance(counts, dict):
            raise RuntimeError("destination response omitted template counts")
        if counts != source["template_counts"]:
            raise RuntimeError("destination template counts mismatch")
        front = request("predict", workpiece_id=source["workpiece_id"], image_path=source["front_query"])
        back = request("predict", workpiece_id=source["workpiece_id"], image_path=source["back_query"])
        if front.get("label") != "front" or back.get("label") != "back":
            raise RuntimeError("destination prediction mismatch")
        request("get_geometry_mask_profile", workpiece_id=source["workpiece_id"])
        destination_report.update(
            workpiece_id=source["workpiece_id"],
            hello=hello,
            template_counts=counts,
            labels={"front": front.get("label"), "back": back.get("label")},
        )
    except Exception as exc:
        primary_error = exc
        report["error"] = str(exc)
    finally:
        shutdown_error: Exception | None = None
        if client is not None and isinstance(hello.get("instance_token"), str) and hello.get("instance_token").strip():
            try:
                request("shutdown", instance_token=destination_instance_token)
            except Exception as exc:
                shutdown_error = exc
        try:
            if getattr(process, "poll", lambda: None)() is None:
                process.wait(timeout=10)
        except Exception as exc:
            if shutdown_error is None:
                shutdown_error = exc
        report["destination"] = destination_report
        report["commands"] = list(report.get("source", {}).get("commands", [])) + list(destination_report["commands"])
        report["results"] = list(report.get("source", {}).get("results", [])) + list(destination_report["results"])
        destination_report["process_exit"] = _process_exit_code(process)
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                if shutdown_error is None:
                    shutdown_error = exc
        if getattr(process, "poll", lambda: 0)() is None:
            try:
                process.terminate(); process.wait(timeout=10)
            except Exception as exc:
                if shutdown_error is None:
                    shutdown_error = exc
        _collect_backend_log(destination_report, process)
        destination_report["process_exit"] = _process_exit_code(process)
        if primary_error is None and shutdown_error is not None:
            primary_error = shutdown_error
            report["error"] = str(shutdown_error)
        if primary_error is None and destination_report["process_exit"] not in (None, 0):
            primary_error = RuntimeError(f"destination exited with code {destination_report['process_exit']}")
            report["error"] = str(primary_error)
        report["ok"] = primary_error is None
        if options.report_path:
            persist_error = _persist_report_safely(Path(options.report_path), report)
            if persist_error is not None and primary_error is None:
                raise persist_error
    if primary_error is not None:
        raise primary_error
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--source-package-root", type=Path)
    parser.add_argument("--destination-package-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--front-template-count", type=int, default=5)
    parser.add_argument("--back-template-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.source_package_root and args.destination_package_root:
        report = run_portability(SmokeOptions(source_package_root=args.source_package_root,
            destination_package_root=args.destination_package_root, dataset_root=args.dataset_root,
            front_template_count=args.front_template_count, back_template_count=args.back_template_count,
            seed=args.seed, report_path=args.report))
        print(json.dumps(report, ensure_ascii=False, default=str)); return 0
    root = args.package_root or args.source_package_root
    if root is None:
        parser.error("--package-root is required")
    report = run_smoke(SmokeOptions(package_root=root, dataset_root=args.dataset_root,
        front_template_count=args.front_template_count, back_template_count=args.back_template_count,
        seed=args.seed, report_path=args.report, source_package_root=args.source_package_root,
        destination_package_root=args.destination_package_root))
    print(json.dumps(report, ensure_ascii=False))
    return 0


__all__ = ["SmokeOptions", "SmokeReport", "run_smoke", "run_portability", "main"]
