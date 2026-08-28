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
    if target.exists():
        shutil.rmtree(target)
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
              temp_root_factory: Callable[..., Path] | None = None) -> dict[str, Any]:
    """Run a packaged protocol smoke test. Factories make the workflow unit-testable."""
    if options.package_root is None:
        raise ValueError("package_root is required")
    package = Path(options.package_root).resolve()
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
    process_factory = process_factory or _real_process_factory
    port = _free_port()
    temp_package = _copy_to_long_temp(package, temp_root_factory)
    config = _package_config(temp_package)
    process = _start_backend(temp_package, config, port, process_factory)
    client = None
    report: SmokeReport = SmokeReport(ok=False, package_root=str(package), edition=config.get("edition"), commands=[], results=[])
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
            response = _ok(client.request(command, **fields), command)
            report["commands"].append(command)
            report["results"].append({"command": command, "response": response})
            return response

        while True:
            hello = request("hello")
            if hello.get("ready") is True:
                break
            if hello.get("status") != "loading":
                raise RuntimeError(f"backend hello failed: {hello}")
            if time.monotonic() >= deadline:
                raise TimeoutError("backend did not become ready")
            time.sleep(0.25)
        initial = request("list_workpieces")
        if initial.get("workpieces"):
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
        request("shutdown", instance_token=getattr(process, "token", None) or hello.get("instance_token"))
        process.wait(timeout=10)
        if process.returncode != 0:
            raise RuntimeError(f"backend exited with code {process.returncode}")
        report.update(ok=True, workpiece_id=workpiece_id, template_counts=counts, process_exit=process.returncode,
                      temp_package=str(temp_package), front_query=front_query, back_query=back_query,
                      labels={"front": "front", "back": "back"})
        stream = getattr(process, "stdout", None)
        if stream is not None:
            try:
                report["backend_log"] = stream.read()
            except Exception:
                pass
    except Exception as exc:
        report["error"] = str(exc)
        report["process_exit"] = getattr(process, "returncode", None)
        raise
    finally:
        if client is not None:
            try: client.close()
            except Exception: pass
        if getattr(process, "poll", lambda: 0)() is None:
            try: process.terminate(); process.wait(timeout=10)
            except Exception: pass
        if options.report_path:
            Path(options.report_path).parent.mkdir(parents=True, exist_ok=True)
            report.setdefault("process_exit", getattr(process, "returncode", None))
            Path(options.report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def run_portability(options: SmokeOptions, **factories: Any) -> SmokeReport:
    """Register on a source edition, transfer its data copy, and validate destination."""
    if not options.source_package_root or not options.destination_package_root:
        raise ValueError("source_package_root and destination_package_root are required")
    destination = Path(options.destination_package_root).resolve()
    workpieces = destination / "data" / "workpieces"
    if workpieces.exists() and any(workpieces.iterdir()):
        raise RuntimeError("destination package data must be empty before portability copy")
    source = run_smoke(SmokeOptions(package_root=options.source_package_root, dataset_root=options.dataset_root,
        front_template_count=options.front_template_count, back_template_count=options.back_template_count,
        seed=options.seed, timeout_seconds=options.timeout_seconds), **factories)
    src_data = Path(source["temp_package"]) / "data"
    destination_factory = factories.get("temp_root_factory")
    if destination_factory is not None:
        original_factory = destination_factory
        destination_factory = lambda: Path(original_factory()).with_name(Path(original_factory()).name + "-destination")
    dest_copy = _copy_to_long_temp(destination, destination_factory)
    shutil.rmtree(dest_copy / "data")
    shutil.copytree(src_data, dest_copy / "data")
    # A destination run against copied data must not register a second workpiece.
    cfg = _package_config(dest_copy); process_factory = factories.get("process_factory") or _real_process_factory
    port = _free_port(); process = _start_backend(dest_copy, cfg, port, process_factory); client = None; hello = {}
    try:
        sf = factories.get("socket_factory")
        client = sf("127.0.0.1", port) if sf else _JsonSocket(socket.create_connection(("127.0.0.1", port), timeout=5))
        deadline = time.monotonic() + options.timeout_seconds
        while True:
            hello = _ok(client.request("hello"), "hello")
            if hello.get("ready") is True: break
            if hello.get("status") != "loading" or time.monotonic() >= deadline: raise RuntimeError("destination did not become ready")
            time.sleep(0.25)
        listed = _ok(client.request("list_workpieces"), "list_workpieces")
        item = next((x for x in listed.get("workpieces", []) if x.get("id") == source["workpiece_id"]), None)
        if item is None:
            raise RuntimeError("copied workpiece missing on destination")
        counts = item.get("template_counts") or item.get("templates")
        if counts is None: raise RuntimeError("destination response omitted template counts")
        if counts != source["template_counts"]: raise RuntimeError("destination template counts mismatch")
        front = _ok(client.request("predict", workpiece_id=source["workpiece_id"], image_path=source["front_query"]), "predict")
        back = _ok(client.request("predict", workpiece_id=source["workpiece_id"], image_path=source["back_query"]), "predict")
        if front.get("label") != "front" or back.get("label") != "back": raise RuntimeError("destination prediction mismatch")
        _ok(client.request("get_geometry_mask_profile", workpiece_id=source["workpiece_id"]), "get_geometry_mask_profile")
        result = SmokeReport(ok=True, source=source, destination={"workpiece_id": source["workpiece_id"], "hello": hello, "template_counts": counts, "labels": {"front": front.get("label"), "back": back.get("label")}, "process_exit": process.returncode, "commands": [{"command":"hello","response":hello},{"command":"list_workpieces","response":listed},{"command":"predict","response":front},{"command":"predict","response":back}]})
        if options.report_path:
            Path(options.report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(options.report_path).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return result
    finally:
        if client:
            client.request("shutdown", instance_token=hello.get("instance_token"))
            process.wait(timeout=10)
            if process.returncode != 0: raise RuntimeError(f"destination exited with code {process.returncode}")
            client.close()
        if process.poll() is None:
            process.terminate(); process.wait(timeout=10)


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


__all__ = ["SmokeOptions", "SmokeReport", "run_smoke", "main"]
