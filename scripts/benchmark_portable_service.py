#!/usr/bin/env python3
"""Command-line entry point for the portable package acceptance benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from release_tools.portable_benchmark import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a packaged offline backend")
    parser.add_argument("--package-zip", type=Path, required=True)
    parser.add_argument("--acceptance-spec", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_benchmark(args.package_zip, args.acceptance_spec, warmup=args.warmup,
                               iterations=args.iterations, compare=args.compare)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if report.get("passed") is True else 2
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"passed": False, "error": {"type": type(exc).__name__, "message": str(exc)},
                                          "process_exit": getattr(exc, "process_exit", None),
                                          "backend_log": getattr(exc, "backend_log", ""),
                                          "lifecycle": {"shutdown": "not_confirmed"}},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
