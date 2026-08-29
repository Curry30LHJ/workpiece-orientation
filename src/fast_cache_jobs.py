"""Single-worker background publication for revision-bound fast caches."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True)
class FastCacheJobSnapshot:
    workpiece_id: str
    library_revision: int
    geometry_profile_revision: int | None
    state: str
    completed: int
    total: int
    elapsed_ms: float
    error: str | None


@dataclass
class _FastCacheJob:
    snapshot: FastCacheJobSnapshot
    build: Callable[[Callable[[dict[str, Any]], None]], Any] | None
    publish: Callable[[Any], bool] | None
    started_at: float | None = None


class FastCacheJobManager:
    """Run revision-bound fast-cache builds one at a time."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fast-cache")
        self._worker: Future[None] | None = None
        self._jobs: dict[tuple[str, int, int | None], _FastCacheJob] = {}
        self._latest: dict[str, tuple[str, int, int | None]] = {}
        self._queue: deque[tuple[str, int, int | None]] = deque()
        self._stopping = False

    def schedule(
        self,
        *,
        workpiece_id: str,
        library_revision: int,
        geometry_profile_revision: int | None,
        build: Callable[[Callable[[dict[str, Any]], None]], Any],
        publish: Callable[[Any], bool],
    ) -> FastCacheJobSnapshot:
        key = (workpiece_id, int(library_revision), geometry_profile_revision)
        with self._condition:
            existing = self._jobs.get(key)
            if existing is not None:
                return self._current_snapshot(existing)
            if self._stopping:
                raise RuntimeError("fast cache job manager is shut down")
            job = _FastCacheJob(
                snapshot=FastCacheJobSnapshot(
                    workpiece_id=workpiece_id,
                    library_revision=int(library_revision),
                    geometry_profile_revision=geometry_profile_revision,
                    state="queued",
                    completed=0,
                    total=0,
                    elapsed_ms=0.0,
                    error=None,
                ),
                build=build,
                publish=publish,
            )
            self._jobs[key] = job
            previous_key = self._latest.get(workpiece_id)
            self._latest[workpiece_id] = key
            if previous_key is not None and previous_key != key:
                previous = self._jobs.get(previous_key)
                if previous is not None and previous.snapshot.state in {"ready", "failed", "stale"}:
                    self._jobs.pop(previous_key, None)
            self._queue.append(key)
            if self._worker is None or self._worker.done():
                self._worker = self._executor.submit(self._drain)
            return job.snapshot

    def snapshot(self, workpiece_id: str) -> FastCacheJobSnapshot | None:
        with self._condition:
            key = self._latest.get(workpiece_id)
            job = self._jobs.get(key) if key is not None else None
            return None if job is None else self._current_snapshot(job)

    def shutdown(self) -> None:
        with self._condition:
            self._stopping = True
            for key in self._queue:
                job = self._jobs[key]
                job.build = None
                job.publish = None
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.result()
        self._executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _current_snapshot(job: _FastCacheJob) -> FastCacheJobSnapshot:
        if job.started_at is None or job.snapshot.state not in {"running"}:
            return job.snapshot
        return replace(
            job.snapshot,
            elapsed_ms=(time.perf_counter() - job.started_at) * 1000.0,
        )

    def _drain(self) -> None:
        while True:
            with self._condition:
                if self._stopping or not self._queue:
                    self._worker = None
                    return
                key = self._queue.popleft()
                job = self._jobs[key]
                job.started_at = time.perf_counter()
                job.snapshot = replace(job.snapshot, state="running")
            try:
                if job.build is None or job.publish is None:
                    return
                cache = job.build(lambda event: self._progress(key, event))
                published = bool(job.publish(cache))
            except Exception as exc:
                state = "failed"
                error = str(exc)
            else:
                state = "ready" if published else "stale"
                error = None
            with self._condition:
                elapsed_ms = (time.perf_counter() - job.started_at) * 1000.0
                job.snapshot = replace(
                    job.snapshot,
                    state=state,
                    elapsed_ms=elapsed_ms,
                    error=error,
                )
                job.build = None
                job.publish = None
                if self._latest.get(job.snapshot.workpiece_id) != key:
                    self._jobs.pop(key, None)

    def _progress(self, key: tuple[str, int, int | None], event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        with self._condition:
            job = self._jobs.get(key)
            if job is None or job.snapshot.state != "running":
                return
            completed = event.get("completed", job.snapshot.completed)
            total = event.get("total", job.snapshot.total)
            if type(completed) is int and completed >= 0 and type(total) is int and total >= 0:
                job.snapshot = replace(job.snapshot, completed=completed, total=total)
