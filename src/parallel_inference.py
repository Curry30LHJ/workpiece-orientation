"""Generic bounded parallel execution for batch work items."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable, Sequence


@dataclass(frozen=True)
class BatchWorkItem:
    index: int
    image_path: Path


@dataclass(frozen=True)
class BatchItemResult:
    index: int
    image_path: Path
    value: object | None
    error: BaseException | None


class BatchInferencePool:
    """A reusable, bounded pool with one persistent context per slot."""

    def __init__(
        self,
        worker_count: int,
        create_worker: Callable[[int], object],
        run_item: Callable[[object, BatchWorkItem], object],
    ) -> None:
        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self._worker_count = worker_count
        self._run_item = run_item
        self._executor = ThreadPoolExecutor(max_workers=worker_count)
        self._contexts: list[object] = []
        self._slot_lock = threading.Lock()
        self._next_slot = 0
        self._thread_slots = threading.local()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        try:
            self._contexts = [create_worker(slot) for slot in range(worker_count)]
        except BaseException:
            self._executor.shutdown(wait=True)
            for context in self._contexts:
                self._close_context(context)
            raise

    @property
    def ready(self) -> bool:
        return not self._closed

    @property
    def worker_count(self) -> int:
        return self._worker_count

    def _context_for_current_thread(self) -> object:
        slot = getattr(self._thread_slots, "slot", None)
        if slot is None:
            with self._slot_lock:
                slot = self._next_slot
                self._next_slot += 1
            self._thread_slots.slot = slot
        return self._contexts[slot % self._worker_count]

    def _run(self, item: BatchWorkItem) -> BatchItemResult:
        try:
            value = self._run_item(self._context_for_current_thread(), item)
        except BaseException as error:
            return BatchItemResult(item.index, item.image_path, None, error)
        return BatchItemResult(item.index, item.image_path, value, None)

    def submit_many(self, items: Sequence[BatchWorkItem]) -> list[BatchItemResult]:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("pool is closed")
            if not items:
                raise ValueError("items must contain at least one item")
            indices = [item.index for item in items]
            if len(set(indices)) != len(indices):
                raise ValueError("duplicate item indices are not allowed")
            futures = [self._executor.submit(self._run, item) for item in items]
        return sorted((future.result() for future in futures), key=lambda result: result.index)

    @staticmethod
    def _close_context(context: object) -> None:
        close = getattr(context, "close", None)
        if callable(close):
            close()

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)
        first_error: BaseException | None = None
        for context in self._contexts:
            try:
                self._close_context(context)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
