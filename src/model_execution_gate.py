from __future__ import annotations

from threading import Condition
from typing import Callable, TypeVar


T = TypeVar("T")


class PriorityModelGate:
    """Serialize model work while favoring queued online inference."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._active = False
        self._online_waiters = 0

    def run_online(self, action: Callable[[], T]) -> T:
        with self._condition:
            self._online_waiters += 1
            try:
                self._condition.wait_for(lambda: not self._active)
                self._active = True
            finally:
                self._online_waiters -= 1
        try:
            return action()
        finally:
            self._release()

    def run_background_step(self, action: Callable[[], T]) -> T:
        with self._condition:
            self._condition.wait_for(
                lambda: not self._active and self._online_waiters == 0
            )
            self._active = True
        try:
            return action()
        finally:
            self._release()

    def _release(self) -> None:
        with self._condition:
            self._active = False
            self._condition.notify_all()
