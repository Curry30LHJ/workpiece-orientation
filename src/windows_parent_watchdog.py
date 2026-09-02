"""Request graceful backend shutdown when its Windows parent process exits."""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import Callable


LOGGER = logging.getLogger(__name__)
_SYNCHRONIZE = 0x00100000
_INFINITE = 0xFFFFFFFF


def _windows_parent_waiter(parent_pid: int) -> Callable[[int], None] | None:
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(_SYNCHRONIZE, False, parent_pid)
    except (AttributeError, OSError):
        LOGGER.warning("Windows parent process monitoring is unavailable for PID %s", parent_pid)
        return None
    if not handle:
        LOGGER.warning("Unable to open parent process PID %s for monitoring", parent_pid)
        return None

    def wait_for_exit(_parent_pid: int) -> None:
        try:
            result = kernel32.WaitForSingleObject(handle, _INFINITE)
            if result != 0:
                raise OSError(f"WaitForSingleObject failed with result {result}")
        finally:
            kernel32.CloseHandle(handle)

    return wait_for_exit


def start_parent_watchdog(
    parent_pid: int | None,
    on_parent_exit: Callable[[], None],
    *,
    wait_for_exit: Callable[[int], None] | None = None,
) -> threading.Thread | None:
    """Start a daemon that requests shutdown after the owning parent exits."""
    if type(parent_pid) is not int or parent_pid <= 0:
        LOGGER.warning("Ignoring invalid parent process PID: %r", parent_pid)
        return None
    waiter = wait_for_exit or _windows_parent_waiter(parent_pid)
    if waiter is None:
        return None

    def watch() -> None:
        try:
            waiter(parent_pid)
        except Exception:
            LOGGER.exception("Parent process monitoring failed for PID %s", parent_pid)
            return
        try:
            on_parent_exit()
        except Exception:
            LOGGER.exception("Unable to request shutdown after parent PID %s exited", parent_pid)

    thread = threading.Thread(
        target=watch,
        name="orientation-parent-watchdog",
        daemon=True,
    )
    thread.start()
    return thread
