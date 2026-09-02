from src.windows_parent_watchdog import start_parent_watchdog


def test_parent_exit_requests_shutdown_once():
    calls = []
    thread = start_parent_watchdog(
        4321,
        lambda: calls.append("shutdown"),
        wait_for_exit=lambda pid: calls.append(f"wait:{pid}"),
    )
    thread.join(timeout=1)
    assert calls == ["wait:4321", "shutdown"]
