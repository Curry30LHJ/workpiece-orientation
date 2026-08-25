from __future__ import annotations

import threading
import time

from src.model_execution_gate import PriorityModelGate


def test_gate_never_runs_two_model_actions_concurrently():
    gate = PriorityModelGate()
    first_entered = threading.Event()
    release_first = threading.Event()
    active = 0
    maximum = 0
    guard = threading.Lock()

    def first_action():
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        first_entered.set()
        assert release_first.wait(1)
        with guard:
            active -= 1

    def second_action():
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
            active -= 1

    first = threading.Thread(
        target=gate.run_background_step,
        args=(first_action,),
    )
    first.start()
    assert first_entered.wait(1)
    second = threading.Thread(target=gate.run_online, args=(second_action,))
    second.start()
    time.sleep(0.05)

    assert maximum == 1
    release_first.set()
    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive()
    assert not second.is_alive()


def test_waiting_online_action_runs_before_next_background_step():
    gate = PriorityModelGate()
    first_entered = threading.Event()
    release_first = threading.Event()
    online_submitted = threading.Event()
    order: list[str] = []

    def first_background():
        order.append("background-1")
        first_entered.set()
        assert release_first.wait(1)

    def online_action():
        order.append("online")

    def submit_online():
        online_submitted.set()
        gate.run_online(online_action)

    first = threading.Thread(
        target=gate.run_background_step,
        args=(first_background,),
    )
    first.start()
    assert first_entered.wait(1)

    online = threading.Thread(target=submit_online)
    online.start()
    assert online_submitted.wait(1)
    time.sleep(0.02)
    second = threading.Thread(
        target=gate.run_background_step,
        args=(lambda: order.append("background-2"),),
    )
    second.start()
    release_first.set()

    for thread in (first, online, second):
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert order == ["background-1", "online", "background-2"]
