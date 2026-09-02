import threading
from pathlib import Path

import pytest

from src.parallel_inference import BatchInferencePool, BatchWorkItem


def make_pool(run_item, worker_count=4, create_worker=None):
    return BatchInferencePool(
        worker_count,
        create_worker or (lambda _: object()),
        lambda worker, item: run_item(item),
    )


def test_submit_many_overlaps_workers_and_restores_input_order():
    barrier = threading.Barrier(2)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def run_item(_item):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            if _item.index < 2:
                barrier.wait(timeout=2)
            # Pairwise barrier participation makes all five tasks complete;
            # the short hold keeps overlap observable without a fragile timeout.
            threading.Event().wait(0.01)
            return _item.index
        finally:
            with lock:
                active -= 1

    pool = make_pool(run_item, worker_count=4)
    try:
        items = [BatchWorkItem(i, Path(f"{i}.png")) for i in range(5)]
        results = pool.submit_many(items)
        assert [item.index for item in results] == list(range(5))
        assert all(item.error is None for item in results)
        assert [item.value for item in results] == list(range(5))
        assert max_active >= 2
        assert max_active <= 4
    finally:
        pool.close()


def test_submit_many_keeps_other_items_when_one_item_fails():
    pool = make_pool(
        lambda item: (_ for _ in ()).throw(ValueError("bad"))
        if item.index == 2
        else item.index
    )
    try:
        results = pool.submit_many(
            [BatchWorkItem(i, Path(f"{i}.png")) for i in range(4)]
        )
        assert [item.index for item in results] == [0, 1, 2, 3]
        assert isinstance(results[2].error, ValueError)
        assert [item.value for item in results if item.error is None] == [0, 1, 3]
    finally:
        pool.close()


def test_submit_many_rejects_empty_input_and_invalid_worker_count():
    with pytest.raises(ValueError, match="worker_count"):
        BatchInferencePool(0, lambda _: object(), lambda *_: None)
    pool = make_pool(lambda item: item.index)
    try:
        with pytest.raises(ValueError, match="at least one"):
            pool.submit_many([])
    finally:
        pool.close()


def test_submit_many_rejects_duplicate_indices():
    pool = make_pool(lambda item: item.index)
    try:
        with pytest.raises(ValueError, match="duplicate"):
            pool.submit_many(
                [BatchWorkItem(1, Path("a.png")), BatchWorkItem(1, Path("b.png"))]
            )
    finally:
        pool.close()


def test_pool_reuses_workers_and_closes_contexts():
    created = []

    class Context:
        def __init__(self, slot):
            self.slot = slot
            self.closed = False

        def close(self):
            self.closed = True

    def create_worker(slot):
        context = Context(slot)
        created.append(context)
        return context

    pool = make_pool(lambda item: item.index, create_worker=create_worker)
    assert pool.ready
    assert pool.worker_count == 4
    assert len(created) == 4
    pool.submit_many([BatchWorkItem(0, Path("a.png"))])
    pool.submit_many([BatchWorkItem(1, Path("b.png"))])
    assert len(created) == 4
    pool.close()
    assert all(context.closed for context in created)
    assert not pool.ready


def test_close_continues_closing_contexts_when_one_fails():
    closed = []

    class Context:
        def __init__(self, slot):
            self.slot = slot

        def close(self):
            closed.append(self.slot)
            if self.slot == 1:
                raise RuntimeError("close failed")

    pool = make_pool(lambda item: item.index, create_worker=Context)
    with pytest.raises(RuntimeError, match="close failed"):
        pool.close()
    assert closed == [0, 1, 2, 3]
    assert not pool.ready


def test_submit_and_close_are_safe_when_called_concurrently():
    pool = make_pool(lambda item: item.index)
    submit_done = threading.Event()

    def submit():
        try:
            results = pool.submit_many([BatchWorkItem(0, Path("a.png"))])
            assert results[0].value == 0
        finally:
            submit_done.set()

    submit_thread = threading.Thread(target=submit)
    close_thread = threading.Thread(target=pool.close)
    submit_thread.start()
    close_thread.start()
    submit_thread.join(timeout=2)
    close_thread.join(timeout=2)
    assert submit_done.is_set()
    assert not submit_thread.is_alive()
    assert not close_thread.is_alive()
