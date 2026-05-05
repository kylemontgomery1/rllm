import asyncio
import queue

import pytest

from rllm.experimental.engine.multiprocess_workflow_engine import MultiProcessWorkflowEngine


class FakeTaskQueue:
    def __init__(self):
        self.puts = []

    def put(self, item):
        self.puts.append(item)


class FakeResultQueue:
    def __init__(self):
        self.items = queue.Queue()

    def put(self, item):
        self.items.put(item)

    def get(self):
        return self.items.get(timeout=2)


def make_engine(n_workers=1):
    engine = MultiProcessWorkflowEngine(
        workflow_cls=object,
        workflow_args={},
        rollout_engine_cls=object,
        n_workers=n_workers,
    )
    engine._single_engine = None
    engine._task_queues = [FakeTaskQueue() for _ in range(n_workers)]
    engine._result_queue = FakeResultQueue()
    engine._processes = [object() for _ in range(n_workers)]
    return engine


@pytest.mark.asyncio
async def test_process_task_with_retry_returns_matching_out_of_order_result():
    engine = make_engine()

    task_a = asyncio.create_task(engine.process_task_with_retry({"x": 1}, "A", 0, 0))
    task_b = asyncio.create_task(engine.process_task_with_retry({"x": 2}, "B", 0, 0))
    await asyncio.sleep(0.1)

    engine._result_queue.put(("B", 0, 0, "episode-B"))
    engine._result_queue.put(("A", 0, 0, "episode-A"))

    assert await task_a == ("A", 0, 0, "episode-A")
    assert await task_b == ("B", 0, 0, "episode-B")
    assert engine._pending_results == {}
    assert engine._result_collector_task.done()


@pytest.mark.asyncio
async def test_execute_tasks_preserves_input_order_when_results_arrive_out_of_order():
    engine = make_engine()

    run = asyncio.create_task(
        engine.execute_tasks(
            [{"x": 1}, {"x": 2}, {"x": 3}],
            task_ids=["A", "B", "C"],
        )
    )
    await asyncio.sleep(0.1)

    engine._result_queue.put(("C", 0, 2, "episode-C"))
    engine._result_queue.put(("A", 0, 0, "episode-A"))
    engine._result_queue.put(("B", 0, 1, "episode-B"))

    assert await run == ["episode-A", "episode-B", "episode-C"]
    assert engine._pending_results == {}


@pytest.mark.asyncio
async def test_duplicate_in_flight_result_key_raises():
    engine = make_engine()

    first = asyncio.create_task(engine.process_task_with_retry({"x": 1}, "A", 0, 0))
    await asyncio.sleep(0.1)

    with pytest.raises(RuntimeError, match="Duplicate in-flight"):
        await engine.process_task_with_retry({"x": 2}, "A", 0, 0)

    engine._result_queue.put(("A", 0, 0, "episode-A"))
    assert await first == ("A", 0, 0, "episode-A")


@pytest.mark.asyncio
async def test_unknown_result_does_not_steal_waiting_result():
    engine = make_engine()

    task = asyncio.create_task(engine.process_task_with_retry({"x": 1}, "A", 0, 0))
    await asyncio.sleep(0.1)

    engine._result_queue.put(("unknown", 0, 0, "episode-unknown"))
    engine._result_queue.put(("A", 0, 0, "episode-A"))

    assert await task == ("A", 0, 0, "episode-A")
    assert engine._pending_results == {}


def test_set_weight_version_updates_shared_value():
    engine = make_engine()
    engine._weight_version = type("Value", (), {"value": 0})()

    engine.set_weight_version(7)

    assert engine._weight_version.value == 7
