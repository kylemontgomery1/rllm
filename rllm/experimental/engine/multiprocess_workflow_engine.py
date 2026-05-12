from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from rllm.agents.agent import Episode

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from rllm.utils.episode_logger import EpisodeLogger
    from rllm.workflows.store import Store
    from rllm.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


# ======================================================================
# Worker process entry point
# ======================================================================

def _worker_main(
    worker_id: int,
    rollout_engine_cls: type,
    rollout_config: Any,
    workflow_cls: type[Workflow],
    workflow_args: dict,
    n_parallel_tasks: int,
    retry_limit: int,
    raise_on_error: bool,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    weight_version: mp.Value,
    gate_event: mp.Event,
    store: Store | None = None,
):
    """Entry point for a worker process. Runs its own asyncio event loop."""
    asyncio.run(_worker_async_main(
        worker_id, rollout_engine_cls, rollout_config, workflow_cls, workflow_args,
        n_parallel_tasks, retry_limit, raise_on_error,
        task_queue, result_queue, weight_version, gate_event, store,
    ))


async def _event_loop_probe(worker_id: int, interval_s: float = 60):
    """Log event loop latency stats every interval_s seconds."""
    import time
    import numpy as np
    buf = []
    last_log = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        await asyncio.sleep(0.01)
        buf.append(time.perf_counter() - t0 - 0.01)
        if time.perf_counter() - last_log >= interval_s and buf:
            arr = np.array(buf)
            logger.info(
                "Worker %d event loop probe (last %ds): n=%d, mean=%.3fs, p50=%.3fs, p90=%.3fs, p99=%.3fs, max=%.3fs",
                worker_id, interval_s, len(arr), np.mean(arr), np.median(arr),
                np.percentile(arr, 90), np.percentile(arr, 99), np.max(arr),
            )
            buf.clear()
            last_log = time.perf_counter()


async def _worker_async_main(
    worker_id: int,
    rollout_engine_cls: type,
    rollout_config: Any,
    workflow_cls: type[Workflow],
    workflow_args: dict,
    n_parallel_tasks: int,
    retry_limit: int,
    raise_on_error: bool,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    weight_version: mp.Value,
    gate_event: mp.Event,
    store: Store | None = None,
):
    """Async main loop for a worker process."""
    from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.utils.logging import configure_logging_from_env

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
    configure_logging_from_env()

    # Build rollout engine from config
    rollout_engine = rollout_engine_cls.from_config(rollout_config)

    # Create workflow engine
    engine = UnifiedWorkflowEngine(
        workflow_cls=workflow_cls,
        workflow_args=workflow_args,
        rollout_engine=rollout_engine,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=retry_limit,
        raise_on_error=raise_on_error,
        store=store,
    )
    await engine.initialize_pool()

    # probe = asyncio.create_task(_event_loop_probe(worker_id))
    logger.debug("Worker %d initialized: %d parallel tasks", worker_id, n_parallel_tasks)

    active_tasks: set[asyncio.Task] = set()

    async def _process_one(task, task_id, rollout_idx, result_idx, kwargs):
        # Sync weight version from shared value
        rollout_engine.weight_version = weight_version.value
        # Wait for gate (blocks during weight sync)
        while not gate_event.is_set():
            await asyncio.sleep(0.05)
        try:
            result = await engine.process_task_with_retry(
                task, task_id, rollout_idx, result_idx, **kwargs,
            )
            result_queue.put(result)
        except Exception as e:
            logger.error("Worker %d task %s:%d failed: %s", worker_id, task_id, rollout_idx, e)
            raise

    while True:
        # Read from task queue in a thread to avoid blocking the event loop
        item = await asyncio.to_thread(task_queue.get)

        if item is None:
            # Shutdown sentinel
            break

        task, task_id, rollout_idx, result_idx, kwargs = item

        # Wait if at capacity
        while len(active_tasks) >= n_parallel_tasks:
            done, active_tasks_new = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            active_tasks.clear()
            active_tasks.update(active_tasks_new)
            for t in done:
                if t.exception():
                    logger.error("Worker %d task exception: %s", worker_id, t.exception())

        t = asyncio.create_task(_process_one(task, task_id, rollout_idx, result_idx, kwargs))
        active_tasks.add(t)

    # Drain remaining tasks
    if active_tasks:
        done, _ = await asyncio.wait(active_tasks)
        for t in done:
            if t.exception():
                logger.error("Worker %d drain exception: %s", worker_id, t.exception())

    # probe.cancel()
    logger.info("Worker %d shut down", worker_id)


# ======================================================================
# MultiProcessWorkflowEngine
# ======================================================================

class MultiProcessWorkflowEngine:
    """Orchestrates workflow execution across N worker processes.

    Same interface as UnifiedWorkflowEngine. When n_workers=1, falls back
    to a single in-process UnifiedWorkflowEngine with no subprocess overhead.
    """

    def __init__(
        self,
        workflow_cls: type[Workflow],
        workflow_args: dict,
        rollout_engine_cls: type,
        rollout_config: Any = None,
        n_parallel_tasks: int = 128,
        n_workers: int = 1,
        retry_limit: int = 3,
        raise_on_error: bool = True,
        episode_logger: EpisodeLogger | None = None,
        post_execute_hook: Callable[[], Coroutine[Any, Any, None]] | None = None,
        store: Store | None = None,
        output_dir: str | Path | None = None,
    ):
        self.workflow_cls = workflow_cls
        self.workflow_args = workflow_args
        self.rollout_engine_cls = rollout_engine_cls
        self.rollout_config = rollout_config
        self.n_workers = n_workers
        self.n_parallel_tasks = n_parallel_tasks
        self.n_parallel_tasks_per_worker = n_parallel_tasks // max(n_workers, 1)
        self.retry_limit = retry_limit
        self.raise_on_error = raise_on_error
        self.episode_logger = episode_logger
        self.post_execute_hook = post_execute_hook
        self.store = store
        self.output_dir = Path(output_dir) if output_dir is not None else None

        # Training step metadata
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"

        # Multi-process state
        self._processes: list[mp.Process] = []
        self._task_queues: list[mp.Queue] = []
        self._result_queue: mp.Queue | None = None
        self._weight_version: mp.Value | None = None
        self._gate_event: mp.Event | None = None
        self._pending_results: dict[tuple[str, int, int], asyncio.Future] = {}
        self._result_collector_task: asyncio.Task | None = None

        # Single-process fallback
        self._single_engine = None

    @staticmethod
    def _result_key(task_id: str, rollout_idx: int, result_idx: int) -> tuple[str, int, int]:
        return task_id, rollout_idx, result_idx

    def _ensure_result_collector(self) -> None:
        if self._single_engine is not None:
            return
        if self._result_queue is None:
            raise RuntimeError("result queue is not initialized")
        if self._result_collector_task is None or self._result_collector_task.done():
            self._result_collector_task = asyncio.create_task(self._collect_results())

    async def _collect_results(self) -> None:
        """Route worker results from the shared queue to their waiting callers."""
        assert self._result_queue is not None
        while True:
            result = await asyncio.to_thread(self._result_queue.get)
            if result is None:
                return

            task_id, rollout_idx, result_idx, _episode = result
            key = self._result_key(task_id, rollout_idx, result_idx)
            future = self._pending_results.pop(key, None)
            if future is None:
                logger.error("Received multiprocess workflow result with no waiter: %s", key)
                if not self._pending_results:
                    return
                continue
            if not future.done():
                future.set_result(result)
            if not self._pending_results:
                return

    def _submit_task(
        self,
        task: dict,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        kwargs: dict,
        worker_idx: int,
    ) -> asyncio.Future:
        self._ensure_result_collector()
        key = self._result_key(task_id, rollout_idx, result_idx)
        if key in self._pending_results:
            raise RuntimeError(f"Duplicate in-flight multiprocess workflow result key: {key}")

        future = asyncio.get_running_loop().create_future()
        self._pending_results[key] = future
        future.add_done_callback(lambda f, k=key: self._pending_results.pop(k, None) if f.cancelled() else None)
        self._task_queues[worker_idx].put((task, task_id, rollout_idx, result_idx, kwargs))
        return future

    async def initialize_pool(self):
        """Initialize workers. Idempotent."""
        if self._single_engine is not None or self._processes:
            return

        if self.n_workers <= 1:
            self._init_single_process()
        else:
            self._init_multi_process()

    def _init_single_process(self):
        """Single-process fallback — no subprocesses."""
        from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine

        rollout_engine = self.rollout_engine_cls.from_config(self.rollout_config)
        self._single_engine = UnifiedWorkflowEngine(
            workflow_cls=self.workflow_cls,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            n_parallel_tasks=self.n_parallel_tasks_per_worker,
            retry_limit=self.retry_limit,
            raise_on_error=self.raise_on_error,
            episode_logger=self.episode_logger,
            post_execute_hook=self.post_execute_hook,
            store=self.store,
            output_dir=self.output_dir,
        )

    def _init_multi_process(self):
        """Spawn N worker processes."""
        self._result_queue = mp.Queue()
        self._weight_version = mp.Value("i", 0)
        self._gate_event = mp.Event()
        self._gate_event.set()  # open by default

        for i in range(self.n_workers):
            task_queue = mp.Queue()
            self._task_queues.append(task_queue)

            p = mp.Process(
                target=_worker_main,
                args=(
                    i,
                    self.rollout_engine_cls,
                    self.rollout_config,
                    self.workflow_cls,
                    self.workflow_args,
                    self.n_parallel_tasks_per_worker,
                    self.retry_limit,
                    self.raise_on_error,
                    task_queue,
                    self._result_queue,
                    self._weight_version,
                    self._gate_event,
                    self.store,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        logger.info(
            "MultiProcessWorkflowEngine: spawned %d workers, %d tasks/worker (%d total)",
            self.n_workers,
            self.n_parallel_tasks_per_worker,
            self.n_workers * self.n_parallel_tasks_per_worker,
        )

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def execute_tasks(
        self,
        tasks: list[dict],
        task_ids: list[str] | None = None,
        is_validation: bool = False,
        post_process_fn=None,
        keep_in_memory: bool = True,
        **kwargs,
    ) -> list[Episode]:
        """Execute tasks across workers. Same interface as UnifiedWorkflowEngine."""
        if self._single_engine is None and not self._processes:
            await self.initialize_pool()

        if self._single_engine is not None:
            return await self._single_engine.execute_tasks(
                tasks, task_ids, is_validation=is_validation,
                post_process_fn=post_process_fn,
                keep_in_memory=keep_in_memory,
                **kwargs,
            )

        return await self._execute_tasks_distributed(
            tasks, task_ids, is_validation=is_validation,
            post_process_fn=post_process_fn,
            keep_in_memory=keep_in_memory,
            **kwargs,
        )

    async def _execute_tasks_distributed(
        self,
        tasks: list[dict],
        task_ids: list[str] | None = None,
        is_validation: bool = False,
        post_process_fn=None,
        keep_in_memory: bool = True,
        **kwargs,
    ) -> list[Episode]:
        """Distribute tasks across worker processes and collect results."""
        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        # Group tasks by task_id, assign groups to workers round-robin
        task_groups: dict[str, list[tuple[int, dict, str]]] = defaultdict(list)
        for i, (task, tid) in enumerate(zip(tasks, task_ids)):
            task_groups[tid].append((i, task, tid))

        # Build per-worker assignments
        worker_orig_indices: list[list[int]] = [[] for _ in range(self.n_workers)]
        futures: list[asyncio.Future] = []
        total_dispatched = 0

        skipped = 0
        for group_idx, (tid, items) in enumerate(task_groups.items()):
            w = group_idx % self.n_workers
            rollout_counter = 0
            for orig_idx, task, task_id in items:
                if self.output_dir is not None and (self.output_dir / f"{task_id}:{rollout_counter}.json").exists():
                    skipped += 1
                    rollout_counter += 1
                    continue
                futures.append(self._submit_task(task, task_id, rollout_counter, orig_idx, kwargs, w))
                worker_orig_indices[w].append(orig_idx)
                rollout_counter += 1
                total_dispatched += 1

        if skipped:
            logger.info(f"Skipped {skipped} tasks with existing output files")

        # Collect results from the shared result queue
        results: list[Episode | None] = [None] * len(tasks)
        collected = 0

        with tqdm(total=total_dispatched, desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                result = await future
                task_id, rollout_idx, result_idx, episode = result

                if self.output_dir is not None:
                    try:
                        self.output_dir.mkdir(parents=True, exist_ok=True)
                        serialize = post_process_fn or (lambda ep: ep.to_dict())
                        episode_data = serialize(episode)
                        episode_path = self.output_dir / f"{task_id}:{rollout_idx}.json"
                        with open(episode_path, "w") as f:
                            json.dump(episode_data, f, ensure_ascii=False)
                    except Exception as e:
                        logger.warning(f"Failed to save episode {task_id}:{rollout_idx}: {e}")

                if keep_in_memory:
                    results[result_idx] = episode

                collected += 1
                pbar.update(1)

        if not keep_in_memory:
            return []

        # Post-execute hook
        if self.post_execute_hook is not None:
            await self.post_execute_hook()

        # Episode logging
        if self.episode_logger is not None:
            try:
                self.episode_logger.log_episodes_batch(
                    results, self.current_step, self.current_mode, self.current_epoch,
                )
            except Exception as e:
                logger.error(f"Failed to log episodes: {e}")

        return results

    async def process_task_with_retry(
        self,
        task: dict,
        task_id: str,
        rollout_idx: int,
        result_idx: int,
        **kwargs,
    ) -> tuple[str, int, int, Episode]:
        """Process a single task. Routes to a worker by task_id hash."""
        if self._single_engine is not None:
            return await self._single_engine.process_task_with_retry(
                task, task_id, rollout_idx, result_idx, **kwargs,
            )

        worker_idx = hash(task_id) % self.n_workers
        future = self._submit_task(task, task_id, rollout_idx, result_idx, kwargs, worker_idx)
        return await future

    # ------------------------------------------------------------------
    # Training step / metadata
    # ------------------------------------------------------------------

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0):
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

        if self._single_engine is not None:
            self._single_engine.set_training_step(step, mode, epoch)

    # ------------------------------------------------------------------
    # Weight version + gate (for training weight sync)
    # ------------------------------------------------------------------

    def set_weight_version(self, version: int):
        if self._single_engine is not None:
            self._single_engine.rollout_engine.weight_version = version
        elif self._weight_version is not None:
            self._weight_version.value = version

    def close_gate(self):
        if self._single_engine is not None:
            self._single_engine.rollout_engine.close_gate()
        elif self._gate_event is not None:
            self._gate_event.clear()

    def open_gate(self):
        if self._single_engine is not None:
            self._single_engine.rollout_engine.open_gate()
        elif self._gate_event is not None:
            self._gate_event.set()

    async def wait_for_drain(self):
        if self._single_engine is not None:
            await self._single_engine.rollout_engine.wait_for_drain()
        else:
            # Poll until all workers report no active calls
            # For now, just wait for the gate to be acknowledged
            # TODO: implement proper drain tracking across workers
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def workflow_queue(self):
        """For _wait_for_all_workflows_idle compatibility."""
        if self._single_engine is not None:
            return self._single_engine.workflow_queue
        return None

    def shutdown(self):
        """Shutdown workers and cleanup resources."""
        if self._single_engine is not None:
            self._single_engine.shutdown()
        else:
            if self._result_queue is not None:
                self._result_queue.put(None)
            if self._result_collector_task is not None:
                self._result_collector_task.cancel()
                self._result_collector_task = None
            for future in self._pending_results.values():
                if not future.done():
                    future.cancel()
            self._pending_results.clear()
            for q in self._task_queues:
                q.put(None)
            for p in self._processes:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
            self._processes.clear()
            self._task_queues.clear()
