"""A time-sliced trainer serves training requests that queue during its GPU turn in that same turn."""

import asyncio
import contextlib
import os
import unittest
from unittest.mock import patch

from server.store import InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor


class CountingSlicer:
  faulted = None

  def __init__(self):
    self.turns = 0

  @contextlib.asynccontextmanager
  async def acquire(self, workload):
    self.turns += 1
    yield


class Worker:
  cpu_offload = True

  def __init__(self):
    self.events = []

  def wake_up(self):
    self.events.append("wake")

  def sleep(self):
    self.events.append("sleep")


class QueueStore(InMemoryStore):
  """Pops the first batch, then hands out what is queued behind it."""

  def __init__(self, first, queued):
    super().__init__()
    self.first, self.queued = first, list(queued)
    self.futures = {}

  async def get_requests_for_model(self, model_id):
    batch, self.first = self.first, []
    return batch

  async def pop_leading_requests_for_model(self, model_id, ops):
    leading = []
    while self.queued and self.queued[0]["op"] in ops:
      leading.append(self.queued.pop(0))
    return leading

  async def set_result(self, request_id, result):
    self.futures[request_id] = result


def processor(store, worker, slicer, **kwargs):
  with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
    return TrainingRequestsProcessor(store, worker, "run-a", time_slicer=slicer, **kwargs)


class TrainerTurnTest(unittest.TestCase):
  def run_batch(self, first, queued):
    store, worker, slicer = QueueStore(first, queued), Worker(), CountingSlicer()
    proc = processor(store, worker, slicer)
    handled = []

    async def handle(request):
      handled.append(request["request_id"])
      return request["request_id"], {"type": request["op"]}

    proc.handle_request = handle
    asyncio.run(proc.run_once())
    return store, worker, slicer, handled

  def test_queued_optim_step_runs_in_the_same_turn(self):
    store, worker, slicer, handled = self.run_batch(
      [{"request_id": "fb", "op": "forward_backward"}], [{"request_id": "os", "op": "optim_step"}]
    )
    self.assertEqual(handled, ["fb", "os"])
    self.assertEqual(slicer.turns, 1)
    self.assertEqual(worker.events, ["wake", "sleep"])
    self.assertEqual(sorted(store.futures), ["fb", "os"])

  def test_a_save_stays_queued_for_the_next_batch(self):
    store, worker, slicer, handled = self.run_batch(
      [{"request_id": "fb", "op": "forward_backward"}],
      [{"request_id": "sv", "op": "save_weights_for_sampler"}, {"request_id": "os", "op": "optim_step"}],
    )
    self.assertEqual(handled, ["fb"])
    self.assertEqual([r["request_id"] for r in store.queued], ["sv", "os"])

  def test_a_failure_in_a_queued_request_answers_it(self):
    store, worker, slicer = QueueStore([{"request_id": "fb", "op": "forward_backward"}], [{"request_id": "os", "op": "optim_step"}]), Worker(), CountingSlicer()
    proc = processor(store, worker, slicer)

    async def handle(request):
      if request["op"] == "optim_step":
        raise RuntimeError("CUDA out of memory")
      return request["request_id"], {"type": request["op"]}

    proc.handle_request = handle
    with self.assertRaises(RuntimeError):
      asyncio.run(proc.run_once())
    self.assertEqual(store.futures["fb"]["type"], "forward_backward")
    self.assertEqual(store.futures["os"]["type"], "RequestFailedResponse")
    self.assertEqual(worker.events, ["wake", "sleep"])


if __name__ == "__main__":
  unittest.main()
