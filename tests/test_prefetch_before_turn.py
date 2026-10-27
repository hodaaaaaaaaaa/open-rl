"""A time-sliced trainer reads a create's checkpoint before it takes the GPU turn."""

import asyncio
import contextlib
import os
import unittest
from unittest.mock import patch

from server.store import InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor


class Slicer:
  faulted = None

  def __init__(self, events):
    self.events = events

  @contextlib.asynccontextmanager
  async def acquire(self, workload):
    self.events.append("acquire")
    yield
    self.events.append("release")


class Worker:
  cpu_offload = True

  def wake_up(self):
    pass

  def sleep(self):
    pass


class OneBatchStore(InMemoryStore):
  def __init__(self, batch):
    super().__init__()
    self.batch = batch

  async def get_requests_for_model(self, model_id):
    batch, self.batch = self.batch, []
    return batch

  async def set_result(self, request_id, result):
    pass


class PrefetchBeforeTurnTest(unittest.TestCase):
  def run_batch(self, batch, prefetch_error=None):
    events = []

    def record(path, log=print):
      events.append(("prefetch", path))
      if prefetch_error:
        raise prefetch_error

    with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
      proc = TrainingRequestsProcessor(OneBatchStore(batch), Worker(), "run-a", time_slicer=Slicer(events))

    async def handle(request):
      events.append(("handle", request["op"]))
      return request["request_id"], {"type": "ok"}

    proc.handle_request = handle
    with patch("server.training_requests_processor.prefetch", side_effect=record):
      asyncio.run(proc.run_once())
    return events

  def test_create_reads_its_checkpoint_before_the_turn(self):
    events = self.run_batch([{"request_id": "c", "model_id": "run-a", "op": "create_model", "base_model": "org/model"}])
    self.assertEqual(events, [("prefetch", "org/model"), "acquire", ("handle", "create_model"), "release"])

  def test_restore_reads_its_state_before_the_turn(self):
    events = self.run_batch([{"request_id": "c", "model_id": "run-a", "op": "create_model_from_state", "state_path": "/mnt/state"}])
    self.assertEqual(events[0], ("prefetch", "/mnt/state"))

  def test_training_requests_read_nothing(self):
    events = self.run_batch([{"request_id": "f", "model_id": "run-a", "op": "forward_backward", "data": [], "loss_fn": "cross_entropy"}])
    self.assertEqual(events[0], "acquire")

  def test_a_failed_prefetch_still_takes_the_turn(self):
    events = self.run_batch([{"request_id": "c", "model_id": "run-a", "op": "create_model", "base_model": "org/model"}], OSError("stale"))
    self.assertEqual(events[1:], ["acquire", ("handle", "create_model"), "release"])


if __name__ == "__main__":
  unittest.main()
