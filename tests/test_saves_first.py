"""A shared trainer's queue serves a job's sampler save before the other jobs' training."""

import asyncio
import unittest

from server.store import InMemoryStore


class InMemorySavesFirstTest(unittest.TestCase):
  def test_a_queued_save_is_served_before_the_round_robin(self):
    async def run():
      store = InMemoryStore()
      for tenant in ("a", "b"):
        await store.put_request({"model_id": tenant, "request_id": f"{tenant}-fb", "op": "forward_backward"}, active_set_id="base")
      await store.put_request({"model_id": "c", "request_id": "c-save", "op": "save_weights_for_sampler"}, active_set_id="base")
      return [[r["request_id"] for r in await store.get_requests(active_set_id="base")] for _ in range(3)]

    self.assertEqual(asyncio.run(run()), [["c-save"], ["a-fb"], ["b-fb"]])

  def test_a_save_already_served_by_the_round_robin_is_skipped(self):
    async def run():
      store = InMemoryStore()
      await store.put_request({"model_id": "a", "request_id": "a-save", "op": "save_weights_for_sampler"})
      first = [r["request_id"] for r in await store.get_requests()]
      await store.put_request({"model_id": "b", "request_id": "b-fb", "op": "forward_backward"})
      return first, [r["request_id"] for r in await store.get_requests()]

    self.assertEqual(asyncio.run(run()), (["a-save"], ["b-fb"]))


if __name__ == "__main__":
  unittest.main()
