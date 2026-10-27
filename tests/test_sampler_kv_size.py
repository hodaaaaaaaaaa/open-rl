"""A sampler keeps the KV cache size its engine profiled and passes it to the next engine with the same setup."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from server import vllm_sampler


def fake_cuda(name="NVIDIA H100 80GB HBM3", total=80 * 2**30):
  return patch.multiple(
    "torch.cuda", is_available=Mock(return_value=True), get_device_properties=Mock(return_value=SimpleNamespace(name=name, total_memory=total))
  )


class KvSizePathTest(unittest.TestCase):
  def setUp(self):
    self.root = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"VLLM_CACHE_ROOT": self.root}))

  def test_same_setup_same_path_and_any_change_another(self):
    kwargs = {"model": "m", "max_num_seqs": 256, "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4]}}
    with fake_cuda():
      path = vllm_sampler.kv_size_path(kwargs)
      self.assertEqual(path, vllm_sampler.kv_size_path(dict(kwargs)))
      self.assertTrue(str(path).startswith(self.root))
      self.assertNotEqual(path, vllm_sampler.kv_size_path({**kwargs, "max_num_seqs": 512}))
    with fake_cuda(name="NVIDIA L4", total=24 * 2**30):
      self.assertNotEqual(path, vllm_sampler.kv_size_path(kwargs))

  def test_off_without_a_cache_root_or_when_disabled(self):
    with fake_cuda(), patch.dict(os.environ, {"OPEN_RL_VLLM_KV_SIZE_CACHE": "0"}):
      self.assertIsNone(vllm_sampler.kv_size_path({"model": "m"}))
    with fake_cuda(), patch.dict(os.environ, {}, clear=True):
      self.assertIsNone(vllm_sampler.kv_size_path({"model": "m"}))

  def test_write_then_read(self):
    path = Path(self.root) / "sizes" / "a.json"
    self.assertIsNone(vllm_sampler.read_kv_size(path))
    vllm_sampler.write_kv_size(path, 123)
    self.assertEqual(vllm_sampler.read_kv_size(path), 123)
    path.write_text("not json")
    self.assertIsNone(vllm_sampler.read_kv_size(path))


class BuildEngineTest(unittest.TestCase):
  def setUp(self):
    self.path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "a.json"
    vllm_sampler.write_kv_size(self.path, 123)
    self.engine_cls = self.enterContext(patch.object(vllm_sampler, "AsyncLLMEngine"))
    self.args = self.enterContext(patch.object(vllm_sampler, "AsyncEngineArgs", side_effect=lambda **kw: kw))

  def test_a_kept_size_skips_profiling(self):
    vllm_sampler.build_engine({"model": "m"}, self.path, 123)
    self.engine_cls.from_engine_args.assert_called_once_with({"model": "m", "kv_cache_memory_bytes": 123})

  def test_no_kept_size_profiles(self):
    vllm_sampler.build_engine({"model": "m"}, self.path, None)
    self.engine_cls.from_engine_args.assert_called_once_with({"model": "m"})

  def test_a_failed_build_with_a_kept_size_drops_it(self):
    self.engine_cls.from_engine_args.side_effect = RuntimeError("CUDA out of memory")
    with self.assertRaises(RuntimeError):
      vllm_sampler.build_engine({"model": "m"}, self.path, 123)
    self.assertFalse(self.path.exists())


class RememberKvSizeTest(unittest.TestCase):
  def test_keeps_what_the_workers_profiled(self):
    path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "a.json"
    engine = Mock(collective_rpc=AsyncMock(return_value=[456]))
    asyncio.run(vllm_sampler.remember_kv_size(engine, path))
    engine.collective_rpc.assert_awaited_once_with(vllm_sampler.worker_kv_cache_bytes)
    self.assertEqual(vllm_sampler.read_kv_size(path), 456)

  def test_a_failed_query_keeps_nothing(self):
    path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "a.json"
    engine = Mock(collective_rpc=AsyncMock(side_effect=RuntimeError("rpc")))
    asyncio.run(vllm_sampler.remember_kv_size(engine, path))
    self.assertFalse(path.exists())

  def test_the_worker_reports_its_profiled_size(self):
    self.assertEqual(vllm_sampler.worker_kv_cache_bytes(SimpleNamespace(available_kv_cache_memory_bytes=7)), 7)
    self.assertIsNone(vllm_sampler.worker_kv_cache_bytes(SimpleNamespace()))


if __name__ == "__main__":
  unittest.main()
