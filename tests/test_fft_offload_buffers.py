"""Pinned host buffers the FFT trainer fills ahead of its first offload."""

import unittest

import torch
import torch.nn as nn

from training.fft_trainer_worker import FFTTrainingWorker


class SimpleModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(4, 4, bias=False)


def make_worker() -> FFTTrainingWorker:
  worker = FFTTrainingWorker()
  worker.base_model_name = "test-simple-model"
  worker.model = SimpleModel()
  worker.trainable_params = list(worker.model.parameters())
  return worker


class HostBufferTest(unittest.TestCase):
  def test_matching_spare_is_used_once(self):
    worker = make_worker()
    param = worker.trainable_params[0]
    spare = torch.empty(param.shape, dtype=param.dtype)
    worker._spare_buffers[(param, "exp_avg")] = spare
    self.assertIs(worker._host_buffer(torch.zeros_like(param), param, "exp_avg"), spare)
    self.assertNotIn((param, "exp_avg"), worker._spare_buffers)
    self.assertIsNot(worker._host_buffer(torch.zeros_like(param), param, "exp_avg"), spare)

  def test_mismatched_spare_is_not_used(self):
    worker = make_worker()
    param = worker.trainable_params[0]
    spare = torch.empty(param.shape, dtype=torch.bfloat16)
    worker._spare_buffers[(param, "grad")] = spare
    buffer = worker._host_buffer(torch.zeros_like(param), param, "grad")
    self.assertIsNot(buffer, spare)
    self.assertEqual(buffer.dtype, param.dtype)

  @unittest.skipUnless(torch.cuda.is_available(), "pinned memory needs CUDA")
  def test_offload_uses_the_buffers_pinned_in_the_background(self):
    worker = make_worker()
    worker.model.to("cuda")
    worker.prepare_model_for_training()
    # The trainer's order under time slicing: offload after loading, then per step
    # reload, backward, offload, reload, optimizer step, offload.
    worker.sleep()
    self.assertIsNotNone(worker._spare_filler)
    worker.wake_up()
    loss = worker.model.fc(torch.ones(2, 4, device="cuda")).sum()
    loss.backward()
    worker.sleep()
    worker.wake_up()
    worker.optim_step({})
    worker.sleep()
    self.assertEqual(worker._spare_buffers, {})
    param = worker.trainable_params[0]
    self.assertTrue(worker._grad_shadow[param][1].is_pinned())
    self.assertTrue(all(buf.is_pinned() for (owner, _), (_, buf) in worker._opt_shadow.items() if owner is param))
    worker.wake_up()
    self.assertEqual(param.device.type, "cuda")


if __name__ == "__main__":
  unittest.main()
