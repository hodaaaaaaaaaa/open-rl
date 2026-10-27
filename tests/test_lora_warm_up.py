"""The LoRA trainer's warm-up pass leaves the job's adapter, gradients and RNG as they were."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from training.lora_trainer_worker import LoraTrainingWorker
from training.types import LoraConfig


def tiny_worker(device: str) -> LoraTrainingWorker:
  torch.manual_seed(0)
  config = LlamaConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=1024,
  )
  worker = LoraTrainingWorker()
  worker.device = torch.device(device)
  worker.base_model = LlamaForCausalLM(config).to(device)
  worker.base_model_name = "tiny"
  worker.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
  return worker


def create(worker: LoraTrainingWorker, warm: bool) -> dict[str, torch.Tensor]:
  worker.warmed_up = not warm
  with patch.object(LoraTrainingWorker, "save_adapter"):
    worker.create_adapter("job", LoraConfig(rank=4, seed=3))
  return {name: p.detach().clone() for name, p in worker.peft_model.named_parameters() if "lora_" in name}


class WarmUpTest(unittest.TestCase):
  def test_cpu_workers_do_not_warm_up(self):
    worker = tiny_worker("cpu")
    create(worker, warm=True)
    self.assertFalse(worker.warmed_up)

  @unittest.skipUnless(torch.cuda.is_available(), "the warm-up runs on CUDA only")
  def test_warm_up_changes_nothing_the_job_sees(self):
    warmed, cold = tiny_worker("cuda"), tiny_worker("cuda")
    warmed_params = create(warmed, warm=True)
    warmed_rng = torch.cuda.get_rng_state()
    cold_params = create(cold, warm=False)
    self.assertEqual(warmed_rng.tolist(), torch.cuda.get_rng_state().tolist())
    self.assertTrue(warmed.warmed_up)
    self.assertEqual(warmed_params.keys(), cold_params.keys())
    for name in warmed_params:
      self.assertTrue(torch.equal(warmed_params[name], cold_params[name]), name)
    self.assertTrue(all(p.grad is None for p in warmed.adapter_states["job"]["trainable_params"]))


if __name__ == "__main__":
  unittest.main()
