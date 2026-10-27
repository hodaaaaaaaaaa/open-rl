"""LoRA passes on a half-precision base model run under autocast and match the fp32 LoRA path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from training.lora_trainer_worker import LoraTrainingWorker
from training.types import Datum, LoraConfig, TensorData


def bf16_worker() -> LoraTrainingWorker:
  torch.manual_seed(0)
  config = LlamaConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=64,
  )
  worker = LoraTrainingWorker()
  worker.device = torch.device("cuda")
  worker.base_model = LlamaForCausalLM(config).to("cuda", torch.bfloat16)
  worker.base_model_name = "tiny"
  worker.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
  return worker


def lora_b_grads(worker: LoraTrainingWorker) -> torch.Tensor:
  return torch.cat([p.grad.flatten().float() for name, p in worker.peft_model.named_parameters() if "lora_B" in name and p.grad is not None])


@unittest.skipUnless(torch.cuda.is_available(), "autocast applies on CUDA")
class LoraAutocastTest(unittest.TestCase):
  def run_step(self, autocast: bool):
    worker = bf16_worker()
    with (
      patch.object(LoraTrainingWorker, "save_adapter"),
      patch.object(LoraTrainingWorker, "half_precision", lambda self: torch.bfloat16 if autocast else None),
    ):
      worker.create_adapter("job", LoraConfig(rank=4, seed=1, lora_dropout=0.0))
      row = list(range(3, 40))
      datum = Datum(
        model_input=row[:-1], loss_fn_inputs={"target_tokens": TensorData(data=row[1:]), "weights": TensorData(data=[1.0] * (len(row) - 1))}
      )
      result = worker.forward_backward([datum], "cross_entropy", model_id="job")
      sample = worker.generate([3, 4, 5], max_tokens=2, model_id="job")
    return worker, result, sample

  def test_matches_the_fp32_lora_path(self):
    fast, fast_result, fast_sample = self.run_step(autocast=True)
    exact, exact_result, _ = self.run_step(autocast=False)
    self.assertTrue(all(not m.cast_input_dtype_enabled for m in fast.peft_model.modules() if hasattr(m, "cast_input_dtype_enabled")))
    self.assertAlmostEqual(
      fast_result["metrics"]["loss:sum"], exact_result["metrics"]["loss:sum"], delta=0.05 * abs(exact_result["metrics"]["loss:sum"])
    )
    cosine = torch.nn.functional.cosine_similarity(lora_b_grads(fast), lora_b_grads(exact), dim=0)
    self.assertGreater(cosine.item(), 0.99)
    self.assertTrue(all(p.dtype == torch.float32 for name, p in fast.peft_model.named_parameters() if "lora_" in name))
    self.assertTrue(fast_sample)


if __name__ == "__main__":
  unittest.main()
