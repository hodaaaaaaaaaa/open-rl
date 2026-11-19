"""Vendored SimpleFSDP on a two-process CPU gloo mesh, against a one-process run on the full batch."""

import os
import pickle
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import LlamaConfig, LlamaForCausalLM

from training import simple_fsdp

WORLD = 2
ROWS, SEQ, VOCAB = 4, 12, 64


def tiny_model() -> LlamaForCausalLM:
  torch.manual_seed(0)
  config = LlamaConfig(
    vocab_size=VOCAB,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    max_position_embeddings=64,
    tie_word_embeddings=False,
  )
  return LlamaForCausalLM(config)


def full_batch() -> torch.Tensor:
  return torch.randint(0, VOCAB, (WORLD * ROWS, SEQ), generator=torch.Generator().manual_seed(1))


def sum_loss(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
  logits = model(input_ids=ids).logits
  return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1), reduction="sum")


def full(value: torch.Tensor) -> torch.Tensor:
  return value.full_tensor() if isinstance(value, DTensor) else value


def max_diff(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> float:
  return max(float((actual[name].detach() - expected[name].detach()).abs().max()) for name in expected)


def run_checks(rank: int, world_size: int, port: int, out_dir: str) -> None:
  os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
  dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=120))
  try:
    reference = tiny_model()
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-2, weight_decay=0.0)
    sum_loss(reference, full_batch()).backward()
    reference_grads = {name: param.grad.clone() for name, param in reference.named_parameters()}
    reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    reference_optimizer.step()

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
    model = simple_fsdp.data_parallel(tiny_model(), device_mesh=mesh, mode="fully_shard")
    results: dict = {}
    # The parametrization hands the forward pass full weights; the stored parameters are this rank's shards.
    with simple_fsdp.disable_active_parametrization():
      stored = dict(model.named_parameters())
    results["sharded"] = all(isinstance(param, DTensor) for param in stored.values())
    results["local_rows"] = (stored["lm_head.weight"].to_local().shape[0], VOCAB // world_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    sum_loss(model, full_batch()[rank * ROWS : (rank + 1) * ROWS]).backward()
    results["grad_diff"] = max_diff({name: full(param.grad) for name, param in model.named_parameters()}, reference_grads)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    results["clip_norm"] = (float(full(norm)), float(reference_norm))
    optimizer.step()
    results["weight_diff"] = max_diff({name: full(param) for name, param in stored.items()}, dict(reference.named_parameters()))

    with open(os.path.join(out_dir, f"rank{rank}.pkl"), "wb") as f:
      pickle.dump(results, f)
  finally:
    dist.destroy_process_group()


class SimpleFSDPTest(unittest.TestCase):
  results: list[dict] = []

  @classmethod
  def setUpClass(cls) -> None:
    # One spawn runs every check; each test asserts on the results of both ranks.
    with tempfile.TemporaryDirectory() as out_dir:
      mp.spawn(run_checks, args=(WORLD, 29562, out_dir), nprocs=WORLD, join=True)
      cls.results = []
      for rank in range(WORLD):
        with open(os.path.join(out_dir, f"rank{rank}.pkl"), "rb") as f:
          cls.results.append(pickle.load(f))

  def test_every_parameter_is_sharded(self) -> None:
    for results in self.results:
      self.assertTrue(results["sharded"])
      local_rows, expected = results["local_rows"]
      self.assertEqual(local_rows, expected)

  def test_sum_loss_gradients_match_reference(self) -> None:
    for results in self.results:
      self.assertLess(results["grad_diff"], 1e-5)

  def test_clip_norm_is_global(self) -> None:
    for results in self.results:
      sharded, reference = results["clip_norm"]
      self.assertAlmostEqual(sharded, reference, places=3)

  def test_adamw_step_matches_reference(self) -> None:
    for results in self.results:
      self.assertLess(results["weight_diff"], 1e-4)


if __name__ == "__main__":
  unittest.main()
