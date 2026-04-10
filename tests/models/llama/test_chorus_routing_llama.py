# coding=utf-8

import unittest

import torch

from transformers.models.llama.modeling_llama import _compute_chorus_routed_injection


class LlamaChorusRoutingTest(unittest.TestCase):
    def test_routed_helper_uses_explicit_gates(self):
        query_states = torch.ones(2, 1, 1, 1, dtype=torch.float32)
        cached_key_value = (
            torch.zeros(2, 1, 2, 1, dtype=torch.float32),
            torch.tensor([[[[1.0], [3.0]]], [[[10.0], [20.0]]]], dtype=torch.float32),
        )
        kwargs = {
            "seq_route_probs": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
            "ch_route_probs": torch.tensor(
                [
                    [[0.2, 0.8], [0.25, 0.75]],
                    [[0.3, 0.7], [0.1, 0.9]],
                ],
                dtype=torch.float32,
            ),
            "memory_source_index": torch.tensor([[1, 1], [0, 0]], dtype=torch.long),
            "memory_channel_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            "disable_v_norm": True,
        }

        routed_output = _compute_chorus_routed_injection(
            query_states=query_states,
            cached_key_value=cached_key_value,
            num_key_value_groups=1,
            head_dim=1,
            attention_dropout=0.0,
            training=False,
            kwargs=kwargs,
        )

        expected = torch.tensor([[[[17.5]]], [[[2.4]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(routed_output, expected, atol=1e-5))

    def test_routed_helper_returns_none_without_metadata(self):
        routed_output = _compute_chorus_routed_injection(
            query_states=torch.ones(1, 1, 1, 1, dtype=torch.float32),
            cached_key_value=(
                torch.zeros(1, 1, 1, 1, dtype=torch.float32),
                torch.zeros(1, 1, 1, 1, dtype=torch.float32),
            ),
            num_key_value_groups=1,
            head_dim=1,
            attention_dropout=0.0,
            training=False,
            kwargs={},
        )
        self.assertIsNone(routed_output)

    def test_routed_helper_rejects_misaligned_metadata(self):
        with self.assertRaisesRegex(ValueError, "memory_channel_index"):
            _compute_chorus_routed_injection(
                query_states=torch.ones(2, 1, 1, 1, dtype=torch.float32),
                cached_key_value=(
                    torch.zeros(2, 1, 2, 1, dtype=torch.float32),
                    torch.zeros(2, 1, 2, 1, dtype=torch.float32),
                ),
                num_key_value_groups=1,
                head_dim=1,
                attention_dropout=0.0,
                training=False,
                kwargs={
                    "seq_route_probs": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
                    "ch_route_probs": torch.full((2, 2, 2), 0.5, dtype=torch.float32),
                    "memory_source_index": torch.tensor([[1, 1], [0, 0]], dtype=torch.long),
                    "memory_channel_index": torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
                    "disable_v_norm": True,
                },
            )


if __name__ == "__main__":
    unittest.main()
