import math
from typing import Dict, Optional

import torch
from torch import nn


def compute_block_sparse_injection(
    query_states: torch.Tensor,
    cached_keys: torch.Tensor,
    cached_values: torch.Tensor,
    block_sparse_metadata: Dict[str, torch.Tensor],
    *,
    head_dim: int,
    attention_dropout: float,
    training: bool,
    disable_v_norm: bool = False,
    original_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz, num_heads, q_len, hidden_dim = query_states.shape
    query_token_indices = block_sparse_metadata["query_token_indices"].to(query_states.device)
    query_group_offsets = block_sparse_metadata["query_group_offsets"].to(query_states.device)

    if query_token_indices.numel() == 0 or query_group_offsets.numel() <= 1:
        return query_states.new_zeros((bsz, num_heads, q_len, hidden_dim))

    query_flat = query_states.transpose(1, 2).contiguous().view(bsz * q_len, num_heads, hidden_dim)
    grouped_queries = query_flat.index_select(0, query_token_indices)
    detail_output = grouped_queries.new_zeros(grouped_queries.shape)

    pair_group_ids = block_sparse_metadata["pair_query_group_ids"].to(query_states.device)
    pair_seq_ids = block_sparse_metadata["pair_seq_ids"].to(query_states.device)
    pair_kv_starts = block_sparse_metadata["pair_kv_starts"].to(query_states.device)
    pair_kv_ends = block_sparse_metadata["pair_kv_ends"].to(query_states.device)
    pair_detail_weights = block_sparse_metadata["pair_detail_weights"].to(query_states.device, dtype=grouped_queries.dtype)

    for pair_idx in range(pair_group_ids.numel()):
        group_id = pair_group_ids[pair_idx].item()
        group_start = query_group_offsets[group_id].item()
        group_end = query_group_offsets[group_id + 1].item()
        if group_end <= group_start:
            continue

        seq_id = pair_seq_ids[pair_idx].item()
        kv_start = pair_kv_starts[pair_idx].item()
        kv_end = pair_kv_ends[pair_idx].item()
        if kv_end <= kv_start:
            continue

        query_group = grouped_queries[group_start:group_end].permute(1, 0, 2)
        key_span = cached_keys[seq_id, :, kv_start:kv_end, :]
        value_span = cached_values[seq_id, :, kv_start:kv_end, :]

        attn_scores = torch.matmul(query_group, key_span.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_probs = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_group.dtype)
        attn_probs = nn.functional.dropout(attn_probs, p=attention_dropout, training=training)
        pair_ctx = torch.matmul(attn_probs, value_span)

        if not disable_v_norm:
            value_norm = torch.norm(value_span, p=2, dim=-1, keepdim=True)
            weighted_value_norm = torch.matmul(attn_probs, value_norm)
            pair_ctx = pair_ctx / (weighted_value_norm + 1e-6)

        detail_output[group_start:group_end] += pair_detail_weights[pair_idx] * pair_ctx.permute(1, 0, 2)

    doc_alpha = block_sparse_metadata["doc_alpha"].to(query_states.device, dtype=grouped_queries.dtype)
    if doc_alpha.numel() > 0:
        doc_values = _doc_memory_values(
            cached_values=cached_values,
            block_sparse_metadata=block_sparse_metadata,
            original_attention_mask=original_attention_mask,
        ).to(grouped_queries.dtype)
        group_doc_ctx = torch.einsum("gb,bhd->ghd", doc_alpha, doc_values)
        summary_output = grouped_queries.new_zeros(grouped_queries.shape)
        for group_id in range(query_group_offsets.numel() - 1):
            group_start = query_group_offsets[group_id].item()
            group_end = query_group_offsets[group_id + 1].item()
            if group_end <= group_start:
                continue
            summary_output[group_start:group_end] = group_doc_ctx[group_id].unsqueeze(0)
    else:
        summary_output = grouped_queries.new_zeros(grouped_queries.shape)

    total_output = summary_output + float(block_sparse_metadata.get("detail_lambda", 1.0)) * detail_output

    scatter_flat = grouped_queries.new_zeros((bsz * q_len, num_heads, hidden_dim))
    scatter_flat.index_add_(0, query_token_indices, total_output)
    scatter_output = scatter_flat.view(bsz, q_len, num_heads, hidden_dim).transpose(1, 2).contiguous()

    routing_query_mask = block_sparse_metadata.get("routing_query_mask")
    if routing_query_mask is not None:
        routing_query_mask = routing_query_mask.to(query_states.device)
        route_mask = routing_query_mask.unsqueeze(1).unsqueeze(-1).bool()
        scatter_output = scatter_output.masked_fill(route_mask, 0.0)
    return scatter_output


def _doc_memory_values(
    cached_values: torch.Tensor,
    block_sparse_metadata: Dict[str, torch.Tensor],
    original_attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    mode = block_sparse_metadata.get("reference_doc_memory_mode", "chorus_token")
    if mode == "chorus_token":
        doc_memory_positions = block_sparse_metadata.get("doc_memory_positions")
        if doc_memory_positions is None:
            raise ValueError("chorus_token doc memory requires doc_memory_positions")
        doc_memory_positions = doc_memory_positions.to(cached_values.device)
        batch_indices = torch.arange(cached_values.size(0), device=cached_values.device)
        return cached_values[batch_indices, :, doc_memory_positions, :]

    if original_attention_mask is None:
        raise ValueError("pooled_doc doc memory requires original_attention_mask")
    value_mask = original_attention_mask[:, : cached_values.shape[-2]].to(cached_values.dtype)
    denom = value_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (cached_values * value_mask[:, None, :, None]).sum(dim=-2) / denom[:, None]
