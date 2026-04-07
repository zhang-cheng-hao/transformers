import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from .modeling_flash_attention_utils import flash_attn_varlen_func


def compute_block_sparse_injection(
    query_states: torch.Tensor,
    cached_keys: torch.Tensor,
    cached_values: torch.Tensor,
    block_sparse_metadata: Dict[str, torch.Tensor],
    summary_query_states: torch.Tensor,
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
    num_groups = query_group_offsets.numel() - 1

    group_starts = query_group_offsets[:-1]
    group_lengths = query_group_offsets[1:] - query_group_offsets[:-1]
    max_group_length = int(group_lengths.max().item())
    group_rel = torch.arange(max_group_length, device=query_states.device)
    group_mask = group_rel.unsqueeze(0) < group_lengths.unsqueeze(1)

    pair_metadata = _pair_metadata(
        block_sparse_metadata=block_sparse_metadata,
        device=query_states.device,
    )
    route_prob = _route_prob(block_sparse_metadata, grouped_queries)
    d2d_read = _d2d_read(
        grouped_queries=grouped_queries,
        group_mask=group_mask,
        cached_values=cached_values,
        block_sparse_metadata=block_sparse_metadata,
        original_attention_mask=original_attention_mask,
    )
    b2b_read, t2t_read = _pair_reads(
        grouped_queries=grouped_queries,
        group_mask=group_mask,
        group_starts=group_starts,
        group_lengths=group_lengths,
        num_groups=num_groups,
        max_group_length=max_group_length,
        cached_keys=cached_keys,
        cached_values=cached_values,
        pair_metadata=pair_metadata,
        route_prob=route_prob,
        head_dim=head_dim,
        attention_dropout=attention_dropout,
        training=training,
        disable_v_norm=disable_v_norm,
        block_sparse_metadata=block_sparse_metadata,
    )

    total_output = (
        d2d_read
        + b2b_read
        + float(block_sparse_metadata.get("detail_lambda", 1.0)) * t2t_read
    )

    scatter_flat = grouped_queries.new_zeros((bsz * q_len, num_heads, hidden_dim))
    scatter_flat.index_add_(0, query_token_indices, total_output)
    scatter_output = scatter_flat.view(bsz, q_len, num_heads, hidden_dim).transpose(1, 2).contiguous()

    routing_query_mask = block_sparse_metadata.get("routing_query_mask")
    if routing_query_mask is not None:
        routing_query_mask = routing_query_mask.to(query_states.device)
        route_mask = routing_query_mask.unsqueeze(1).unsqueeze(-1).bool()
        scatter_output = scatter_output.masked_fill(route_mask, 0.0)
    return scatter_output


def _d2d_read(
    grouped_queries: torch.Tensor,
    group_mask: torch.Tensor,
    cached_values: torch.Tensor,
    block_sparse_metadata: Dict[str, torch.Tensor],
    original_attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    d2d_prob = block_sparse_metadata["d2d_prob"].to(grouped_queries.device, dtype=grouped_queries.dtype)
    if d2d_prob.numel() == 0:
        return grouped_queries.new_zeros(grouped_queries.shape)

    doc_values = _doc_memory_values(
        cached_values=cached_values,
        block_sparse_metadata=block_sparse_metadata,
        original_attention_mask=original_attention_mask,
    ).to(grouped_queries.dtype)
    group_doc_ctx = torch.einsum("gb,bhd->ghd", d2d_prob, doc_values)
    return group_doc_ctx[:, None, :, :].expand(-1, group_mask.size(1), -1, -1)[group_mask]


def _b2b_read(
    grouped_queries: torch.Tensor,
    group_mask: torch.Tensor,
    group_block_ctx: torch.Tensor,
    max_group_length: int,
) -> torch.Tensor:
    return group_block_ctx[:, None, :, :].expand(-1, max_group_length, -1, -1)[group_mask]


def _t2t_read(
    grouped_queries: torch.Tensor,
    group_mask: torch.Tensor,
    group_detail: torch.Tensor,
) -> torch.Tensor:
    return group_detail


def _route_prob(
    block_sparse_metadata: Dict[str, torch.Tensor],
    grouped_queries: torch.Tensor,
) -> torch.Tensor:
    return block_sparse_metadata["route_prob"].to(grouped_queries.device, dtype=grouped_queries.dtype)


def _pair_metadata(
    block_sparse_metadata: Dict[str, torch.Tensor],
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    pair_group_ids = block_sparse_metadata["pair_query_group_ids"].to(device)
    if pair_group_ids.numel() == 0:
        return None

    pair_seq_ids = block_sparse_metadata["pair_seq_ids"].to(device)
    pair_kv_starts = block_sparse_metadata["pair_kv_starts"].to(device)
    pair_kv_ends = block_sparse_metadata["pair_kv_ends"].to(device)
    return {
        "pair_group_ids": pair_group_ids,
        "pair_seq_ids": pair_seq_ids,
        "pair_kv_starts": pair_kv_starts,
        "pair_kv_ends": pair_kv_ends,
    }


def _pair_reads(
    grouped_queries: torch.Tensor,
    group_mask: torch.Tensor,
    group_starts: torch.Tensor,
    group_lengths: torch.Tensor,
    num_groups: int,
    max_group_length: int,
    cached_keys: torch.Tensor,
    cached_values: torch.Tensor,
    pair_metadata: Optional[Dict[str, torch.Tensor]],
    route_prob: torch.Tensor,
    summary_query_states: torch.Tensor,
    *,
    head_dim: int,
    attention_dropout: float,
    training: bool,
    disable_v_norm: bool,
    block_sparse_metadata: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pair_metadata is None or route_prob.numel() == 0:
        zeros = grouped_queries.new_zeros(grouped_queries.shape)
        return zeros, zeros

    # Stream sparse pairs in chunks so peak temporary KV/ctx tensors scale with chunk size, not total pairs.
    pair_chunk_size = _pair_chunk_size(block_sparse_metadata, pair_metadata["pair_group_ids"].numel())
    cached_keys_seq_first = cached_keys.permute(0, 2, 1, 3).contiguous()
    cached_values_seq_first = cached_values.permute(0, 2, 1, 3).contiguous()
    group_block_ctx = grouped_queries.new_zeros((num_groups, grouped_queries.size(1), grouped_queries.size(2)))
    detail_output = grouped_queries.new_zeros(grouped_queries.shape)
    use_flash_varlen = _can_use_flash_varlen(query_states=grouped_queries, cached_keys=cached_keys_seq_first)
    grouped_queries_padded = None if use_flash_varlen else _build_grouped_queries_padded(
        grouped_queries=grouped_queries,
        group_mask=group_mask,
        group_starts=group_starts,
        max_group_length=max_group_length,
    )

    total_pairs = pair_metadata["pair_group_ids"].numel()
    for chunk_start in range(0, total_pairs, pair_chunk_size):
        chunk_end = min(chunk_start + pair_chunk_size, total_pairs)
        pair_group_ids = pair_metadata["pair_group_ids"][chunk_start:chunk_end]
        pair_seq_ids = pair_metadata["pair_seq_ids"][chunk_start:chunk_end]
        pair_kv_starts = pair_metadata["pair_kv_starts"][chunk_start:chunk_end]
        pair_kv_ends = pair_metadata["pair_kv_ends"][chunk_start:chunk_end]
        chunk_route_prob = route_prob[chunk_start:chunk_end]

        pair_q_lens = group_lengths.index_select(0, pair_group_ids)
        if use_flash_varlen:
            flat_queries, flat_query_positions, cu_seqlens_q, max_seqlen_q = _gather_flat_queries_chunk(
                grouped_queries=grouped_queries,
                group_starts=group_starts,
                pair_group_ids=pair_group_ids,
                pair_q_lens=pair_q_lens,
            )
            packed_kv = _gather_flat_kv_chunk(
                cached_keys_seq_first=cached_keys_seq_first,
                cached_values_seq_first=cached_values_seq_first,
                pair_seq_ids=pair_seq_ids,
                pair_kv_starts=pair_kv_starts,
                pair_kv_ends=pair_kv_ends,
            )
            flat_summary_queries, cu_seqlens_summary_q = _build_flat_summary_queries(
                summary_query_states=summary_query_states,
                num_pairs=pair_group_ids.numel(),
                dtype=grouped_queries.dtype,
                device=grouped_queries.device,
            )
            flat_summary_ctx = _flash_varlen_t2t_read(
                flat_queries=flat_summary_queries,
                cu_seqlens_q=cu_seqlens_summary_q,
                max_seqlen_q=1,
                flat_keys=packed_kv["flat_keys"],
                flat_values=packed_kv["flat_values"],
                cu_seqlens_k=packed_kv["cu_seqlens_k"],
                max_seqlen_k=packed_kv["max_seqlen_k"],
                attention_dropout=attention_dropout,
                training=training,
                disable_v_norm=disable_v_norm,
            )
            weighted_block_summary = _summary_query_read(
                summary_ctx=flat_summary_ctx,
                route_prob=chunk_route_prob,
                dtype=grouped_queries.dtype,
            )
            group_block_ctx.index_add_(0, pair_group_ids, weighted_block_summary)
            flat_ctx = _flash_varlen_t2t_read(
                flat_queries=flat_queries,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                flat_keys=packed_kv["flat_keys"],
                flat_values=packed_kv["flat_values"],
                cu_seqlens_k=packed_kv["cu_seqlens_k"],
                max_seqlen_k=packed_kv["max_seqlen_k"],
                attention_dropout=attention_dropout,
                training=training,
                disable_v_norm=disable_v_norm,
            )
            token_route_prob = torch.repeat_interleave(chunk_route_prob, pair_q_lens).to(flat_ctx.dtype)
            detail_output.index_add_(0, flat_query_positions, flat_ctx * token_route_prob[:, None, None])
        else:
            pair_cache = _gather_pair_cache_chunk(
                cached_keys_seq_first=cached_keys_seq_first,
                cached_values_seq_first=cached_values_seq_first,
                pair_group_ids=pair_group_ids,
                pair_seq_ids=pair_seq_ids,
                pair_kv_starts=pair_kv_starts,
                pair_kv_ends=pair_kv_ends,
            )
            pair_summary_queries = _build_pair_summary_queries(
                summary_query_states=summary_query_states,
                num_pairs=pair_group_ids.numel(),
                dtype=grouped_queries.dtype,
                device=grouped_queries.device,
            )
            pair_summary_ctx = _pair_attention_dense(
                pair_queries=pair_summary_queries,
                pair_keys=pair_cache["pair_keys"],
                pair_values=pair_cache["pair_values"],
                kv_mask=pair_cache["kv_mask"],
                head_dim=head_dim,
                attention_dropout=attention_dropout,
                training=training,
                disable_v_norm=disable_v_norm,
            ).squeeze(2)
            weighted_block_summary = _summary_query_read(
                summary_ctx=pair_summary_ctx,
                route_prob=chunk_route_prob,
                dtype=grouped_queries.dtype,
            )
            group_block_ctx.index_add_(0, pair_group_ids, weighted_block_summary)
            pair_queries = grouped_queries_padded.index_select(0, pair_group_ids).permute(0, 2, 1, 3).contiguous()
            pair_ctx = _pair_attention_dense(
                pair_queries=pair_queries,
                pair_keys=pair_cache["pair_keys"],
                pair_values=pair_cache["pair_values"],
                kv_mask=pair_cache["kv_mask"],
                head_dim=head_dim,
                attention_dropout=attention_dropout,
                training=training,
                disable_v_norm=disable_v_norm,
            )

            weighted_pair_ctx = pair_ctx * chunk_route_prob[:, None, None, None]
            weighted_pair_ctx = weighted_pair_ctx.permute(0, 2, 1, 3).contiguous()
            group_detail = grouped_queries.new_zeros((num_groups, max_group_length, grouped_queries.size(1), grouped_queries.size(2)))
            group_detail.index_add_(0, pair_group_ids, weighted_pair_ctx)
            detail_output = detail_output + group_detail[group_mask]

    b2b_read = _b2b_read(
        grouped_queries=grouped_queries,
        group_mask=group_mask,
        group_block_ctx=group_block_ctx,
        max_group_length=max_group_length,
    )
    t2t_read = _t2t_read(
        grouped_queries=grouped_queries,
        group_mask=group_mask,
        group_detail=detail_output,
    )
    return b2b_read, t2t_read


def _pair_chunk_size(
    block_sparse_metadata: Dict[str, torch.Tensor],
    total_pairs: int,
) -> int:
    chunk_size = block_sparse_metadata.get("pair_chunk_size")
    if isinstance(chunk_size, torch.Tensor):
        chunk_size = int(chunk_size.reshape(-1)[0].item()) if chunk_size.numel() > 0 else 0
    elif chunk_size is not None:
        chunk_size = int(chunk_size)
    else:
        chunk_size = 1024
    chunk_size = max(chunk_size, 1)
    return min(chunk_size, max(total_pairs, 1))


def _gather_pair_cache_chunk(
    cached_keys_seq_first: torch.Tensor,
    cached_values_seq_first: torch.Tensor,
    pair_group_ids: torch.Tensor,
    pair_seq_ids: torch.Tensor,
    pair_kv_starts: torch.Tensor,
    pair_kv_ends: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    kv_lengths = pair_kv_ends - pair_kv_starts
    max_kv_length = int(kv_lengths.max().item())
    kv_rel = torch.arange(max_kv_length, device=pair_seq_ids.device)
    kv_mask = kv_rel.unsqueeze(0) < kv_lengths.unsqueeze(1)
    kv_positions = pair_kv_starts.unsqueeze(1) + kv_rel.unsqueeze(0)
    safe_kv_positions = kv_positions.masked_fill(~kv_mask, 0)

    pair_keys = cached_keys_seq_first[pair_seq_ids[:, None], safe_kv_positions].permute(0, 2, 1, 3).contiguous()
    pair_values = cached_values_seq_first[pair_seq_ids[:, None], safe_kv_positions].permute(0, 2, 1, 3).contiguous()
    return {
        "pair_group_ids": pair_group_ids,
        "pair_keys": pair_keys,
        "pair_values": pair_values,
        "kv_lengths": kv_lengths,
        "kv_mask": kv_mask,
    }


def _pair_attention_dense(
    pair_queries: torch.Tensor,
    pair_keys: torch.Tensor,
    pair_values: torch.Tensor,
    kv_mask: torch.Tensor,
    *,
    head_dim: int,
    attention_dropout: float,
    training: bool,
    disable_v_norm: bool,
) -> torch.Tensor:
    dropout_p = attention_dropout if training else 0.0
    if _can_use_sdpa(attention_dropout=attention_dropout, training=training):
        # Prefer fused SDPA when safe so we avoid explicitly materializing attention scores/probabilities.
        pair_ctx = nn.functional.scaled_dot_product_attention(
            pair_queries,
            pair_keys,
            pair_values,
            attn_mask=kv_mask[:, None, None, :],
            dropout_p=dropout_p,
            is_causal=False,
        )
        if disable_v_norm:
            return pair_ctx
        value_norm = torch.norm(pair_values, p=2, dim=-1, keepdim=True)
        weighted_value_norm = nn.functional.scaled_dot_product_attention(
            pair_queries,
            pair_keys,
            value_norm,
            attn_mask=kv_mask[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return pair_ctx / (weighted_value_norm + 1e-6)

    attn_scores = torch.matmul(pair_queries, pair_keys.transpose(-2, -1)) / math.sqrt(head_dim)
    min_dtype = torch.finfo(attn_scores.dtype).min
    attn_scores = attn_scores.masked_fill(~kv_mask[:, None, None, :], min_dtype)
    attn_probs = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(pair_queries.dtype)
    attn_probs = nn.functional.dropout(attn_probs, p=attention_dropout, training=training)
    pair_ctx = torch.matmul(attn_probs, pair_values)

    if disable_v_norm:
        return pair_ctx

    value_norm = torch.norm(pair_values, p=2, dim=-1, keepdim=True)
    weighted_value_norm = torch.matmul(attn_probs, value_norm)
    return pair_ctx / (weighted_value_norm + 1e-6)


def _build_grouped_queries_padded(
    grouped_queries: torch.Tensor,
    group_mask: torch.Tensor,
    group_starts: torch.Tensor,
    max_group_length: int,
) -> torch.Tensor:
    group_rel = torch.arange(max_group_length, device=grouped_queries.device)
    group_positions = group_starts.unsqueeze(1) + group_rel.unsqueeze(0)
    safe_group_positions = group_positions.masked_fill(~group_mask, 0)
    grouped_queries_padded = grouped_queries.index_select(0, safe_group_positions.reshape(-1)).view(
        group_mask.size(0), max_group_length, grouped_queries.size(1), grouped_queries.size(2)
    )
    return grouped_queries_padded * group_mask[:, :, None, None].to(grouped_queries.dtype)


def _gather_flat_queries_chunk(
    grouped_queries: torch.Tensor,
    group_starts: torch.Tensor,
    pair_group_ids: torch.Tensor,
    pair_q_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    max_q_len = int(pair_q_lens.max().item())
    q_rel = torch.arange(max_q_len, device=grouped_queries.device)
    pair_q_starts = group_starts.index_select(0, pair_group_ids)
    q_mask = q_rel.unsqueeze(0) < pair_q_lens.unsqueeze(1)
    q_positions = pair_q_starts.unsqueeze(1) + q_rel.unsqueeze(0)
    flat_query_positions = q_positions[q_mask]
    flat_queries = grouped_queries.index_select(0, flat_query_positions)
    cu_seqlens_q = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=grouped_queries.device),
            torch.cumsum(pair_q_lens, dim=0, dtype=torch.int32),
        ]
    )
    return flat_queries, flat_query_positions, cu_seqlens_q, max_q_len


def _gather_flat_kv_chunk(
    cached_keys_seq_first: torch.Tensor,
    cached_values_seq_first: torch.Tensor,
    pair_seq_ids: torch.Tensor,
    pair_kv_starts: torch.Tensor,
    pair_kv_ends: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    kv_lengths = (pair_kv_ends - pair_kv_starts).to(torch.int32)
    max_kv_length = int(kv_lengths.max().item())
    kv_rel = torch.arange(max_kv_length, device=pair_seq_ids.device)
    kv_mask = kv_rel.unsqueeze(0) < kv_lengths.unsqueeze(1)
    kv_positions = pair_kv_starts.unsqueeze(1) + kv_rel.unsqueeze(0)
    flat_cache_positions = (pair_seq_ids.unsqueeze(1) * cached_keys_seq_first.size(1) + kv_positions)[kv_mask]

    flat_cached_keys = cached_keys_seq_first.reshape(-1, cached_keys_seq_first.size(2), cached_keys_seq_first.size(3))
    flat_cached_values = cached_values_seq_first.reshape(-1, cached_values_seq_first.size(2), cached_values_seq_first.size(3))
    flat_keys = flat_cached_keys.index_select(0, flat_cache_positions)
    flat_values = flat_cached_values.index_select(0, flat_cache_positions)
    cu_seqlens_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=pair_seq_ids.device),
            torch.cumsum(kv_lengths, dim=0, dtype=torch.int32),
        ]
    )
    return {
        "flat_keys": flat_keys,
        "flat_values": flat_values,
        "kv_lengths": kv_lengths,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_k": max_kv_length,
    }


def _build_pair_summary_queries(
    summary_query_states: torch.Tensor,
    num_pairs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    summary_query = summary_query_states.to(device=device, dtype=dtype)
    return summary_query.unsqueeze(0).unsqueeze(2).expand(num_pairs, -1, 1, -1).contiguous()


def _build_flat_summary_queries(
    summary_query_states: torch.Tensor,
    num_pairs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    summary_query = summary_query_states.to(device=device, dtype=dtype)
    flat_queries = summary_query.unsqueeze(0).expand(num_pairs, -1, -1).contiguous()
    cu_seqlens_q = torch.arange(num_pairs + 1, device=device, dtype=torch.int32)
    return flat_queries, cu_seqlens_q


def _summary_query_read(
    summary_ctx: torch.Tensor,
    route_prob: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return summary_ctx.to(dtype) * route_prob[:, None, None].to(dtype)


def _flash_varlen_t2t_read(
    flat_queries: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    flat_keys: torch.Tensor,
    flat_values: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    *,
    attention_dropout: float,
    training: bool,
    disable_v_norm: bool,
) -> torch.Tensor:
    flat_ctx = flash_attn_varlen_func(
        flat_queries,
        flat_keys,
        flat_values,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=attention_dropout if training else 0.0,
        softmax_scale=None,
        causal=False,
    )
    if disable_v_norm:
        return flat_ctx

    flat_value_norm = torch.norm(flat_values, p=2, dim=-1, keepdim=True)
    weighted_value_norm = flash_attn_varlen_func(
        flat_queries,
        flat_keys,
        flat_value_norm,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
    )
    return flat_ctx / (weighted_value_norm + 1e-6)


def _can_use_sdpa(
    *,
    attention_dropout: float,
    training: bool,
) -> bool:
    if not hasattr(nn.functional, "scaled_dot_product_attention"):
        return False
    if training and attention_dropout > 0:
        return False
    return True


def _can_use_flash_varlen(
    *,
    query_states: torch.Tensor,
    cached_keys: torch.Tensor,
) -> bool:
    if not is_flash_attn_2_available():
        return False
    if query_states.device.type != "cuda" or cached_keys.device.type != "cuda":
        return False
    if query_states.dtype not in (torch.float16, torch.bfloat16):
        return False
    if cached_keys.dtype != query_states.dtype:
        return False
    return True


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
