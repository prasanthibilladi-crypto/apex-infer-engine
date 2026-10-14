import math
import torch
import torch.nn.functional as F
from typing import List
from .block_allocator import BlockAllocator, BlockTable

class PagedAttentionEngine:
    """
    Computes Scaled Dot-Product Multi-Head Attention directly over non-contiguous 
    paged memory blocks using Block Tables (replaces standard contiguous attention).
    """
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.scale = 1.0 / math.sqrt(allocator.head_dim)

    def forward_decode(
        self,
        layer_idx: int,
        query: torch.Tensor,
        block_tables: List[BlockTable],
    ) -> torch.Tensor:
        """
        Computes attention for the decoding step (1 new token per sequence).
        Args:
            layer_idx: current transformer layer
            query: Shape [batch_size, 1, num_heads, head_dim]
            block_tables: list of BlockTable objects for each sequence in the batch
        Returns:
            output: Shape [batch_size, 1, num_heads, head_dim]
        """
        batch_size = query.shape[0]
        num_heads = self.allocator.num_heads
        head_dim = self.allocator.head_dim
        block_size = self.allocator.block_size
        outputs = []

        for b in range(batch_size):
            q_b = query[b, 0]  # Shape: [num_heads, head_dim]
            table = block_tables[b]
            seq_len = table.num_tokens

            if seq_len == 0:
                outputs.append(torch.zeros_like(query[b]))
                continue

            # 1. Fetch physical blocks for this sequence from preallocated pool
            phys_ids = table.physical_block_ids
            # Keys shape: [num_blocks, block_size, num_heads, head_dim]
            keys_blocks = self.allocator.kv_cache_pool[layer_idx, phys_ids, 0]
            values_blocks = self.allocator.kv_cache_pool[layer_idx, phys_ids, 1]

            # Flatten across blocks to reconstruct logical sequence: [num_blocks * block_size, num_heads, head_dim]
            keys_flat = keys_blocks.view(-1, num_heads, head_dim)[:seq_len]
            values_flat = values_blocks.view(-1, num_heads, head_dim)[:seq_len]

            # Transpose to [num_heads, seq_len, head_dim] for standard multi-head attention math
            k = keys_flat.permute(1, 0, 2)
            v = values_flat.permute(1, 0, 2)
            q = q_b.unsqueeze(1)  # [num_heads, 1, head_dim]

            # 2. Scaled Dot-Product Attention: (Q * K^T) * scale
            # q: [H, 1, D], k.transpose: [H, D, L] -> scores: [H, 1, L]
            scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)

            # 3. Weighted Value aggregation: attn_weights * V
            # attn_weights: [H, 1, L], v: [H, L, D] -> out: [H, 1, D]
            context = torch.bmm(attn_weights, v)

            # Transpose back to [1, num_heads, head_dim]
            outputs.append(context.permute(1, 0, 2))

        return torch.stack(outputs, dim=0)
