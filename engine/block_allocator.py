import torch
from typing import List, Dict, Optional

class PhysicalBlock:
    """Represents a single allocated physical memory block."""
    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0

class BlockTable:
    """Maps logical token sequence blocks to physical block IDs for a specific request."""
    def __init__(self, request_id: str, block_size: int = 16):
        self.request_id = request_id
        self.block_size = block_size
        self.physical_block_ids: List[int] = []
        self.num_tokens = 0

    def logical_to_physical(self, token_idx: int) -> (int, int):
        """
        Translates logical token index into (physical_block_id, block_offset).
        Mimics OS virtual-to-physical memory translation.
        """
        logical_block_idx = token_idx // self.block_size
        offset = token_idx % self.block_size
        if logical_block_idx >= len(self.physical_block_ids):
            raise IndexError(f"Logical block {logical_block_idx} not allocated yet.")
        physical_block_id = self.physical_block_ids[logical_block_idx]
        return physical_block_id, offset

class BlockAllocator:
    """
    Virtual Memory Paged KV-Cache Manager (OS-style Virtual Paging).
    Manages non-contiguous physical GPU/CPU memory blocks to eliminate KV-cache fragmentation.
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        total_num_blocks: int = 512,
        block_size: int = 16,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_num_blocks = total_num_blocks
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        # Preallocated Physical KV Memory Pool:
        # Shape: [num_layers, total_num_blocks, 2 (Key & Value), block_size, num_heads, head_dim]
        self.kv_cache_pool = torch.zeros(
            (num_layers, total_num_blocks, 2, block_size, num_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        # Free List of Physical Blocks
        self.free_block_ids: List[int] = list(range(total_num_blocks))
        self.allocated_blocks: Dict[int, PhysicalBlock] = {}
        
        # Mapping from request_id -> BlockTable
        self.active_tables: Dict[str, BlockTable] = {}

    def allocate_sequence(self, request_id: str, initial_prompt_len: int) -> BlockTable:
        """Allocates initial physical blocks needed to store a prompt's KV cache."""
        blocks_needed = (initial_prompt_len + self.block_size - 1) // self.block_size
        if len(self.free_block_ids) < blocks_needed:
            raise MemoryError(
                f"Out of Memory: Requested {blocks_needed} blocks, only {len(self.free_block_ids)} free."
            )

        table = BlockTable(request_id=request_id, block_size=self.block_size)
        table.num_tokens = initial_prompt_len

        for _ in range(blocks_needed):
            block_id = self.free_block_ids.pop(0)
            phys_block = PhysicalBlock(block_id=block_id, block_size=self.block_size)
            phys_block.ref_count = 1
            self.allocated_blocks[block_id] = phys_block
            table.physical_block_ids.append(block_id)

        self.active_tables[request_id] = table
        return table

    def append_token(self, request_id: str) -> (int, int):
        """
        Allocates memory for a newly generated token.
        Only fetches a new physical block if the current block is completely full.
        """
        table = self.active_tables.get(request_id)
        if not table:
            raise KeyError(f"No active BlockTable found for {request_id}")

        curr_token_idx = table.num_tokens
        logical_block_idx = curr_token_idx // self.block_size
        offset = curr_token_idx % self.block_size

        if offset == 0:
            # Need a new physical block
            if not self.free_block_ids:
                raise MemoryError("Out of Memory: No free physical blocks available for new token.")
            block_id = self.free_block_ids.pop(0)
            phys_block = PhysicalBlock(block_id=block_id, block_size=self.block_size)
            phys_block.ref_count = 1
            self.allocated_blocks[block_id] = phys_block
            table.physical_block_ids.append(block_id)

        table.num_tokens += 1
        phys_block_id = table.physical_block_ids[logical_block_idx]
        return phys_block_id, offset

    def write_kv(
        self,
        layer_idx: int,
        phys_block_id: int,
        offset: int,
        key_token: torch.Tensor,
        value_token: torch.Tensor,
    ):
        """
        Writes a single token's Key and Value vectors into the preallocated physical block.
        key_token / value_token shape: [num_heads, head_dim]
        """
        self.kv_cache_pool[layer_idx, phys_block_id, 0, offset] = key_token
        self.kv_cache_pool[layer_idx, phys_block_id, 1, offset] = value_token

    def free_sequence(self, request_id: str):
        """Reclaims all physical memory blocks assigned to a completed sequence."""
        table = self.active_tables.pop(request_id, None)
        if not table:
            return

        for block_id in table.physical_block_ids:
            if block_id in self.allocated_blocks:
                self.allocated_blocks[block_id].ref_count -= 1
                if self.allocated_blocks[block_id].ref_count <= 0:
                    del self.allocated_blocks[block_id]
                    self.free_block_ids.append(block_id)

    def get_memory_stats(self) -> Dict[str, float]:
        """Calculates memory utilization and fragmentation metrics."""
        total_blocks = self.total_num_blocks
        used_blocks = len(self.allocated_blocks)
        free_blocks = len(self.free_block_ids)

        # Calculate exact internal fragmentation (wasted slots in the last block of sequences)
        total_tokens_stored = sum(t.num_tokens for t in self.active_tables.values())
        total_slots_allocated = used_blocks * self.block_size
        internal_fragmentation_pct = (
            ((total_slots_allocated - total_tokens_stored) / total_slots_allocated * 100.0)
            if total_slots_allocated > 0 else 0.0
        )

        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": free_blocks,
            "utilization_pct": (used_blocks / total_blocks) * 100.0,
            "internal_fragmentation_pct": internal_fragmentation_pct,
        }
