import time
from typing import List, Dict, Optional, Callable
from .block_allocator import BlockAllocator, BlockTable

class InferenceRequest:
    """Represents an active client inference request."""
    def __init__(
        self,
        request_id: str,
        prompt_tokens: List[int],
        max_new_tokens: int = 64,
        eos_token_id: int = 2,
    ):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens: List[int] = []
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.block_table: Optional[BlockTable] = None
        self.is_finished = False
        self.arrival_time = time.time()
        self.finish_time: Optional[float] = None

class ContinuousBatchScheduler:
    """
    Iteration-Level Continuous Batching Scheduler.
    Dynamically evicts finished requests and admits waiting requests at every forward-pass iteration.
    """
    def __init__(self, allocator: BlockAllocator, max_batch_size: int = 8):
        self.allocator = allocator
        self.max_batch_size = max_batch_size
        self.waiting_queue: List[InferenceRequest] = []
        self.running_batch: List[InferenceRequest] = []
        self.completed_requests: List[InferenceRequest] = []

    def add_request(self, request: InferenceRequest):
        """Adds a new request to the pending queue."""
        self.waiting_queue.append(request)

    def schedule_next_iteration(self) -> List[InferenceRequest]:
        """
        Admits waiting requests into running batch if slots and memory blocks are available.
        """
        while self.waiting_queue and len(self.running_batch) < self.max_batch_size:
            candidate = self.waiting_queue[0]
            blocks_needed = (len(candidate.prompt_tokens) + self.allocator.block_size - 1) // self.allocator.block_size
            
            if len(self.allocator.free_block_ids) >= blocks_needed:
                req = self.waiting_queue.pop(0)
                # Allocate physical memory blocks for this sequence
                req.block_table = self.allocator.allocate_sequence(
                    request_id=req.request_id,
                    initial_prompt_len=len(req.prompt_tokens)
                )
                self.running_batch.append(req)
            else:
                # Memory full, wait for active requests to finish
                break

        return self.running_batch

    def step(self, model_forward_step_fn: Callable[[List[InferenceRequest]], List[int]]):
        """
        Executes a single continuous batching iteration:
        1. Schedule active requests
        2. Run one model forward pass across all active requests
        3. Check termination (EOS / max tokens) and evict finished sequences immediately
        """
        active_batch = self.schedule_next_iteration()
        if not active_batch:
            return []

        # Run forward step (predicts next token for each sequence in the active batch)
        next_tokens = model_forward_step_fn(active_batch)

        still_running = []
        finished_this_step = []

        for req, token_id in zip(active_batch, next_tokens):
            req.output_tokens.append(token_id)
            
            # Check termination condition
            if token_id == req.eos_token_id or len(req.output_tokens) >= req.max_new_tokens:
                req.is_finished = True
                req.finish_time = time.time()
                # Immediately reclaim physical memory blocks!
                self.allocator.free_sequence(req.request_id)
                self.completed_requests.append(req)
                finished_this_step.append(req)
            else:
                # Increment token count in paged memory
                self.allocator.append_token(req.request_id)
                still_running.append(req)

        self.running_batch = still_running
        return finished_this_step

    def has_pending_work(self) -> bool:
        """Returns True if there are requests currently running or waiting."""
        return bool(self.running_batch or self.waiting_queue)
